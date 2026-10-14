"""Train dual-role: freeze S5 recon; train plain classification line + presence head.

Design
------
  Line A (skip_net / S5): frozen SEG mask provider.
  Line B (cls_net / none): no-skip decoder line; LesionPresenceHead on μ → yes/no.
  Gate at infer: keep S5 mask iff P(present) >= tau (VAL-picked).

Loss (no grad into S5)
------
  L = BCE(head) + λ_plain * DiceCE(plain SEG) + β * KL(cls μ)

Phases
------
  1) head-only (cls frozen) — warm start
  2) unfreeze cls encoder+decoder with small LR + head

Does not overwrite none/ or s5_concat/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from train_lesion_presence import (
    PIXEL_THR,
    _prep,
    metrics_bundle,
)
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    combined_binary_loss_sigmoid,
    safe_load,
    safe_print,
    split_dice_stats,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
EMPTYFP_LAST = RESULT_DIR / "none_emptyfp" / "seg_last.pt"
EMPTYFP_BEST = RESULT_DIR / "none_emptyfp" / "seg_best.pt"
OUT_DEFAULT = RESULT_DIR / "dual_role_train"
# original: none SEG_BATCH=16, s5 uses 8 (8GB). Dual keeps S5 on device → batch=8 only.
BATCH = 8
WARM_EPOCHS = 8
FT_EPOCHS = 40
LR_HEAD = 1e-3
LR_CLS = 3e-5
LAMBDA_PLAIN = 0.25
POS_WEIGHT = 1.5
ES_PATIENCE = 12
SENS_DROP = 0.02
NE_DROP = 0.02

CLS_INITS = {
    "none": NONE_BEST,
    "emptyfp": EMPTYFP_BEST if EMPTYFP_BEST.exists() else EMPTYFP_LAST,
}


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_init(model: DualRoleVAE, cls_ckpt: Path, cls_name: str) -> None:
    st = safe_load(S5_BEST, map_location="cpu")
    model.skip_net.load_state_dict(st["model"])
    del st
    st = safe_load(cls_ckpt, map_location="cpu")
    key = "model" if "model" in st else "state_dict"
    model.cls_net.load_state_dict(st[key])
    del st
    safe_print(f"init: skip←s5_concat, cls←{cls_name} ({cls_ckpt.name}), head=random")


def step_loss(model: DualRoleVAE, flair, seg, bce: nn.Module, lambda_plain: float):
    """Train cls line only — S5 is frozen and not forwarded here."""
    flair, seg = _prep(flair, seg)
    plain_logits, presence_logit, mu, logvar = model.cls_forward_det(flair)
    y = (seg.flatten(1).sum(dim=1) > 0).float()
    loss_bce = bce(presence_logit, y)
    if lambda_plain > 0:
        dice_p, _, _ = combined_binary_loss_sigmoid(plain_logits, seg)
    else:
        dice_p = plain_logits.new_zeros(())
    kl = 1 + logvar - torch.square(mu) - torch.exp(logvar)
    kl = (-0.5 * kl).sum(dim=-1).mean()
    loss = loss_bce + lambda_plain * dice_p + BETA_LOCAL * kl
    return loss, float(loss_bce.item()), float(dice_p.item()), float(y.mean().item())


@torch.inference_mode()
def val_auroc_only(model: DualRoleVAE, loader) -> float:
    """Light VAL like presence train: scores from cls μ+head only (no S5)."""
    model.eval()
    model.gate_thr = None
    ys, yt = [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        _plain, presence_logit, _mu, _lv = model.cls_forward_det(flair)
        y = (seg.flatten(1).sum(dim=1) > 0)
        ys.append(torch.sigmoid(presence_logit).detach().cpu().numpy())
        yt.append(y.detach().cpu().numpy())
    return binary_auroc(np.concatenate(yt), np.concatenate(ys))


@torch.inference_mode()
def collect_split(model: DualRoleVAE, loader):
    """Full dual collect for final gate judge only."""
    model.eval()
    model.gate_thr = None
    bags = {
        "gt": [], "mass_s5": [], "dice_s5": [],
        "mass_plain": [], "dice_plain": [],
        "cls_prob": [],
    }
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        skip_l, plain_l, presence_logit, _, _ = model(flair, return_parts=True)
        prob_s = torch.sigmoid(skip_l)
        pred_s = (prob_s >= PIXEL_THR).float()
        prob_p = torch.sigmoid(plain_l)
        pred_p = (prob_p >= PIXEL_THR).float()
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        gs = seg_b.sum(dim=(1, 2, 3))
        ps_s = pred_s.sum(dim=(1, 2, 3))
        ps_p = pred_p.sum(dim=(1, 2, 3))
        inter_s = (pred_s * seg_b).sum(dim=(1, 2, 3))
        inter_p = (pred_p * seg_b).sum(dim=(1, 2, 3))
        dice_s = (2 * inter_s + 1e-6) / (ps_s + gs + 1e-6)
        dice_p = (2 * inter_p + 1e-6) / (ps_p + gs + 1e-6)
        both_s = (ps_s == 0) & (gs == 0)
        both_p = (ps_p == 0) & (gs == 0)
        dice_s = torch.where(both_s, torch.ones_like(dice_s), dice_s)
        dice_p = torch.where(both_p, torch.ones_like(dice_p), dice_p)
        p = torch.sigmoid(presence_logit)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass_s5"].append(ps_s.cpu().numpy())
        bags["dice_s5"].append(dice_s.cpu().numpy())
        bags["mass_plain"].append(ps_p.cpu().numpy())
        bags["dice_plain"].append(dice_p.cpu().numpy())
        bags["cls_prob"].append(p.detach().cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in bags.items()}


def eval_gate(bag, thr: float) -> dict:
    keep = bag["cls_prob"] >= thr
    m = metrics_bundle(bag["gt"], bag["mass_s5"], bag["dice_s5"], keep)
    yt = bag["gt"] > 0
    yp = bag["cls_prob"] >= thr
    m["presence_acc"] = float((yp == yt).mean())
    m["presence_recall"] = float((yp & yt).sum() / max(1, yt.sum()))
    m["presence_auroc"] = binary_auroc(yt, bag["cls_prob"])
    m["thr"] = thr
    return m


def pick_thr(val: dict, floor_ne: float, floor_sens: float):
    best = None
    rows = []
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_gate(val, thr)
        ok = m["nonempty_dice"] >= floor_ne - NE_DROP and m["sensitivity"] >= floor_sens - SENS_DROP
        rows.append({**{k: m[k] for k in (
            "overall_dice", "nonempty_dice", "empty_fp_rate", "sensitivity",
            "presence_acc", "presence_recall", "presence_auroc", "TT", "FT", "TF", "FF",
        )}, "thr": thr, "ok": ok})
        if ok and (best is None or m["empty_fp_rate"] < best["empty_fp_rate"]):
            best = dict(m)
    return best, rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def flat_metrics(m: dict, name: str) -> dict:
    return {
        "name": name,
        "overall_dice": m["overall_dice"],
        "nonempty_dice": m["nonempty_dice"],
        "empty_fp_rate": m["empty_fp_rate"],
        "sensitivity": m["sensitivity"],
        "TT": m.get("TT"), "FT": m.get("FT"), "TF": m.get("TF"), "FF": m.get("FF"),
        "presence_auroc": m.get("presence_auroc"),
        "presence_acc": m.get("presence_acc"),
        "thr": m.get("thr"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-epochs", type=int, default=WARM_EPOCHS)
    ap.add_argument("--ft-epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--lambda-plain", type=float, default=LAMBDA_PLAIN)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument(
        "--cls-init", choices=sorted(CLS_INITS.keys()), default="none",
        help="classification-line init: none | emptyfp (best empty-FP FT)",
    )
    ap.add_argument(
        "--out", type=str, default="",
        help="output dir under results/bp_original (default dual_role_train[_emptyfp])",
    )
    args = ap.parse_args()

    cls_ckpt = CLS_INITS[args.cls_init]
    if args.out:
        OUT = RESULT_DIR / args.out
    else:
        OUT = OUT_DEFAULT if args.cls_init == "none" else RESULT_DIR / f"dual_role_train_{args.cls_init}"

    for p in (S5_BEST, cls_ckpt):
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_print(f"OUT={OUT} cls_init={args.cls_init}")

    data = datasets.bp_train_30(batch_size=args.batch, num_workers=NUM_WORKERS)
    train_loader = data.train_loader
    val_loader = data.val_loader
    test_loader = data.test_loader
    safe_print(
        f"DEVICE={DEVICE} n_train={len(data.train)} n_val={len(data.val)} "
        f"batch={args.batch} (original none=16, s5=8 → dual uses 8)"
    )

    model = DualRoleVAE(base=BASE).to(DEVICE)
    load_init(model, cls_ckpt, args.cls_init)
    model.freeze_skip()

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

    history = []
    best_auroc = -1.0
    best_path = ckpt_dir / "dual_role_best_auroc.pt"
    last_path = ckpt_dir / "dual_role_last.pt"
    es_wait = 0

    def run_epoch(ep: int, phase: str, opt, lambda_plain: float):
        model.train()
        model.freeze_skip()
        model.skip_net.eval()
        if phase == "warm":
            model.cls_net.eval()
        tot = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "n": 0}
        for batch in train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, lb, ld, _ = step_loss(model, flair, seg, bce, lambda_plain)
            loss.backward()
            opt.step()
            tot["loss"] += float(loss.item())
            tot["bce"] += lb
            tot["dice"] += ld
            tot["n"] += 1
        n = max(1, tot["n"])
        return {k: (tot[k] / n if k != "n" else tot[k]) for k in tot}

    # ---- phase 1: head only ----
    model.set_cls_trainable(encoder=False, decoder=False, head=True)
    opt = torch.optim.AdamW(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    safe_print(f"phase1 warm head-only epochs={args.warm_epochs} device={DEVICE}")
    for ep in range(1, args.warm_epochs + 1):
        t0 = time.time()
        tr = run_epoch(ep, "warm", opt, lambda_plain=0.0)
        auroc = val_auroc_only(model, val_loader)
        rec = {
            "phase": "warm", "epoch": ep,
            "train_loss": tr["loss"], "train_bce": tr["bce"], "train_plain_dice_loss": tr["dice"],
            "val_auroc": auroc,
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[warm {ep}/{args.warm_epochs}] loss={tr['loss']:.4f} bce={tr['bce']:.4f} "
            f"auroc={auroc:.4f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(),
            "epoch": ep, "phase": "warm", "val_auroc": auroc,
            "cls_init": args.cls_init,
            "cls_ckpt": str(cls_ckpt),
            "batch": args.batch,
            "design": "dual_role_s5_frozen_cls_head",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es_wait = 0
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es_wait += 1

    # ---- phase 2: FT cls encoder+decoder + head ----
    model.set_cls_trainable(encoder=True, decoder=True, head=True)
    params = [
        {"params": model.head.parameters(), "lr": LR_HEAD * 0.5},
        {"params": model.cls_net.parameters(), "lr": LR_CLS},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    es_wait = 0
    safe_print(
        f"phase2 FT cls+head epochs={args.ft_epochs} λ_plain={args.lambda_plain} "
        f"lr_cls={LR_CLS} cls_init={args.cls_init}"
    )
    # reload best warm weights
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_skip()
    model.set_cls_trainable(encoder=True, decoder=True, head=True)
    best_auroc = float(st.get("val_auroc", best_auroc))

    for ep in range(1, args.ft_epochs + 1):
        t0 = time.time()
        tr = run_epoch(ep, "ft", opt, lambda_plain=args.lambda_plain)
        auroc = val_auroc_only(model, val_loader)
        rec = {
            "phase": "ft", "epoch": ep,
            "train_loss": tr["loss"], "train_bce": tr["bce"], "train_plain_dice_loss": tr["dice"],
            "val_auroc": auroc,
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[ft {ep}/{args.ft_epochs}] loss={tr['loss']:.4f} bce={tr['bce']:.4f} "
            f"auroc={auroc:.4f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(),
            "epoch": ep, "phase": "ft", "val_auroc": auroc,
            "lambda_plain": args.lambda_plain,
            "cls_init": args.cls_init,
            "cls_ckpt": str(cls_ckpt),
            "batch": args.batch,
            "design": "dual_role_s5_frozen_cls_head",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es_wait = 0
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es_wait += 1
            if es_wait >= ES_PATIENCE:
                safe_print(f"early stop FT at ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # ---- final judge (S5 gate metrics once) ----
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_skip()
    model.eval()

    val = collect_split(model, val_loader)
    test = collect_split(model, test_loader)

    s5_v = metrics_bundle(val["gt"], val["mass_s5"], val["dice_s5"], np.ones(len(val["gt"]), bool))
    s5_t = metrics_bundle(test["gt"], test["mass_s5"], test["dice_s5"], np.ones(len(test["gt"]), bool))
    plain_v = metrics_bundle(val["gt"], val["mass_plain"], val["dice_plain"], np.ones(len(val["gt"]), bool))
    plain_t = metrics_bundle(test["gt"], test["mass_plain"], test["dice_plain"], np.ones(len(test["gt"]), bool))

    # floor: keep ne near plain C0 (role: raise ne via S5 under presence constraint)
    # also report pick vs S5 sens floor soft
    best_plain, curve = pick_thr(val, plain_v["nonempty_dice"], plain_v["sensitivity"])
    write_csv(OUT / "thr_curve_val.csv", curve)

    rows = [
        flat_metrics({**s5_v, "presence_auroc": None, "presence_acc": None, "thr": None}, "VAL_S5_raw"),
        flat_metrics({**plain_v, "presence_auroc": None, "presence_acc": None, "thr": None}, "VAL_plain_raw"),
        flat_metrics({**s5_t, "presence_auroc": None, "presence_acc": None, "thr": None}, "TEST_S5_raw"),
        flat_metrics({**plain_t, "presence_auroc": None, "presence_acc": None, "thr": None}, "TEST_plain_raw"),
    ]

    picks = {}
    if best_plain is not None:
        thr = float(best_plain["thr"])
        tv = eval_gate(val, thr)
        tt = eval_gate(test, thr)
        rows.append(flat_metrics(tv, f"VAL_gate@plainFloor_{thr:.2f}"))
        rows.append(flat_metrics(tt, f"TEST_gate@plainFloor_{thr:.2f}"))
        picks["vs_plain_floor"] = {"thr": thr, "val": flat_metrics(tv, "val"), "test": flat_metrics(tt, "test")}
        safe_print(
            f"PICK vs plain thr={thr:.2f} "
            f"VAL ne={tv['nonempty_dice']:.4f} fp={tv['empty_fp_rate']:.3f} "
            f"TEST ne={tt['nonempty_dice']:.4f} fp={tt['empty_fp_rate']:.3f} "
            f"auroc={tt['presence_auroc']:.4f}"
        )
    else:
        safe_print("no thr under plain floor")

    # also fixed 0.20 / 0.50 and best-FP unconstrained among ne >= 0.55
    for thr in (0.20, 0.30, 0.50):
        rows.append(flat_metrics(eval_gate(val, thr), f"VAL_gate@{thr:.2f}"))
        rows.append(flat_metrics(eval_gate(test, thr), f"TEST_gate@{thr:.2f}"))

    # unconstrained: max ne - 0.5*fp style on VAL
    best_bal = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_gate(val, thr)
        score = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
        if best_bal is None or score > best_bal["score"]:
            best_bal = {**m, "score": score}
    if best_bal is not None:
        thr = float(best_bal["thr"])
        tt = eval_gate(test, thr)
        rows.append(flat_metrics(best_bal, f"VAL_gate@bal_{thr:.2f}"))
        rows.append(flat_metrics(tt, f"TEST_gate@bal_{thr:.2f}"))
        picks["balanced"] = {"thr": thr, "val": flat_metrics(best_bal, "val"), "test": flat_metrics(tt, "test")}
        safe_print(
            f"PICK bal thr={thr:.2f} "
            f"TEST ne={tt['nonempty_dice']:.4f} fp={tt['empty_fp_rate']:.3f}"
        )

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": {
            "line_A": "S5 skip decoder — frozen recon / SEG mask",
            "line_B": f"plain no-skip decoder (init {args.cls_init}) + LesionPresenceHead(μ) — presence",
            "cls_init": args.cls_init,
            "cls_ckpt": str(cls_ckpt),
            "infer": "keep S5 mask iff P(head)>=tau else blank",
            "loss": "BCE(head) + λ_plain*DiceCE(plain) + β KL; no grad into S5",
            "phases": ["warm head-only", "FT cls encoder+decoder+head"],
        },
        "best_ckpt": str(best_path),
        "best_val_auroc": best_auroc,
        "picks": picks,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Dual-role retrain (S5 recon + cls={args.cls_init} + head)",
        "",
        report["design"]["line_A"],
        report["design"]["line_B"],
        report["design"]["infer"],
        "",
        f"best VAL AUROC (ckpt)={best_auroc:.4f}",
        "",
        "## TEST",
    ]
    for r in rows:
        if r["name"].startswith("TEST"):
            lines.append(
                f"- {r['name']}: ne={r['nonempty_dice']:.4f} fp={r['empty_fp_rate']:.3f}"
                + (f" auroc={r['presence_auroc']:.4f}" if r.get("presence_auroc") is not None else "")
            )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
