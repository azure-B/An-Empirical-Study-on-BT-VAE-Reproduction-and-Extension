"""Train shared-encoder dual decoder.

Freeze encoder + skip decoder (S5). Train cls decoder + presence head.
Infer: keep S5 mask iff P(head(μ)) ≥ τ.

Usage:
  python train_shared_dual.py
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

from data_load_BT import datasets
from models_shared_dual import SharedDualVAE, load_shared_from_ckpts
from train_lesion_presence import PIXEL_THR, _prep, metrics_bundle
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    combined_binary_loss_sigmoid,
    safe_load,
    safe_print,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT = RESULT_DIR / "shared_dual"
BATCH = 8
WARM_EPOCHS = 8
FT_EPOCHS = 40
LR_HEAD = 1e-3
LR_CLS_DEC = 3e-5
LAMBDA_CLS = 0.25
POS_WEIGHT = 1.5
ES_PATIENCE = 12
SENS_DROP = 0.02
NE_DROP = 0.02


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


def step_loss(model: SharedDualVAE, flair, seg, bce: nn.Module, lambda_cls: float):
    flair, seg = _prep(flair, seg)
    # encoder+skip frozen: still need forward for μ; no grad into them via freeze
    skip_logits, cls_logits, presence_logit, mu, logvar = model.forward_parts(flair)
    y = (seg.flatten(1).sum(dim=1) > 0).float()
    loss_bce = bce(presence_logit, y)
    if lambda_cls > 0:
        dice_c, _, _ = combined_binary_loss_sigmoid(cls_logits, seg)
    else:
        dice_c = presence_logit.new_zeros(())
    # KL on μ only if encoder trainable — currently frozen, keep tiny for stability=0
    loss = loss_bce + lambda_cls * dice_c
    # detach skip from graph explicitly (should already be no-grad params)
    _ = skip_logits
    return loss, float(loss_bce.item()), float(dice_c.item())


@torch.inference_mode()
def val_auroc_only(model: SharedDualVAE, loader) -> float:
    model.eval()
    model.gate_thr = None
    ys, yt = [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        _s, _c, logit, _mu, _lv = model.forward_parts(flair)
        y = (seg.flatten(1).sum(dim=1) > 0)
        ys.append(torch.sigmoid(logit).cpu().numpy())
        yt.append(y.cpu().numpy())
    return binary_auroc(np.concatenate(yt), np.concatenate(ys))


@torch.inference_mode()
def collect_split(model: SharedDualVAE, loader):
    model.eval()
    model.gate_thr = None
    bags = {
        "gt": [], "mass_s5": [], "dice_s5": [],
        "mass_cls": [], "dice_cls": [], "cls_prob": [],
    }
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        skip_l, cls_l, logit, _, _ = model.forward_parts(flair)
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)

        def _dice_mass(logits):
            prob = torch.sigmoid(logits)
            pred = (prob >= PIXEL_THR).float()
            ps = pred.sum(dim=(1, 2, 3))
            gs = seg_b.sum(dim=(1, 2, 3))
            inter = (pred * seg_b).sum(dim=(1, 2, 3))
            dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
            both = (ps == 0) & (gs == 0)
            dice = torch.where(both, torch.ones_like(dice), dice)
            return dice, ps, gs

        d_s, m_s, gs = _dice_mass(skip_l)
        d_c, m_c, _ = _dice_mass(cls_l)
        p = torch.sigmoid(logit)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass_s5"].append(m_s.cpu().numpy())
        bags["dice_s5"].append(d_s.cpu().numpy())
        bags["mass_cls"].append(m_c.cpu().numpy())
        bags["dice_cls"].append(d_c.cpu().numpy())
        bags["cls_prob"].append(p.cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in bags.items()}


def eval_gate(bag, thr: float) -> dict:
    keep = bag["cls_prob"] >= thr
    m = metrics_bundle(bag["gt"], bag["mass_s5"], bag["dice_s5"], keep)
    yt = bag["gt"] > 0
    yp = bag["cls_prob"] >= thr
    m["presence_acc"] = float((yp == yt).mean())
    m["presence_recall"] = float((yp & yt).sum() / max(1, int(yt.sum())))
    m["presence_auroc"] = binary_auroc(yt, bag["cls_prob"])
    m["thr"] = thr
    return m


def pick_thr(val, floor_ne, floor_sens):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_gate(val, thr)
        ok = (
            m["nonempty_dice"] >= floor_ne - NE_DROP
            and m["sensitivity"] >= floor_sens - SENS_DROP
        )
        if ok and (best is None or m["empty_fp_rate"] < best["empty_fp_rate"]):
            best = dict(m)
    return best


def pick_cls_f1(val):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        yt = val["gt"] > 0
        yp = val["cls_prob"] >= thr
        tp = int((yp & yt).sum())
        fp = int((yp & ~yt).sum())
        fn = int((~yp & yt).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if best is None or f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "prec": prec, "rec": rec}
    return best


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def flat(m, name):
    return {
        "name": name,
        "overall_dice": m["overall_dice"],
        "nonempty_dice": m["nonempty_dice"],
        "empty_fp_rate": m["empty_fp_rate"],
        "sensitivity": m["sensitivity"],
        "TT": m.get("TT"), "FT": m.get("FT"), "TF": m.get("TF"), "FF": m.get("FF"),
        "presence_auroc": m.get("presence_auroc"),
        "presence_recall": m.get("presence_recall"),
        "thr": m.get("thr"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-epochs", type=int, default=WARM_EPOCHS)
    ap.add_argument("--ft-epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--lambda-cls", type=float, default=LAMBDA_CLS)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    for p in (S5_BEST, NONE_BEST):
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(batch_size=args.batch, num_workers=NUM_WORKERS)
    train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.test_loader
    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} "
        f"n={len(data.train)}/{len(data.val)}/{len(data.test)}"
    )
    safe_print("design: shared S5 encoder | decoder_skip(frozen) | decoder_cls+head(train)")

    model = SharedDualVAE(base=BASE).to(DEVICE)
    load_shared_from_ckpts(
        model,
        safe_load(S5_BEST, map_location="cpu"),
        safe_load(NONE_BEST, map_location="cpu"),
    )
    model.freeze_recon()
    safe_print("loaded encoder+skip←s5, decoder_cls←none; recon frozen")

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    history = []
    best_auroc = -1.0
    best_path = ckpt_dir / "shared_dual_best_auroc.pt"
    last_path = ckpt_dir / "shared_dual_last.pt"

    def run_epoch(phase, opt, lambda_cls):
        model.train()
        model.freeze_recon()
        if phase == "warm":
            model.decoder_cls.eval()
        tot = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "n": 0}
        for batch in train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, lb, ld = step_loss(model, flair, seg, bce, lambda_cls)
            loss.backward()
            opt.step()
            tot["loss"] += float(loss.item())
            tot["bce"] += lb
            tot["dice"] += ld
            tot["n"] += 1
        n = max(1, tot["n"])
        return {k: (tot[k] / n if k != "n" else tot[k]) for k in tot}

    # phase 1: head only
    model.set_cls_trainable(decoder=False, head=True)
    opt = torch.optim.AdamW(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    safe_print(f"phase1 warm head-only ep={args.warm_epochs}")
    for ep in range(1, args.warm_epochs + 1):
        t0 = time.time()
        tr = run_epoch("warm", opt, 0.0)
        auroc = val_auroc_only(model, val_loader)
        rec = {
            "phase": "warm", "epoch": ep,
            "train_loss": tr["loss"], "train_bce": tr["bce"],
            "val_auroc": auroc, "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[warm {ep}/{args.warm_epochs}] loss={tr['loss']:.4f} "
            f"auroc={auroc:.4f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(),
            "epoch": ep, "phase": "warm", "val_auroc": auroc,
            "design": "shared_encoder_dual_decoder",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            safe_print(f"  * best auroc={auroc:.4f}")

    # phase 2: FT cls decoder + head
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_recon()
    model.set_cls_trainable(decoder=True, head=True)
    best_auroc = float(st.get("val_auroc", best_auroc))
    opt = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": LR_HEAD * 0.5},
            {"params": model.decoder_cls.parameters(), "lr": LR_CLS_DEC},
        ],
        weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    es = 0
    safe_print(f"phase2 FT decoder_cls+head ep={args.ft_epochs} λ={args.lambda_cls}")
    for ep in range(1, args.ft_epochs + 1):
        t0 = time.time()
        tr = run_epoch("ft", opt, args.lambda_cls)
        auroc = val_auroc_only(model, val_loader)
        rec = {
            "phase": "ft", "epoch": ep,
            "train_loss": tr["loss"], "train_bce": tr["bce"], "train_cls_dice": tr["dice"],
            "val_auroc": auroc, "sec": time.time() - t0,
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
            "lambda_cls": args.lambda_cls,
            "design": "shared_encoder_dual_decoder",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es = 0
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= ES_PATIENCE:
                safe_print(f"early stop FT @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # final judge
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_recon()
    model.eval()
    val = collect_split(model, val_loader)
    test = collect_split(model, test_loader)

    ones_v = np.ones(len(val["gt"]), dtype=bool)
    ones_t = np.ones(len(test["gt"]), dtype=bool)
    s5_v = metrics_bundle(val["gt"], val["mass_s5"], val["dice_s5"], ones_v)
    s5_t = metrics_bundle(test["gt"], test["mass_s5"], test["dice_s5"], ones_t)

    # pick τ: (1) keep S5 ne within 0.02 (2) cls-F1
    best_ne = pick_thr(val, s5_v["nonempty_dice"], s5_v["sensitivity"])
    best_f1 = pick_cls_f1(val)
    rows = [
        flat({**s5_t, "presence_auroc": None, "presence_recall": None, "thr": None}, "TEST_S5_raw"),
    ]
    picks = {}
    if best_ne is not None:
        thr = float(best_ne["thr"])
        mt = eval_gate(test, thr)
        rows.append(flat(mt, f"TEST_gate@S5floor_{thr:.2f}"))
        picks["s5_floor"] = {"thr": thr, "test": flat(mt, "test")}
        safe_print(
            f"PICK S5floor τ={thr:.2f} TEST ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} sens={mt['sensitivity']:.3f}"
        )
    thr_f = float(best_f1["thr"])
    mt_f = eval_gate(test, thr_f)
    rows.append(flat(mt_f, f"TEST_gate@clsF1_{thr_f:.2f}"))
    picks["cls_f1"] = {"thr": thr_f, "val_f1": best_f1, "test": flat(mt_f, "test")}
    safe_print(
        f"PICK clsF1 τ={thr_f:.2f} TEST ne={mt_f['nonempty_dice']:.4f} "
        f"fp={mt_f['empty_fp_rate']:.3f} sens={mt_f['sensitivity']:.3f}"
    )
    for thr in (0.05, 0.10, 0.15, 0.20, 0.30):
        rows.append(flat(eval_gate(test, thr), f"TEST_gate@{thr:.2f}"))

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": {
            "encoder": "shared S5 encoder (frozen)",
            "decoder_skip": "S5 concat-skip (frozen) → SEG",
            "decoder_cls": "none-style no-skip (trainable) + LesionPresenceHead(μ)",
            "infer": "keep skip mask iff P(head)>=tau",
        },
        "best_val_auroc": best_auroc,
        "picks": picks,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Shared-encoder dual decoder",
        "",
        "1 encoder (S5) → decoder_skip (freeze) + decoder_cls+head (train)",
        f"best VAL AUROC={best_auroc:.4f}",
        "",
        "## TEST",
    ]
    for r in rows:
        lines.append(
            f"- {r['name']}: ne={r['nonempty_dice']:.4f} fp={r['empty_fp_rate']:.3f}"
            + (f" sens={r['sensitivity']:.3f}" if r.get("sensitivity") is not None else "")
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
