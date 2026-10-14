"""Train: frozen BT encoder → μ+skips → S5 decoder (SEG); skips → ClsHead (presence).

Phase1: train decoder_skip (SEG), encoder frozen
Phase2: freeze decoder_skip; train SkipPresenceHead on encoder skips (not μ)
Infer: keep skip mask iff P≥τ
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
from models_bt_shared_dual import BTSharedDualVAE, load_bt_shared
from train_lesion_presence import PIXEL_THR, _prep, metrics_bundle
from train_original_bp import (
    BASE,
    BT_BEST,
    BT_LAST,
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
OUT = RESULT_DIR / "bt_shared_dual"
BATCH = 8
SEG_EPOCHS = 60
HEAD_EPOCHS = 40
LR_SKIP = 1e-4
LR_HEAD = 1e-3
POS_WEIGHT = 1.5
ES_SEG = 15
ES_HEAD = 12
NE_DROP = 0.02
SENS_DROP = 0.02


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_compatible(model: BTSharedDualVAE, ckpt: dict) -> list[str]:
    """Load encoder + decoder_skip (+ head if shapes match) from old/new ckpts."""
    src = ckpt["model"] if "model" in ckpt else ckpt
    dst = model.state_dict()
    filtered = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = sorted(filtered.keys())
    safe_print(f"partial load n={len(loaded)} missing={len(missing)} unexpected={len(unexpected)}")
    return loaded


def seg_step(model, flair, seg):
    flair, seg = _prep(flair, seg)
    skip_l, _logit, _mu, _lv, _sk = model.forward_parts(flair)
    dice_s, _, _ = combined_binary_loss_sigmoid(skip_l, seg)
    return dice_s, skip_l


def head_step(model, flair, seg, bce):
    flair, seg = _prep(flair, seg)
    _skip, logit, _mu, _lv, _sk = model.forward_parts(flair)
    y = (seg.flatten(1).sum(dim=1) > 0).float()
    loss_bce = bce(logit, y)
    return loss_bce, float(loss_bce.item())


@torch.inference_mode()
def eval_skip_seg(model, loader):
    model.eval()
    dices, gt, pred = [], [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        loss, skip_l = seg_step(model, flair, seg)
        loss_sum += float(loss.item())
        n += 1
        dices.append(batch_dice(skip_l, seg))
        ps, gs = batch_pred_gt_sums(skip_l, seg)
        pred.append(ps)
        gt.append(gs)
    st = split_dice_stats(np.concatenate(dices), np.concatenate(gt), np.concatenate(pred))
    st["val_loss"] = loss_sum / max(1, n)
    return st


@torch.inference_mode()
def val_auroc(model, loader):
    model.eval()
    ys, yt = [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        _s, logit, _, _, _ = model.forward_parts(flair)
        ys.append(torch.sigmoid(logit).cpu().numpy())
        yt.append((seg.flatten(1).sum(1) > 0).cpu().numpy())
    return binary_auroc(np.concatenate(yt), np.concatenate(ys))


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    bags = {"gt": [], "mass_s5": [], "dice_s5": [], "cls_prob": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        skip_l, logit, _, _, _ = model.forward_parts(flair)
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        prob = torch.sigmoid(skip_l)
        pred = (prob >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass_s5"].append(ps.cpu().numpy())
        bags["dice_s5"].append(dice.cpu().numpy())
        bags["cls_prob"].append(torch.sigmoid(logit).cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in bags.items()}


def eval_gate(bag, thr):
    keep = bag["cls_prob"] >= thr
    m = metrics_bundle(bag["gt"], bag["mass_s5"], bag["dice_s5"], keep)
    yt = bag["gt"] > 0
    yp = keep
    m["presence_recall"] = float((yp & yt).sum() / max(1, int(yt.sum())))
    m["presence_auroc"] = binary_auroc(yt, bag["cls_prob"])
    m["thr"] = thr
    return m


def pick_s5_floor(val, floor_ne, floor_sens):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_gate(val, thr)
        ok = m["nonempty_dice"] >= floor_ne - NE_DROP and m["sensitivity"] >= floor_sens - SENS_DROP
        if ok and (best is None or m["empty_fp_rate"] < best["empty_fp_rate"]):
            best = dict(m)
    return best


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-epochs", type=int, default=SEG_EPOCHS)
    ap.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument(
        "--resume-seg",
        type=str,
        default="",
        help="path to seg ckpt; continue phase1 then phase2 (partial load OK)",
    )
    ap.add_argument(
        "--skip-seg",
        action="store_true",
        help="skip phase1; load --resume-seg and train head only",
    )
    args = ap.parse_args()

    bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
    if not bt_ckpt.exists():
        raise FileNotFoundError(bt_ckpt)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(batch_size=args.batch, num_workers=NUM_WORKERS)
    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} BT={bt_ckpt.name} "
        f"n={len(data.train)}/{len(data.val)}/{len(data.test)}"
    )
    safe_print("design: BT enc(freeze) → μ+skips→S5 SEG | skips→ClsHead (no μ)")

    model = BTSharedDualVAE(base=BASE).to(DEVICE)
    load_bt_shared(
        model,
        bt_ckpt,
        safe_load(S5_BEST, map_location="cpu") if S5_BEST.exists() else None,
    )
    model.freeze_encoder()

    best_seg_path = ckpt_dir / "bt_shared_best_seg.pt"
    best_head_path = ckpt_dir / "bt_shared_best_auroc.pt"
    last_path = ckpt_dir / "bt_shared_last.pt"
    resume_path = Path(args.resume_seg) if args.resume_seg else None
    if resume_path and resume_path.exists():
        load_compatible(model, safe_load(resume_path, map_location="cpu"))
        safe_print(f"resumed weights from {resume_path}")
    else:
        safe_print("encoder←BT frozen; skip_dec←s5 init; head=random")

    history = []
    best_val = float("inf")
    best_ne = -1.0

    # ----- Phase 1: train S5 decoder -----
    if not args.skip_seg:
        model.set_trainable(skip_dec=True, head=False)
        opt = torch.optim.AdamW(
            [p for p in model.decoder_skip.parameters() if p.requires_grad],
            lr=LR_SKIP, weight_decay=1e-4,
        )
        sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        es = 0
        # seed best from current eval if resuming
        if resume_path and resume_path.exists():
            st0 = eval_skip_seg(model, data.val_loader)
            best_val = float(st0["val_loss"])
            best_ne = float(st0["mean_dice_nonempty"] or 0.0)
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": 0, "phase": "seg",
                    "val_loss": best_val, "val_ne": best_ne,
                    "design": "bt_mu_s5_cls_skips",
                    "bt_ckpt": str(bt_ckpt),
                },
                best_seg_path,
            )
            safe_print(f"resume baseline val_loss={best_val:.4f} ne={best_ne:.4f}")

        safe_print(f"phase1 SEG decoder_skip ep≤{args.seg_epochs}")
        for ep in range(1, args.seg_epochs + 1):
            model.train()
            model.freeze_encoder()
            model.head.eval()
            tr, ntr = 0.0, 0
            t0 = time.time()
            for batch in data.train_loader:
                flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss, _ = seg_step(model, flair, seg)
                loss.backward()
                opt.step()
                tr += float(loss.item())
                ntr += 1
            st = eval_skip_seg(model, data.val_loader)
            ne = float(st["mean_dice_nonempty"] or 0.0)
            vl = float(st["val_loss"])
            rec = {
                "phase": "seg", "epoch": ep,
                "train_loss": tr / max(1, ntr),
                "val_loss": vl,
                "val_dice": float(st["dice_mean"]),
                "val_ne": ne,
                "val_fp": float(st["empty_fp_rate"]),
                "sec": time.time() - t0,
            }
            history.append(rec)
            safe_print(
                f"[seg {ep}/{args.seg_epochs}] tr={rec['train_loss']:.4f} val={vl:.4f} "
                f"ne={ne:.4f} fp={rec['val_fp']:.3f} ({rec['sec']:.1f}s)"
            )
            sch.step(vl)
            payload = {
                "model": model.state_dict(),
                "epoch": ep, "phase": "seg",
                "val_loss": vl, "val_ne": ne,
                "design": "bt_mu_s5_cls_skips",
                "bt_ckpt": str(bt_ckpt),
            }
            atomic_torch_save(payload, last_path)
            if vl < best_val - 1e-4:
                best_val = vl
                best_ne = ne
                atomic_torch_save(payload, best_seg_path)
                es = 0
                safe_print(f"  * best seg val_loss={vl:.4f} ne={ne:.4f}")
            else:
                es += 1
                if es >= ES_SEG:
                    safe_print(f"early stop SEG @ ep {ep}")
                    break
    else:
        if not best_seg_path.exists() and not (resume_path and resume_path.exists()):
            raise FileNotFoundError("need --resume-seg or existing best_seg for --skip-seg")
        src = resume_path if resume_path and resume_path.exists() else best_seg_path
        load_compatible(model, safe_load(src, map_location="cpu"))
        st0 = eval_skip_seg(model, data.val_loader)
        best_val = float(st0["val_loss"])
        best_ne = float(st0["mean_dice_nonempty"] or 0.0)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "epoch": 0, "phase": "seg",
                "val_loss": best_val, "val_ne": best_ne,
                "design": "bt_mu_s5_cls_skips",
                "bt_ckpt": str(bt_ckpt),
            },
            best_seg_path,
        )
        safe_print(f"skip-seg baseline val_loss={best_val:.4f} ne={best_ne:.4f}")

    # ----- Phase 2: freeze S5; train head on skips -----
    st = safe_load(best_seg_path, map_location="cpu")
    load_compatible(model, st)
    model.freeze_encoder()
    for p in model.decoder_skip.parameters():
        p.requires_grad = False
    model.decoder_skip.eval()
    model.set_trainable(skip_dec=False, head=True)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    best_auroc = -1.0
    es = 0
    safe_print(f"phase2 SkipPresenceHead(skips) ep≤{args.head_epochs}")
    for ep in range(1, args.head_epochs + 1):
        model.train()
        model.freeze_encoder()
        model.decoder_skip.eval()
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, lb = head_step(model, flair, seg, bce)
            loss.backward()
            opt.step()
            tr += lb
            ntr += 1
        auroc = val_auroc(model, data.val_loader)
        rec = {
            "phase": "head", "epoch": ep,
            "train_bce": tr / max(1, ntr), "val_auroc": auroc,
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[head {ep}/{args.head_epochs}] bce={rec['train_bce']:.4f} "
            f"auroc={auroc:.4f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(), "epoch": ep, "phase": "head",
            "val_auroc": auroc, "design": "bt_mu_s5_cls_skips",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_head_path)
            es = 0
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= ES_HEAD:
                safe_print(f"early stop head @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # ----- Final judge -----
    st = safe_load(best_head_path, map_location="cpu")
    load_compatible(model, st)
    model.freeze_encoder()
    model.eval()
    val = collect(model, data.val_loader)
    test = collect(model, data.test_loader)
    ones = np.ones(len(test["gt"]), dtype=bool)
    raw_t = metrics_bundle(test["gt"], test["mass_s5"], test["dice_s5"], ones)
    raw_v = metrics_bundle(val["gt"], val["mass_s5"], val["dice_s5"], np.ones(len(val["gt"]), bool))

    rows = [{
        "name": "TEST_skip_raw",
        "overall_dice": raw_t["overall_dice"],
        "nonempty_dice": raw_t["nonempty_dice"],
        "empty_fp_rate": raw_t["empty_fp_rate"],
        "sensitivity": raw_t["sensitivity"],
        "thr": None,
    }]
    pick = pick_s5_floor(val, raw_v["nonempty_dice"], raw_v["sensitivity"])
    if pick:
        thr = float(pick["thr"])
        mt = eval_gate(test, thr)
        rows.append({
            "name": f"TEST_gate@S5floor_{thr:.2f}",
            "overall_dice": mt["overall_dice"],
            "nonempty_dice": mt["nonempty_dice"],
            "empty_fp_rate": mt["empty_fp_rate"],
            "sensitivity": mt["sensitivity"],
            "thr": thr,
            "presence_auroc": mt["presence_auroc"],
        })
        safe_print(
            f"PICK τ={thr:.2f} TEST ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} sens={mt['sensitivity']:.3f}"
        )
    for thr in [i / 100 for i in range(1, 16)] + [0.20, 0.30]:
        mt = eval_gate(test, thr)
        rows.append({
            "name": f"TEST_gate@{thr:.2f}",
            "overall_dice": mt["overall_dice"],
            "nonempty_dice": mt["nonempty_dice"],
            "empty_fp_rate": mt["empty_fp_rate"],
            "sensitivity": mt["sensitivity"],
            "thr": thr,
        })
        if thr in (0.05, 0.10, 0.15, 0.20):
            safe_print(
                f"  τ={thr:.2f} ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f}"
            )

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": {
            "encoder": "BT encoder frozen",
            "mu": "→ Decoder_skip (S5) only",
            "cls": "SkipPresenceHead(e0..e4 avg+max pool) — not μ",
            "infer": "keep skip mask iff P>=tau",
        },
        "best_seg_val_loss": best_val,
        "best_seg_val_ne": best_ne,
        "best_head_auroc": best_auroc,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# BT shared: μ→S5, cls←skips",
        "",
        "BT encoder (freeze) → μ+skips→S5 SEG | skips→ClsHead",
        f"SEG best VAL ne≈{best_ne:.4f} | head AUROC={best_auroc:.4f}",
        "",
        "## TEST",
    ]
    for r in rows:
        if "S5floor" in r["name"] or r["name"] == "TEST_skip_raw" or r["name"] in (
            "TEST_gate@0.05", "TEST_gate@0.10", "TEST_gate@0.15", "TEST_gate@0.20",
        ):
            lines.append(
                f"- {r['name']}: ne={r['nonempty_dice']:.4f} fp={r['empty_fp_rate']:.3f}"
            )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# BT: μ → S5 only; cls ← encoder skips",
            "",
            "```",
            "x → BT Encoder (frozen)",
            "      ├─ skips (e0..e4) → ClsHead → P → τ",
            "      └─ μ + skips → Decoder_skip (S5) → SEG",
            "```",
            "",
            "μ is not shared with the classifier — only feeds the S5 decoder.",
            f"BT ckpt: `{bt_ckpt}`",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
