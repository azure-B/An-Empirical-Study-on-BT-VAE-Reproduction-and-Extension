"""1-encoder · 2-decoder + cls Head, TF/FP-aware training.

TF = FN (blank nonempty), FT = FP (keep empty). Hard-weight:
  - FT risk: GT-empty & SEG fires (mass>0)
  - TF risk: GT-nonempty & SEG fires (must not blank)

τ ∉ BCE; VAL calibrate τ (+ optional min-mass) → TEST.
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
from models_bt_cascade_cls import BTCascade1Enc2DecHead, load_from_cascade_focal
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)

FOCAL_SEG = RESULT_DIR / "bt_cascade_focal" / "checkpoints" / "bt_cascade_best_seg.pt"
OUT = RESULT_DIR / "cascade_1enc2dec_head_tffp"
WPROD_NE, WPROD_FP = 0.601, 0.305
BATCH = 8
EPOCHS = 40
LR = 1e-4
LR_HEAD = 1e-3
W_FT = 3.0   # empty + SEG blob → FP if keep
W_TF = 2.5   # nonempty + SEG blob → FN if blank
ES = 10


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, presence, *_ = model.forward_parts(flair)
        score = torch.sigmoid(presence)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        dice = torch.where((ps == 0) & (gs == 0), torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(score.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def eval_keep(bag, keep) -> dict:
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "TT": cm["TT"], "FT": cm["FT"], "TF": cm["TF"], "FF": cm["FF"],
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "tf_fp_sum": cm["TF"] + cm["FT"],
    }


def pick_gate(val, test, ne_floor=0.60):
    masses = [0, 80, 160, 320, 640, 1280, 2560]
    thrs = [i / 100.0 for i in range(1, 95)]
    best = None
    for thr in thrs:
        for mmin in masses:
            keep = (val["score"] >= thr) & (val["mass"] >= mmin)
            mv = eval_keep(val, keep)
            if mv["nonempty_dice"] + 1e-9 < ne_floor:
                continue
            # prefer low FP, then low TF+FT, then high ne
            key = (mv["empty_fp_rate"], mv["tf_fp_sum"], -mv["nonempty_dice"])
            if best is None or key < (
                best["empty_fp_rate"], best["tf_fp_sum"], -best["nonempty_dice"]
            ):
                best = {**mv, "thr": thr, "min_mass": mmin}
    if best is None:
        for thr in thrs:
            for mmin in masses:
                keep = (val["score"] >= thr) & (val["mass"] >= mmin)
                mv = eval_keep(val, keep)
                j = mv["nonempty_dice"] - 0.5 * mv["empty_fp_rate"] - 0.001 * mv["tf_fp_sum"]
                if best is None or j > best.get("_j", -1e9):
                    best = {**mv, "thr": thr, "min_mass": mmin, "_j": j}
    keep_t = (test["score"] >= best["thr"]) & (test["mass"] >= best["min_mass"])
    mt = eval_keep(test, keep_t)
    mt.update({"thr": best["thr"], "min_mass": best["min_mass"]})
    return best, mt


def val_select_score(bag) -> float:
    """Higher better: AUROC + Youden@0.5 favoring low TF+FT."""
    yt = bag["gt"] > 0
    auroc = binary_auroc(yt, bag["score"])
    keep = bag["score"] >= 0.5
    cm = conf_matrix(yt, keep)
    youden = cm["sensitivity"] + cm["specificity"] - 1.0
    # penalize TF+FT normalized
    err = (cm["TF"] + cm["FT"]) / max(1, len(yt))
    return float(auroc + 0.3 * youden - 0.5 * err)


def train_epoch(model, loader, opt, w_ft: float, w_tf: float):
    model.train()
    model.freeze_encoder()
    model.freeze_seg()
    total, n = 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        y = presence_y(seg)
        opt.zero_grad(set_to_none=True)
        seg_l, presence, *_ = model.forward_parts(flair)
        with torch.no_grad():
            pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
            pmass = pred.sum(dim=(1, 2, 3))
            seg_fire = pmass > 0
        # sample weights for TF / FP risks
        w = torch.ones_like(y)
        w = torch.where((y < 0.5) & seg_fire, torch.full_like(w, w_ft), w)
        w = torch.where((y > 0.5) & seg_fire, torch.full_like(w, w_tf), w)
        loss = (bce(presence, y) * w).mean()
        loss.backward()
        opt.step()
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--w-ft", type=float, default=W_FT)
    ap.add_argument("--w-tf", type=float, default=W_TF)
    ap.add_argument("--eval-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "cls_best_tffp.pt"

    (OUT / "DESIGN.md").write_text(
        "\n".join(
            [
                "# 1enc · 2dec + cls Head (TF/FP-aware)",
                "",
                "```text",
                "Encoder (freeze)",
                "  ├─ Decoder_seg (freeze) → mask, mass",
                "  └─ Decoder_cls → cls_map → Head_map",
                "       + Head_μ(μ) → p",
                "keep if p≥τ ∧ m≥m_min",
                "```",
                "",
                "Hard weights in BCE:",
                f"- FT risk (empty & SEG fire): ×{args.w_ft}",
                f"- TF risk (nonempty & SEG fire): ×{args.w_tf}",
                "- τ not in loss; VAL pick min empty-FP @ ne floor",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False
    )
    model = BTCascade1Enc2DecHead(base=BASE).to(DEVICE)
    nload = load_from_cascade_focal(model, safe_load(FOCAL_SEG, map_location="cpu"))
    model.freeze_encoder()
    model.freeze_seg()
    model.set_trainable(True)
    safe_print(f"loaded focal n={nload}; train Dec_cls+Heads; w_ft={args.w_ft} w_tf={args.w_tf}")

    if args.eval_only:
        st = safe_load(best_path, map_location="cpu")
        model.load_state_dict(st["model"], strict=False)
        model.freeze_encoder()
        model.freeze_seg()
        val, test = collect(model, data.val_loader), collect(model, data.test_loader)
        _, mt = pick_gate(val, test, 0.60)
        safe_print(
            f"TEST ne60 → ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
            f"TF={mt['TF']} FT={mt['FT']}"
        )
        return

    head_params = list(model.head_mu.parameters()) + list(model.head_map.parameters()) + [model.fuse_w]
    dec_params = list(model.decoder_cls.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": dec_params, "lr": LR},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_sel, bad = -1e9, 0
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, data.train_loader, opt, args.w_ft, args.w_tf)
        val = collect(model, data.val_loader)
        auroc = binary_auroc(val["gt"] > 0, val["score"])
        sel = val_select_score(val)
        cm = conf_matrix(val["gt"] > 0, val["score"] >= 0.5)
        rec = {
            "ep": ep,
            "train_loss": tr_loss,
            "val_auroc": auroc,
            "val_sel": sel,
            "val_TF": cm["TF"],
            "val_FT": cm["FT"],
            "val_sens": cm["sensitivity"],
            "val_spec": cm["specificity"],
        }
        hist.append(rec)
        safe_print(
            f"[1enc2dec-head {ep}/{args.epochs}] loss={tr_loss:.4f} auroc={auroc:.4f} "
            f"sel={sel:.4f} @0.5 TF={cm['TF']} FT={cm['FT']} "
            f"sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f}"
        )
        sch.step(sel)
        payload = {
            "model": model.state_dict(),
            "ep": ep,
            "val_sel": sel,
            "val_auroc": auroc,
            "design": "1enc2dec_head_tffp",
            "w_ft": args.w_ft,
            "w_tf": args.w_tf,
        }
        atomic_torch_save(payload, ckpt_dir / "cls_last.pt")
        if sel > best_sel + 1e-4:
            best_sel = sel
            bad = 0
            atomic_torch_save(payload, best_path)
            safe_print(f"  * best sel={sel:.4f}")
        else:
            bad += 1
            if bad >= ES:
                safe_print(f"early stop @ {ep}")
                break

    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"], strict=False)
    model.freeze_encoder()
    model.freeze_seg()
    val, test = collect(model, data.val_loader), collect(model, data.test_loader)
    auroc = binary_auroc(val["gt"] > 0, val["score"])

    v60, t60 = pick_gate(val, test, 0.60)
    v57, t57 = pick_gate(val, test, 0.57)
    raw_keep = np.ones(len(test["gt"]), dtype=bool)
    raw = metrics_bundle(test["gt"], test["mass"], test["dice"], raw_keep)

    rows = [
        {"name": "VAL_ne60", **{k: v60[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "TF", "FT", "sensitivity")}},
        {"name": "TEST_ne60", **{k: t60[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "TF", "FT", "sensitivity")}},
        {"name": "TEST_ne57", **{k: t57[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "TF", "FT", "sensitivity")}},
        {"name": "messy_wprod", "nonempty_dice": WPROD_NE, "empty_fp_rate": WPROD_FP},
    ]
    write_csv(OUT / "final_metrics.csv", rows)
    write_csv(OUT / "history.csv", hist)

    lines = [
        "# 1enc·2dec + cls Head (TF/FP-aware)",
        "",
        f"VAL AUROC={auroc:.4f} · raw SEG ne={float(raw['nonempty_dice']):.3f}",
        "",
        "| system | ne | empty FP | TF | FT | note |",
        "|---|---:|---:|---:|---:|---|",
        f"| messy wprod+mass | {WPROD_NE:.3f} | {WPROD_FP:.3f} | — | — | 3 backbones |",
        (
            f"| **1enc2dec head+mass (ne≥0.60)** | **{t60['nonempty_dice']:.3f}** | "
            f"**{t60['empty_fp_rate']:.3f}** | {t60['TF']} | {t60['FT']} | "
            f"τ={t60['thr']:.2f} m≥{t60['min_mass']} |"
        ),
        (
            f"| 1enc2dec head+mass (ne≥0.57) | {t57['nonempty_dice']:.3f} | "
            f"{t57['empty_fp_rate']:.3f} | {t57['TF']} | {t57['FT']} | "
            f"τ={t57['thr']:.2f} m≥{t57['min_mass']} |"
        ),
        "",
        f"weights: w_FT={args.w_ft} w_TF={args.w_tf} · seconds={time.time()-t0:.0f}",
    ]
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "val_auroc": auroc,
                "test_ne60": t60,
                "test_ne57": t57,
                "messy": {"ne": WPROD_NE, "fp": WPROD_FP},
                "w_ft": args.w_ft,
                "w_tf": args.w_tf,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    safe_print(
        f"TEST ne≥0.60 → ne={t60['nonempty_dice']:.4f} fp={t60['empty_fp_rate']:.3f} "
        f"TF={t60['TF']} FT={t60['FT']} (messy {WPROD_NE}/{WPROD_FP})"
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
