"""Evaluate current BP original none best ckpt on val/test. CPU to avoid GPU clash."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_load_BT import datasets
import importlib.util

spec = importlib.util.spec_from_file_location("original_models1", ROOT / "original" / "models1.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


@torch.no_grad()
def metrics_batch(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    inter = (pred * target).sum(dim=(1, 2, 3))
    pred_s = pred.sum(dim=(1, 2, 3))
    gt_s = target.sum(dim=(1, 2, 3))
    union = pred_s + gt_s - inter
    dice = (2 * inter + 1e-6) / (pred_s + gt_s + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    prec = (inter + 1e-6) / (pred_s + 1e-6)
    rec = (inter + 1e-6) / (gt_s + 1e-6)
    # empty-empty: dice/iou/prec/rec = 1
    both_empty = (pred_s == 0) & (gt_s == 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    prec = torch.where(both_empty, torch.ones_like(prec), prec)
    rec = torch.where(both_empty, torch.ones_like(rec), rec)
    # gt empty, pred nonempty: rec conventionally 1? keep 0 from formula (inter=0, gt=0 -> rec~0)
    # paper cares about FP on negatives; leave rec as 0 when gt empty and pred not
    return {
        "dice": dice.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "prec": prec.cpu().numpy(),
        "rec": rec.cpu().numpy(),
        "gt": gt_s.cpu().numpy(),
        "pred": pred_s.cpu().numpy(),
    }


def summarize(arrs):
    d, iou, p, r, gt = arrs["dice"], arrs["iou"], arrs["prec"], arrs["rec"], arrs["gt"]
    ne = gt > 0
    empty = ~ne
    out = {
        "n": int(len(d)),
        "n_nonempty": int(ne.sum()),
        "n_empty": int(empty.sum()),
        "mean_dice": float(d.mean()),
        "median_dice": float(np.median(d)),
        "mean_iou": float(iou.mean()),
        "mean_prec": float(p.mean()),
        "mean_rec": float(r.mean()),
        "mean_dice_nonempty": float(d[ne].mean()) if ne.any() else None,
        "median_dice_nonempty": float(np.median(d[ne])) if ne.any() else None,
        "mean_iou_nonempty": float(iou[ne].mean()) if ne.any() else None,
        "mean_prec_nonempty": float(p[ne].mean()) if ne.any() else None,
        "mean_rec_nonempty": float(r[ne].mean()) if ne.any() else None,
        "empty_mean_dice": float(d[empty].mean()) if empty.any() else None,
        "empty_fp_rate": float((arrs["pred"][empty] > 0).mean()) if empty.any() else None,
    }
    return out


@torch.no_grad()
def run_split(model, loader, name):
    bags = {k: [] for k in ["dice", "iou", "prec", "rec", "gt", "pred"]}
    for batch in loader:
        flair, seg = batch[0], batch[1]
        if flair.ndim == 3:
            flair = flair.unsqueeze(1)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        out, _, _ = model(flair.to(DEVICE))
        m = metrics_batch(out, seg.to(DEVICE))
        for k in bags:
            bags[k].append(m[k])
    arrs = {k: np.concatenate(v) for k, v in bags.items()}
    s = summarize(arrs)
    print(name, json.dumps(s, indent=2), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "s5"], default="none")
    args = ap.parse_args()
    folder = "none" if args.mode == "none" else "s5_concat"
    ckpt = ROOT / "results" / "bp_original" / folder / "seg_best.pt"
    out = ROOT / "results" / "bp_original" / folder / "eval_now.json"

    st = safe_load(ckpt)
    print("mode", args.mode, "ckpt ep", st.get("best_ep"), "best_loss", st.get("best_loss"), flush=True)
    if args.mode == "none":
        model = mod.VAE(base=32).to(DEVICE)
    else:
        from models_bp_skip import VAE as SkipVAE
        model = SkipVAE(base=32).to(DEVICE)
    model.load_state_dict(st["model"])
    model.eval()
    data = datasets.bp_train_30(batch_size=8, num_workers=0)
    report = {
        "ckpt": str(ckpt),
        "mode": args.mode,
        "best_ep": st.get("best_ep"),
        "best_val_loss": st.get("best_loss"),
        "val": run_split(model, data.val_loader, "VAL"),
        "test": run_split(model, data.test_loader, "TEST"),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
