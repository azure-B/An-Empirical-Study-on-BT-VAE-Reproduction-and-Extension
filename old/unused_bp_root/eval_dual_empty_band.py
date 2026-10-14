"""Frozen S5 + none. Sweep how hard none must fire to keep S5.

Calibrate τ on VAL so empty_fp is about 0.30 / 0.20 / 0.10, then TEST.
No training. original/ untouched.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from data_load_BT import datasets
from models_dual_dec import DualVAE
from train_dual_dec import BATCH, load_s5_none
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    safe_print,
    split_dice_stats,
)

OUT = RESULT_DIR / "dual_dec"
THR = 0.5
TARGETS = (0.30, 0.20, 0.10)


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


@torch.no_grad()
def collect(vae, loader):
    vae.eval()
    skips, plains, gts, peaks, masses = [], [], [], [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        _m, skip_l, plain_l, _, _ = vae(flair, return_heads=True)
        skip = (torch.sigmoid(skip_l) > THR).float().cpu()
        plain = (torch.sigmoid(plain_l) > THR).float().cpu()
        peak = torch.sigmoid(plain_l).amax(dim=(1, 2, 3)).cpu().numpy()
        mass = plain.sum(dim=(1, 2, 3)).numpy()
        skips.append(skip.numpy())
        plains.append(plain.numpy())
        gts.append(seg.cpu().numpy())
        peaks.append(peak)
        masses.append(mass)
    skip = np.concatenate(skips, 0)
    plain = np.concatenate(plains, 0)
    gt = np.concatenate(gts, 0)
    if gt.ndim == 3:
        gt = gt[:, None]
    if skip.ndim == 3:
        skip = skip[:, None]
    if plain.ndim == 3:
        plain = plain[:, None]
    return skip, plain, gt, np.concatenate(peaks, 0), np.concatenate(masses, 0)


def stats_pred(pred, gt):
    pred_s = pred.reshape(pred.shape[0], -1).sum(1)
    gt_s = gt.reshape(gt.shape[0], -1).sum(1)
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(1)
    dices = (2.0 * inter + 1e-6) / (pred_s + gt_s + 1e-6)
    return split_dice_stats(dices, gt_s, pred_s)


def stats_peak(skip, gt, peak, tau):
    pred = skip.copy()
    pred[peak < tau] = 0
    return stats_pred(pred, gt)


def stats_mass(skip, gt, mass, tau):
    pred = skip.copy()
    pred[mass < tau] = 0
    return stats_pred(pred, gt)


def pack(st):
    return {
        "dice": float(st["dice_mean"]),
        "nonempty": float(st["mean_dice_nonempty"]),
        "empty_fp": float(st["empty_fp_rate"]),
    }


def nearest_tau(fn, grid, target_fp):
    best_tau, best = None, None
    for tau in grid:
        st = fn(tau)
        if best is None or abs(st["empty_fp_rate"] - target_fp) < abs(best["empty_fp_rate"] - target_fp):
            best_tau, best = float(tau), st
    return best_tau, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    vae = DualVAE(base=BASE)
    load_s5_none(vae)
    vae.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    v_skip, v_plain, v_gt, v_peak, v_mass = collect(vae, data.val_loader)
    t_skip, t_plain, t_gt, t_peak, t_mass = collect(vae, data.test_loader)
    del vae
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    mass_grid = np.unique(np.concatenate([
        np.array([0.0]),
        np.quantile(v_mass, np.linspace(0.05, 0.95, 19)),
    ]))
    safe_print("VAL mass-gate  τ_px   dice   nonempty  empty_fp")
    mass_curve = []
    for tau in mass_grid:
        st = stats_mass(v_skip, v_gt, v_mass, tau)
        row = {"tau_mass": float(tau), **pack(st)}
        mass_curve.append(row)
        safe_print(
            f"  {tau:7.1f}  {row['dice']:.4f}  {row['nonempty']:.4f}  {row['empty_fp']:.3f}"
        )

    and_v = stats_pred(v_skip * v_plain, v_gt)
    and_t = stats_pred(t_skip * t_plain, t_gt)
    safe_print("pixel AND VAL", pack(and_v))
    safe_print("pixel AND TEST", pack(and_t))

    picks = []
    for tgt in TARGETS:
        tau, vst = nearest_tau(lambda t: stats_mass(v_skip, v_gt, v_mass, t), mass_grid, tgt)
        tst = stats_mass(t_skip, t_gt, t_mass, tau)
        rec = {
            "target_empty_fp": tgt,
            "gate": "none_mass",
            "tau_mass": tau,
            "val": pack(vst),
            "test": pack(tst),
        }
        picks.append(rec)
        safe_print(
            f"PICK empty~{tgt:.2f} mass≥{tau:.1f}  "
            f"VAL ne={vst['mean_dice_nonempty']:.4f} fp={vst['empty_fp_rate']:.3f}  "
            f"TEST ne={tst['mean_dice_nonempty']:.4f} fp={tst['empty_fp_rate']:.3f} "
            f"dice={tst['dice_mean']:.4f}"
        )

    out = {
        "note": "keep S5 only if none pred-mass >= τ; τ from VAL empty_fp 0.30/0.20/0.10",
        "val_mass_curve": mass_curve,
        "pixel_and": {"val": pack(and_v), "test": pack(and_t)},
        "picks": picks,
        "raw_s5": {
            "val": pack(stats_pred(v_skip, v_gt)),
            "test": pack(stats_pred(t_skip, t_gt)),
        },
    }
    (OUT / "empty_band.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    safe_print("wrote", OUT / "empty_band.json")


if __name__ == "__main__":
    main()
