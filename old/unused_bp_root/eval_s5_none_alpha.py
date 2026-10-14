"""Frozen S5 + none + α. Six empty-FP control rules, same val/test.

α=0 → closer to S5 (high nonempty, high empty FP)
α=1 → stronger empty suppression
No training. original/ untouched.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

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

OUT = RESULT_DIR / "s5_none_alpha"
ALPHAS = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 11))
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
    s5s, nones, gts, peaks, masses = [], [], [], [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        _m, skip_l, plain_l, _, _ = vae(flair, return_heads=True)
        s5 = torch.sigmoid(skip_l).squeeze(1).cpu().numpy().astype(np.float16)
        none = torch.sigmoid(plain_l).squeeze(1).cpu().numpy().astype(np.float16)
        s5s.append(s5)
        nones.append(none)
        gts.append((seg.squeeze(1).cpu().numpy() > 0).astype(np.uint8))
        peaks.append(none.max(axis=(1, 2)))
        masses.append((none > 0.5).reshape(none.shape[0], -1).sum(1))
    return (
        np.concatenate(s5s, 0),
        np.concatenate(nones, 0),
        np.concatenate(gts, 0),
        np.concatenate(peaks, 0).astype(np.float32),
        np.concatenate(masses, 0).astype(np.float32),
    )


def stats_bin(pred, gt):
    pred_s = pred.reshape(pred.shape[0], -1).sum(1)
    gt_s = gt.reshape(gt.shape[0], -1).sum(1)
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(1)
    dices = (2.0 * inter + 1e-6) / (pred_s + gt_s + 1e-6)
    return split_dice_stats(dices, gt_s, pred_s)


def pack(st):
    return {
        "dice": float(st["dice_mean"]),
        "nonempty": float(st["mean_dice_nonempty"]),
        "empty_fp": float(st["empty_fp_rate"]),
    }


def dilate(bin_hw, radius: int):
    if radius <= 0:
        return bin_hw
    t = torch.from_numpy(bin_hw.astype(np.float32)).unsqueeze(1)
    k = 2 * radius + 1
    y = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return (y.squeeze(1).numpy() > 0.5)


def apply(method: str, alpha: float, s5, none, peak, mass):
    a = float(alpha)
    s5f = s5.astype(np.float32)
    nf = none.astype(np.float32)
    if method == "mix_prob":
        p = (1.0 - a) * s5f + a * nf
        return (p > 0.5).astype(np.uint8)
    if method == "prod_gate":
        p = s5f * np.power(np.clip(nf, 1e-6, 1.0), a)
        return (p > 0.5).astype(np.uint8)
    if method == "pixel_veto":
        return ((s5f > 0.5) & (nf >= a)).astype(np.uint8)
    if method == "mass_cgm":
        pred = (s5f > 0.5).astype(np.uint8)
        # α=0: keep all; α=1: blank below 95th pct of none mass
        tau = np.quantile(mass, a) if a > 0 else -1.0
        pred[mass < tau] = 0
        return pred
    if method == "adapt_thr":
        # none_peak high → keep S5 thr=0.5; none unsure → raise thr
        thr = 0.5 + a * 0.45 * (1.0 - peak)
        return (s5f > thr[:, None, None]).astype(np.uint8)
    if method == "dilate_and":
        r = int(round((1.0 - a) * 7))
        nbin = (nf > 0.5).astype(np.uint8)
        nd = dilate(nbin, r)
        return ((s5f > 0.5) & nd).astype(np.uint8)
    raise ValueError(method)


METHODS = [
    {
        "id": "mix_prob",
        "name": "prob mix",
        "paper": "Kamnitsas et al. ensemble averaging; 2024 low-precision ensemble (Sundaresan/PMC11228952)",
        "formula": "((1-α) S5 + α none) > 0.5",
    },
    {
        "id": "prod_gate",
        "name": "hard product gate",
        "paper": "SG-Net hard spatial gating, over-segmentation paradox (arXiv:2511.22606)",
        "formula": "(S5 * none^α) > 0.5",
    },
    {
        "id": "pixel_veto",
        "name": "none pixel veto",
        "paper": "decision-level complementary fusion / MEFN DTF (arXiv:2406.18327)",
        "formula": "S5>0.5 AND none≥α",
    },
    {
        "id": "mass_cgm",
        "name": "none-mass CGM",
        "paper": "UNet3+ classification-guided module (Huang et al. 2020); UNet# CGM (CMPB 2024)",
        "formula": "keep S5 slice iff none_mass ≥ Q_α",
    },
    {
        "id": "adapt_thr",
        "name": "adaptive S5 threshold",
        "paper": "uncertainty-calibrated fusion (MEFN 2024); none peak as trust",
        "formula": "S5 > 0.5 + α·0.45·(1-none_peak)",
    },
    {
        "id": "dilate_and",
        "name": "dilated AND",
        "paper": "anatomically constrained ensemble (cascaded U-Net CAC 2025); none as prior",
        "formula": "S5 AND dilate(none, (1-α)·7px)",
    },
]


def nearest(curve, target_fp):
    best = min(curve, key=lambda r: abs(r["empty_fp"] - target_fp))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    vae = DualVAE(base=BASE)
    load_s5_none(vae)
    vae.to(DEVICE)
    v = collect(vae, data.val_loader)
    t = collect(vae, data.test_loader)
    del vae
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    report = {"alphas": list(ALPHAS), "methods": []}
    for spec in METHODS:
        mid = spec["id"]
        safe_print("====", mid)
        vs5, vnone, vgt, vpeak, vmass = v
        ts5, tnone, tgt, tpeak, tmass = t
        val_curve = []
        test_curve = []
        for a in ALPHAS:
            vp = apply(mid, a, vs5, vnone, vpeak, vmass)
            tp = apply(mid, a, ts5, tnone, tpeak, tmass)
            vr = {"alpha": a, **pack(stats_bin(vp, vgt))}
            tr = {"alpha": a, **pack(stats_bin(tp, tgt))}
            val_curve.append(vr)
            test_curve.append(tr)
            safe_print(
                f"  α={a:.2f} VAL ne={vr['nonempty']:.4f} fp={vr['empty_fp']:.3f}  "
                f"TEST ne={tr['nonempty']:.4f} fp={tr['empty_fp']:.3f} dice={tr['dice']:.4f}"
            )
        picks = []
        for tgt_fp in TARGETS:
            vb = nearest(val_curve, tgt_fp)
            tb = next(x for x in test_curve if x["alpha"] == vb["alpha"])
            picks.append({
                "target_empty_fp": tgt_fp,
                "alpha": vb["alpha"],
                "val": {k: vb[k] for k in ("alpha", "dice", "nonempty", "empty_fp")},
                "test": {k: tb[k] for k in ("alpha", "dice", "nonempty", "empty_fp")},
            })
            safe_print(
                f"  PICK fp~{tgt_fp:.2f} α={vb['alpha']:.2f} "
                f"TEST ne={tb['nonempty']:.4f} fp={tb['empty_fp']:.3f}"
            )
        report["methods"].append({**spec, "val": val_curve, "test": test_curve, "picks": picks})

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
