"""Slice-presence scores used by the classification gate."""
from __future__ import annotations

import numpy as np

from pipeline.engine import split_dice_stats


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(np.int32)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(1, int(y.sum()))
    precision = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_r)
        prev_r = float(r)
    return float(ap)


def conf_matrix(y: np.ndarray, pred: np.ndarray) -> dict:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tt = int((y & pred).sum())
    ft = int((~y & pred).sum())
    tf = int((y & ~pred).sum())
    ff = int((~y & ~pred).sum())
    sens = tt / max(1, tt + tf)
    spec = ff / max(1, ff + ft)
    prec = tt / max(1, tt + ft)
    npv = ff / max(1, ff + tf)
    return {
        "TT": tt,
        "FT": ft,
        "TF": tf,
        "FF": ff,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "empty_fp_rate": float(ft / max(1, ft + ff)),
    }


def apply_case_rejection(dices_raw, pred_mass, gt, keep_mask: np.ndarray):
    """Blank a slice when the head says no lesion. Empty-empty Dice stays 1."""
    pred = pred_mass.copy()
    pred[~keep_mask] = 0
    dices = dices_raw.copy()
    rejected = ~keep_mask
    empty = gt == 0
    dices[rejected & empty] = 1.0
    dices[rejected & ~empty] = 0.0
    return split_dice_stats(dices, gt, pred), pred


def metrics_bundle(gt, pred_mass, dices, keep: np.ndarray) -> dict:
    st, pred = apply_case_rejection(dices, pred_mass, gt, keep)
    gt_p = gt > 0
    final_p = pred > 0
    cm = conf_matrix(gt_p, final_p)
    return {
        "overall_dice": float(st["dice_mean"]),
        "nonempty_dice": float(st["mean_dice_nonempty"]),
        "empty_fp_rate": float(st["empty_fp_rate"]),
        **cm,
    }
