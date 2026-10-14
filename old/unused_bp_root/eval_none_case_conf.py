"""VAL logits → case confidence (top-k). No training.

If empty vs nonempty confidence separates, apply constrained case rejection.
If not, recommend an independent presence-classification head next.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RESULT_DIR,
    safe_load,
    safe_print,
    split_dice_stats,
)

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT = RESULT_DIR / "none_case_conf"
BATCH = 16
THR = 0.5
TOPKS = (1, 5, 10, 50, 100, 500)
# keep nonempty dice within this of raw when picking rejection threshold
NE_FLOOR_DELTA = 0.02


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def quantiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {}
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def collect(model, loader):
    model.eval()
    bags = {f"topk{k}": [] for k in TOPKS}
    bags.update({"peak": [], "mass": [], "mean_pos": [], "gt": [], "pred_mass": []})
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        logits, _, _ = model(flair)
        prob = torch.sigmoid(logits).flatten(1)  # [B, HW]
        gt = seg.flatten(1).sum(1).cpu().numpy()
        pred = (prob > THR).float().sum(1).cpu().numpy()
        bags["gt"].append(gt)
        bags["pred_mass"].append(pred)
        bags["peak"].append(prob.amax(1).cpu().numpy())
        bags["mass"].append(prob.mean(1).cpu().numpy())
        pos = prob > THR
        mean_pos = []
        for i in range(prob.size(0)):
            sel = prob[i][pos[i]]
            mean_pos.append(float(sel.mean()) if sel.numel() else 0.0)
        bags["mean_pos"].append(np.array(mean_pos, dtype=np.float32))
        # top-k mean over all pixels (case confidence)
        vals, _ = torch.topk(prob, k=max(TOPKS), dim=1)
        for k in TOPKS:
            bags[f"topk{k}"].append(vals[:, :k].mean(1).cpu().numpy())
    out = {k: np.concatenate(v, 0) for k, v in bags.items()}
    return out


def dice_at_reject(pred_mass, gt, conf, thr, higher_is_more_present=True):
    """Blank slice if conf says empty."""
    if higher_is_more_present:
        keep = conf >= thr
    else:
        keep = conf <= thr
    # reject = force empty
    pred = pred_mass.copy()
    pred[~keep] = 0
    # rebuild dice from pred/gt sums only (empty-empty=1 approximation via split_dice_stats)
    # need per-slice dice: use binary empty/nonempty approx from sums isn't enough for nonempty dice
    # Use: if both empty dice=1; if gt empty pred>0 dice=0; else need pixel overlap — we don't have masks.
    # So only report empty_fp and presence-style proxy from sums + store conf for rejection stats.
    empty = gt == 0
    nonempty = ~empty
    empty_fp = float((pred[empty] > 0).mean()) if empty.any() else None
    # nonempty hit rate (predicted something)
    ne_hit = float((pred[nonempty] > 0).mean()) if nonempty.any() else None
    # FN rate among nonempty
    ne_fn = float((pred[nonempty] == 0).mean()) if nonempty.any() else None
    return {
        "empty_fp": empty_fp,
        "nonempty_hit": ne_hit,
        "nonempty_fn": ne_fn,
        "n_rejected": int((~keep).sum()),
        "n_rejected_empty": int((~keep & empty).sum()),
        "n_rejected_nonempty": int((~keep & nonempty).sum()),
    }


@torch.no_grad()
def collect_with_dice(model, loader, conf_name: str, thr: float, higher=True):
    """Full pixel dice under case rejection."""
    model.eval()
    dices, gts, preds = [], [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        logits, _, _ = model(flair)
        prob = torch.sigmoid(logits)
        flat = prob.flatten(1)
        if conf_name.startswith("topk"):
            k = int(conf_name.replace("topk", ""))
            conf = torch.topk(flat, k=k, dim=1).values.mean(1)
        elif conf_name == "peak":
            conf = flat.amax(1)
        elif conf_name == "mass":
            conf = flat.mean(1)
        else:
            raise ValueError(conf_name)
        keep = conf >= thr if higher else conf <= thr
        pred = (prob > THR).float()
        pred[~keep] = 0
        # dice
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        inter = (pred * seg).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        d = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        d = torch.where(both, torch.ones_like(d), d)
        dices.append(d.cpu().numpy())
        gts.append(gs.cpu().numpy())
        preds.append(ps.cpu().numpy())
    return split_dice_stats(
        np.concatenate(dices), np.concatenate(gts), np.concatenate(preds)
    )


def pack_stats(st):
    return {
        "dice": float(st["dice_mean"]),
        "nonempty": float(st["mean_dice_nonempty"]),
        "empty_fp": float(st["empty_fp_rate"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not NONE_BEST.exists():
        raise FileNotFoundError(NONE_BEST)

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    none_ep = int(st.get("best_ep", -1))
    del st
    vae.to(DEVICE).eval()
    safe_print("DEVICE", DEVICE, "none ep", none_ep, "VAL case-confidence diagnostic")

    val = collect(vae, data.val_loader)
    gt = val["gt"]
    empty = gt == 0
    nonempty = ~empty
    pred_pos = val["pred_mass"] > 0
    empty_fp = empty & pred_pos
    empty_tn = empty & ~pred_pos

    conf_names = [f"topk{k}" for k in TOPKS] + ["peak", "mass"]
    dist = {}
    safe_print("\n=== confidence distributions (VAL) ===")
    for name in conf_names:
        s = val[name].astype(np.float64)
        row = {
            "all": quantiles(s),
            "nonempty": quantiles(s[nonempty]),
            "empty": quantiles(s[empty]),
            "empty_fp": quantiles(s[empty_fp]),
            "empty_tn": quantiles(s[empty_tn]),
            "auroc_present": auroc(nonempty, s),
            "auroc_empty_fp_vs_tn": auroc(empty_fp[empty], s[empty]) if empty.any() else None,
            "mean_nonempty": float(s[nonempty].mean()),
            "mean_empty": float(s[empty].mean()),
            "mean_empty_fp": float(s[empty_fp].mean()) if empty_fp.any() else None,
            "mean_empty_tn": float(s[empty_tn].mean()) if empty_tn.any() else None,
            "gap_ne_empty": float(s[nonempty].mean() - s[empty].mean()),
            "gap_fp_tn": (
                float(s[empty_fp].mean() - s[empty_tn].mean())
                if empty_fp.any() and empty_tn.any()
                else None
            ),
        }
        dist[name] = row
        safe_print(
            f"{name:8s} AUROC(present)={row['auroc_present']:.3f}  "
            f"mean ne={row['mean_nonempty']:.4f} empty={row['mean_empty']:.4f} "
            f"gap={row['gap_ne_empty']:+.4f}  "
            f"emptyFP={row['mean_empty_fp']} emptyTN={row['mean_empty_tn']}"
        )

    # best score by AUROC for presence
    best_name = max(conf_names, key=lambda n: dist[n]["auroc_present"])
    best_auroc = dist[best_name]["auroc_present"]
    # separation criterion: AUROC >= 0.75 and nonempty median > empty_fp median
    ne_med = dist[best_name]["nonempty"].get("p50", 0)
    fp_med = dist[best_name]["empty_fp"].get("p50", 1)
    separated = bool(best_auroc >= 0.75 and ne_med > fp_med)
    safe_print(
        f"\nBEST conf={best_name} AUROC={best_auroc:.3f} "
        f"ne_med={ne_med:.4f} emptyFP_med={fp_med:.4f} SEPARATED={separated}"
    )

    raw = collect_with_dice(vae, data.val_loader, "peak", thr=-1.0, higher=True)  # keep all
    # peak thr=-1 keeps all
    floor = float(raw["mean_dice_nonempty"]) - NE_FLOOR_DELTA
    safe_print("RAW VAL", pack_stats(raw), "floor", f"{floor:.4f}")

    rejection = {"applied": False, "reason": None}
    if separated:
        # sweep thr on VAL for best conf: reject if conf < thr
        conf = val[best_name]
        grid = np.unique(np.quantile(conf, np.linspace(0.0, 0.95, 40)))
        candidates = []
        for thr in grid:
            stv = collect_with_dice(vae, data.val_loader, best_name, float(thr), higher=True)
            ok = stv["mean_dice_nonempty"] is not None and stv["mean_dice_nonempty"] >= floor
            candidates.append({"thr": float(thr), "ok": bool(ok), **pack_stats(stv)})
            safe_print(
                f"  thr={thr:.4f} dice={stv['dice_mean']:.4f} "
                f"ne={stv['mean_dice_nonempty']:.4f} fp={stv['empty_fp_rate']:.3f} "
                f"{'OK' if ok else 'drop'}"
            )
        ok_rows = [c for c in candidates if c["ok"]]
        if ok_rows:
            pick = min(ok_rows, key=lambda c: c["empty_fp"])
            thr = pick["thr"]
            test_raw = collect_with_dice(vae, data.test_loader, "peak", -1.0, True)
            test_rej = collect_with_dice(vae, data.test_loader, best_name, thr, True)
            rejection = {
                "applied": True,
                "conf": best_name,
                "thr": thr,
                "val_curve": candidates,
                "val_pick": pick,
                "test_raw": pack_stats(test_raw),
                "test_rejected": pack_stats(test_rej),
            }
            safe_print("PICK thr", thr, "VAL", pick)
            safe_print("TEST raw ", pack_stats(test_raw))
            safe_print("TEST rej ", pack_stats(test_rej))
        else:
            rejection = {
                "applied": False,
                "reason": "no thr kept nonempty floor",
                "val_curve": candidates,
            }
    else:
        rejection = {
            "applied": False,
            "reason": (
                f"no separation: best {best_name} AUROC={best_auroc:.3f} "
                f"(need >=0.75 and nonempty median > empty-FP median). "
                "Next single change: independent lesion-presence classification head "
                "(not a new segmentation loss)."
            ),
        }
        safe_print(rejection["reason"])

    report = {
        "mode": "none_case_conf_diagnostic",
        "ckpt": str(NONE_BEST),
        "none_ep": none_ep,
        "n_val": int(len(gt)),
        "n_nonempty": int(nonempty.sum()),
        "n_empty": int(empty.sum()),
        "n_empty_fp": int(empty_fp.sum()),
        "n_empty_tn": int(empty_tn.sum()),
        "distributions": dist,
        "best_conf": best_name,
        "best_auroc": best_auroc,
        "separated": separated,
        "raw_val": pack_stats(raw),
        "rejection": rejection,
        "next_step": (
            "constrained_case_rejection"
            if rejection.get("applied")
            else "independent_presence_classification_head"
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # also dump raw arrays for plots
    np.savez_compressed(
        OUT / "val_conf.npz",
        gt=gt,
        pred_mass=val["pred_mass"],
        **{n: val[n] for n in conf_names},
    )
    safe_print("wrote", OUT / "report.json")
    safe_print("NEXT:", report["next_step"])


if __name__ == "__main__":
    main()
