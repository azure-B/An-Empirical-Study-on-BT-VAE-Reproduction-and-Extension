"""Blank empty FPs only if nonempty (TT) cannot be hurt on VAL.

Frozen none / S5. No training. original/ untouched.

1) slice mass: τ = min pred-mass among VAL nonempty slices with mass>0.
   Blank a slice iff pred-mass < τ. VAL nonempty Dice is identical.
2) patient max-mass: τ = min over VAL nonempty patients of max slice mass.
   Blank every slice of a patient iff that patient's max mass < τ.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_neighbor_gate import (
    THR,
    index_originals,
    list_split,
    load_gray,
    load_mask,
    label_path,
    parse_stem,
)

BATCH = 8
from models_bp_skip import VAE as SkipVAE
from train_original_bp import (
    BASE,
    DEVICE,
    OrigVAE,
    RESULT_DIR,
    safe_load,
    safe_print,
    split_dice_stats,
)

OUT = RESULT_DIR / "tt_safe_blank"


def stats_of(pred, gt):
    pred_s = pred.reshape(pred.shape[0], -1).sum(axis=1)
    gt_s = gt.reshape(gt.shape[0], -1).sum(axis=1)
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(axis=1)
    dices = (2.0 * inter + 1e-6) / (pred_s + gt_s + 1e-6)
    return split_dice_stats(dices, gt_s, pred_s), pred_s, gt_s


@torch.no_grad()
def infer(model, idx, stems):
    model.eval()
    preds, gts, names, peaks = [], [], [], []
    b_img, b_gt, b_name = [], [], []

    def flush():
        if not b_img:
            return
        x = torch.stack(b_img, 0).unsqueeze(1).to(DEVICE)
        logits, _, _ = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > THR).float().cpu()
        peak = prob.amax(dim=(1, 2, 3)).cpu().numpy()
        preds.append(pred[:, 0])
        gts.append(torch.stack(b_gt, 0))
        peaks.append(peak)
        names.extend(b_name)
        b_img.clear()
        b_gt.clear()
        b_name.clear()

    for stem in stems:
        orig = idx[stem]
        b_img.append(load_gray(orig))
        b_gt.append(load_mask(label_path(orig)))
        b_name.append(stem)
        if len(b_img) >= BATCH:
            flush()
    flush()
    pred = torch.cat(preds, 0).numpy()
    gt = torch.cat(gts, 0).numpy()
    peak = np.concatenate(peaks, 0)
    return names, pred, gt, peak


def patient_ids(names):
    return [parse_stem(s)[0] for s in names]


def patient_max(names, values):
    bag = defaultdict(list)
    for n, v in zip(names, values):
        bag[parse_stem(n)[0]].append(float(v))
    return {pid: max(vs) for pid, vs in bag.items()}


def nonempty_patients(names, gt_s):
    bag = defaultdict(float)
    for n, g in zip(names, gt_s):
        bag[parse_stem(n)[0]] += float(g)
    return {pid for pid, s in bag.items() if s > 0}


def blank_slice_mass(pred, pred_s, tau):
    out = pred.copy()
    out[pred_s < tau] = 0
    return out


def blank_patient(pred, names, scores, tau):
    out = pred.copy()
    for i, n in enumerate(names):
        if scores[parse_stem(n)[0]] < tau:
            out[i] = 0
    return out


def pack(st):
    return {
        "dice": st["dice_mean"],
        "nonempty": st["mean_dice_nonempty"],
        "empty_fp": st["empty_fp_rate"],
    }


def eval_model(tag, model, idx):
    rec = {"tag": tag}
    caches = {}
    for split in ("val", "test"):
        names, pred, gt, peak = infer(model, idx, list_split(split))
        base, pred_s, gt_s = stats_of(pred, gt)
        caches[split] = (names, pred, gt, peak, pred_s, gt_s, base)
        rec[split] = {"raw": pack(base)}
        safe_print(f"{tag} {split} raw", pack(base))

    names, pred, gt, peak, pred_s, gt_s, base = caches["val"]
    ne = gt_s > 0
    pos_mass = pred_s[ne]
    pos_mass = pos_mass[pos_mass > 0]
    slice_tau = float(pos_mass.min()) if pos_mass.size else 0.0
    pmax = patient_max(names, pred_s)
    ne_p = nonempty_patients(names, gt_s)
    ne_scores = [pmax[p] for p in ne_p]
    pat_tau = float(min(ne_scores)) if ne_scores else 0.0
    rec["slice_tau_mass"] = slice_tau
    rec["patient_tau_mass"] = pat_tau
    safe_print(f"{tag} VAL τ slice={slice_tau:.1f} patient={pat_tau:.1f}")

    for split in ("val", "test"):
        names, pred, gt, peak, pred_s, gt_s, base = caches[split]
        sl = blank_slice_mass(pred, pred_s, slice_tau)
        st_sl, _, _ = stats_of(sl, gt)
        pmax = patient_max(names, pred_s)
        pt = blank_patient(pred, names, pmax, pat_tau)
        st_pt, _, _ = stats_of(pt, gt)
        rec[split]["slice_mass_0fn"] = pack(st_sl)
        rec[split]["patient_mass_0fn"] = pack(st_pt)
        rec[split]["nonempty_delta_slice"] = (
            st_sl["mean_dice_nonempty"] - base["mean_dice_nonempty"]
        )
        rec[split]["nonempty_delta_patient"] = (
            st_pt["mean_dice_nonempty"] - base["mean_dice_nonempty"]
        )
        n_pat_blank = sum(
            1 for pid, sc in pmax.items() if sc < pat_tau
        )
        rec[split]["n_patients_blanked"] = n_pat_blank
        rec[split]["n_patients"] = len(pmax)
        safe_print(
            f"{tag} {split} slice0fn {pack(st_sl)} dTT={rec[split]['nonempty_delta_slice']:+.4f}"
        )
        safe_print(
            f"{tag} {split} patient0fn {pack(st_pt)} dTT={rec[split]['nonempty_delta_patient']:+.4f} "
            f"blanked_pat={n_pat_blank}/{len(pmax)}"
        )
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = index_originals()
    rows = []

    none = OrigVAE(base=BASE)
    st = safe_load(RESULT_DIR / "none" / "seg_best.pt", map_location="cpu")
    none.load_state_dict(st["model"])
    none.to(DEVICE).eval()
    del st
    rows.append(eval_model("none", none, idx))
    del none
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    s5 = SkipVAE(base=BASE)
    st = safe_load(RESULT_DIR / "s5_concat" / "seg_best.pt", map_location="cpu")
    s5.load_state_dict(st["model"])
    s5.to(DEVICE).eval()
    del st
    rows.append(eval_model("s5", s5, idx))

    (OUT / "report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    safe_print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
