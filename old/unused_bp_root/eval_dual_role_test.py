"""Role-split dual test (no train): skip=recon mask, plain=presence gate.

A) hard: PresenceHead(plain μ) → keep S5 else blank
B) soft: merge_skip_gated(S5, plain logits)  [prior DualVAE init]
C) hard: plain pred-mass / max-prob thr (VAL pick) → keep S5 else blank

Case dumps + VAL/TEST tables.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_dual_dec import merge_skip_gated
from models_presence import PresenceHead
from train_lesion_presence import (
    PIXEL_THR,
    _prep,
    metrics_bundle,
    patient_id,
    sample_id,
    seg_forward_det,
)
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RESULT_DIR,
    safe_load,
    safe_print,
)
from eval_s5_presence_combo import seg_forward_det_s5

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OLD_HEAD = RESULT_DIR / "none_presence" / "head_best.pt"
OUT = RESULT_DIR / "dual_role_test"
BATCH = 8
OLD_TAU = 0.20
SENS_DROP = 0.02
NE_DROP = 0.02


@torch.inference_mode()
def collect(vae_plain, vae_skip, head, ds, batch_size: int):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    bags = {
        "names": [], "gt": [],
        "dice_s5": [], "mass_s5": [],
        "dice_plain": [], "mass_plain": [],
        "dice_merge": [], "mass_merge": [],
        "p_head": [], "plain_maxp": [],
    }
    i0 = 0
    vae_plain.eval()
    vae_skip.eval()
    head.eval()

    def dice_mass(out, seg_b):
        prob = torch.sigmoid(out)
        pred = (prob >= PIXEL_THR).float()
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        return dice, ps, gs, prob

    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_b = seg.unsqueeze(1) if seg.ndim == 3 else seg
        out_p, mu_p, _ = seg_forward_det(vae_plain, flair)
        out_s, _, _ = seg_forward_det_s5(vae_skip, flair)
        merged = merge_skip_gated(out_s, out_p)

        d_s, m_s, gs, _ = dice_mass(out_s, seg_b)
        d_p, m_p, _, prob_p = dice_mass(out_p, seg_b)
        d_m, m_m, _, _ = dice_mass(merged, seg_b)
        p_h = torch.sigmoid(head(mu_p))
        if p_h.ndim > 1:
            p_h = p_h.view(p_h.size(0), -1).squeeze(1)
        maxp = prob_p.amax(dim=(1, 2, 3))

        b = flair.size(0)
        for j in range(b):
            idx = i0 + j
            if idx >= len(ds.filenames):
                break
            bags["names"].append(ds.filenames[idx])
        bags["gt"].append(gs.cpu().numpy())
        bags["dice_s5"].append(d_s.cpu().numpy())
        bags["mass_s5"].append(m_s.cpu().numpy())
        bags["dice_plain"].append(d_p.cpu().numpy())
        bags["mass_plain"].append(m_p.cpu().numpy())
        bags["dice_merge"].append(d_m.cpu().numpy())
        bags["mass_merge"].append(m_m.cpu().numpy())
        bags["p_head"].append(p_h.detach().cpu().numpy())
        bags["plain_maxp"].append(maxp.cpu().numpy())
        i0 += b
    n = len(bags["names"])
    return {k: (np.concatenate(v, 0)[:n] if k != "names" else v[:n]) for k, v in bags.items()}


def eval_hard(bag, keep, dice_key, mass_key, name):
    return {"name": name, **metrics_bundle(bag["gt"], bag[mass_key], bag[dice_key], keep)}


def pick_thr(score_v, bag_v, bag_t, score_t, c0_ne, c0_sens, label):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        mv = eval_hard(bag_v, score_v >= thr, "dice_s5", "mass_s5", f"{label}@{thr:.2f}")
        ok = mv["nonempty_dice"] >= c0_ne - NE_DROP and mv["sensitivity"] >= c0_sens - SENS_DROP
        if ok and (best is None or mv["empty_fp_rate"] < best["empty_fp_rate"]):
            best = {**mv, "thr": thr}
    if best is None:
        return None, None
    mt = eval_hard(bag_t, score_t >= best["thr"], "dice_s5", "mass_s5", f"{label}@VAL{best['thr']:.2f}")
    return best, mt


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def flat(r):
    keys = ("name", "overall_dice", "nonempty_dice", "empty_fp_rate",
            "sensitivity", "specificity", "TT", "FT", "TF", "FF")
    return {k: r[k] for k in keys}


def dump_cases(path, bag, keep_head, keep_maxp, keep_mass):
    gt = bag["gt"] > 0
    rows = []
    for i, name in enumerate(bag["names"]):
        gt_p = bool(gt[i])
        s5_p = bool(bag["mass_s5"][i] > 0)
        plain_p = bool(bag["mass_plain"][i] > 0)
        rows.append({
            "sample_id": sample_id(name),
            "patient_id": patient_id(name),
            "gt_present": int(gt_p),
            "group": ("nonempty" if gt_p else ("empty_fp_s5" if s5_p else "empty_tn_s5")),
            "s5_present": int(s5_p),
            "plain_present": int(plain_p),
            "dice_s5": float(bag["dice_s5"][i]),
            "dice_plain": float(bag["dice_plain"][i]),
            "dice_merge": float(bag["dice_merge"][i]),
            "p_head": float(bag["p_head"][i]),
            "plain_maxp": float(bag["plain_maxp"][i]),
            "mass_plain": float(bag["mass_plain"][i]),
            "keep_head": int(keep_head[i]),
            "keep_maxp": int(keep_maxp[i]),
            "keep_mass": int(keep_mass[i]),
            "agree_head_plainmass": int(bool(keep_head[i]) == plain_p),
        })
    write_csv(path, rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (NONE_BEST, S5_BEST, OLD_HEAD):
        if not p.exists():
            raise FileNotFoundError(p)

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    plain = OrigVAE(base=BASE)
    plain.load_state_dict(safe_load(NONE_BEST, map_location="cpu")["model"])
    plain.to(DEVICE).eval()
    skip = SkipVAE(base=BASE)
    skip.load_state_dict(safe_load(S5_BEST, map_location="cpu")["model"])
    skip.to(DEVICE).eval()
    head = PresenceHead(32 * BASE)
    head.load_state_dict(safe_load(OLD_HEAD, map_location="cpu")["head"])
    head.to(DEVICE).eval()

    safe_print("collect VAL…")
    val = collect(plain, skip, head, data.val, BATCH)
    safe_print("collect TEST…")
    test = collect(plain, skip, head, data.test, BATCH)

    ones_v = np.ones(len(val["gt"]), dtype=bool)
    ones_t = np.ones(len(test["gt"]), dtype=bool)

    rows_v, rows_t = [], []

    # refs
    for tag, dice_k, mass_k in (
        ("S5_raw", "dice_s5", "mass_s5"),
        ("plain_raw", "dice_plain", "mass_plain"),
        ("soft_merge", "dice_merge", "mass_merge"),
    ):
        rows_v.append(eval_hard(val, ones_v, dice_k, mass_k, tag))
        rows_t.append(eval_hard(test, ones_t, dice_k, mass_k, tag))

    # A) hard presence head on plain μ → S5 mask
    kh_v = val["p_head"] >= OLD_TAU
    kh_t = test["p_head"] >= OLD_TAU
    rows_v.append(eval_hard(val, kh_v, "dice_s5", "mass_s5", "A_hard_head@0.20"))
    rows_t.append(eval_hard(test, kh_t, "dice_s5", "mass_s5", "A_hard_head@0.20"))

    # also blank-reject on plain mask alone for reference
    rows_v.append(eval_hard(val, kh_v, "dice_plain", "mass_plain", "A_hard_head_plainMask@0.20"))
    rows_t.append(eval_hard(test, kh_t, "dice_plain", "mass_plain", "A_hard_head_plainMask@0.20"))

    s5_c0 = next(r for r in rows_v if r["name"] == "S5_raw")
    # B already soft_merge continuous
    # C) VAL thr on plain_maxp / plain mass>0 is binary — use maxp continuous
    bv, bt = pick_thr(
        val["plain_maxp"], val, test, test["plain_maxp"],
        s5_c0["nonempty_dice"], s5_c0["sensitivity"], "C_hard_maxp",
    )
    if bv is not None:
        rows_v.append({**bv, "name": f"C_hard_maxp@VAL{bv['thr']:.2f}"})
        rows_t.append(bt)
        safe_print(f"PICK maxp thr={bv['thr']:.2f} VAL fp={bv['empty_fp_rate']:.3f} ne={bv['nonempty_dice']:.4f}")
    else:
        safe_print("PICK maxp: no constrained thr vs S5 C0")

    # C2: thr vs plain C0 floor (more realistic — don't force S5 ne floor)
    plain_c0 = next(r for r in rows_v if r["name"] == "plain_raw")
    bv2, bt2 = pick_thr(
        val["plain_maxp"], val, test, test["plain_maxp"],
        plain_c0["nonempty_dice"], plain_c0["sensitivity"], "C_maxp_vsPlain",
    )
    if bv2 is not None:
        rows_v.append({**bv2, "name": f"C_maxp_vsPlain@VAL{bv2['thr']:.2f}"})
        rows_t.append(bt2)
        safe_print(
            f"PICK maxp vsPlain thr={bv2['thr']:.2f} "
            f"VAL fp={bv2['empty_fp_rate']:.3f} ne={bv2['nonempty_dice']:.4f}"
        )

    # head thr search vs plain floor (keep ne near plain, minimize FP) on S5 mask
    best_h, test_h = None, None
    for i in range(1, 100):
        thr = i / 100.0
        mv = eval_hard(val, val["p_head"] >= thr, "dice_s5", "mass_s5", f"head@{thr:.2f}")
        ok = (
            mv["nonempty_dice"] >= plain_c0["nonempty_dice"] - NE_DROP
            and mv["sensitivity"] >= plain_c0["sensitivity"] - SENS_DROP
        )
        if ok and (best_h is None or mv["empty_fp_rate"] < best_h["empty_fp_rate"]):
            best_h = {**mv, "thr": thr}
    if best_h is not None:
        test_h = eval_hard(
            test, test["p_head"] >= best_h["thr"], "dice_s5", "mass_s5",
            f"A_head_vsPlain@VAL{best_h['thr']:.2f}",
        )
        rows_v.append({**best_h, "name": f"A_head_vsPlain@VAL{best_h['thr']:.2f}"})
        rows_t.append(test_h)
        safe_print(
            f"PICK head vsPlain thr={best_h['thr']:.2f} "
            f"VAL fp={best_h['empty_fp_rate']:.3f} ne={best_h['nonempty_dice']:.4f}"
        )

    # mass>0 plain as keep (binary)
    km_v = val["mass_plain"] > 0
    km_t = test["mass_plain"] > 0
    rows_v.append(eval_hard(val, km_v, "dice_s5", "mass_s5", "C_hard_plainMass>0"))
    rows_t.append(eval_hard(test, km_t, "dice_s5", "mass_s5", "C_hard_plainMass>0"))

    keep_maxp_v = val["plain_maxp"] >= (bv2["thr"] if bv2 else 0.5)
    dump_cases(
        OUT / "val_cases.csv", val, kh_v,
        keep_maxp_v, km_v,
    )
    dump_cases(
        OUT / "test_cases.csv", test, kh_t,
        test["plain_maxp"] >= (bv2["thr"] if bv2 else 0.5), km_t,
    )

    write_csv(OUT / "val_table.csv", [flat(r) for r in rows_v])
    write_csv(OUT / "test_table.csv", [flat(r) for r in rows_t])

    # agreement on VAL empty-FP of S5
    gt = val["gt"] > 0
    efp = (~gt) & (val["mass_s5"] > 0)
    agree = {
        "n_s5_empty_fp": int(efp.sum()),
        "rejected_by_head": int((efp & ~kh_v).sum()),
        "rejected_by_plainMass": int((efp & ~km_v).sum()),
        "rejected_by_both": int((efp & ~kh_v & ~km_v).sum()),
        "kept_by_both": int((efp & kh_v & km_v).sum()),
        "nonempty_killed_head": int((gt & ~kh_v).sum()),
        "nonempty_killed_plainMass": int((gt & ~km_v).sum()),
    }

    report = {
        "role": {
            "skip": "S5 decoder = reconstruction / SEG mask",
            "plain": "none encoder μ → PresenceHead = lesion yes/no (A); "
                     "plain decoder logits = soft/hard gate (B/C)",
        },
        "val_agreement_s5_empty_fp": agree,
        "val": [flat(r) for r in rows_v],
        "test": [flat(r) for r in rows_t],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    safe_print("\n=== TEST ===")
    for r in rows_t:
        safe_print(
            f"{r['name']:36s} dice={r['overall_dice']:.4f} ne={r['nonempty_dice']:.4f} "
            f"fp={r['empty_fp_rate']:.3f} sens={r['sensitivity']:.3f} "
            f"TT={r['TT']} FT={r['FT']} TF={r['TF']} FF={r['FF']}"
        )
    safe_print("VAL S5-emptyFP agree", json.dumps(agree))
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
