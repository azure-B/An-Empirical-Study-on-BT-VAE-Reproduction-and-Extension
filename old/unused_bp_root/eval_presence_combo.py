"""Enumerate case-level rejection combos: C0 / top-k / new head / old presence.

No training. Uses frozen none + saved heads + fixed thr from prior VAL picks.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_lesion_presence import LesionPresenceHead
from models_presence import PresenceHead
from train_lesion_presence import (
    PIXEL_THR,
    TOPK_C1,
    TOPK_THR_C1,
    _prep,
    conf_matrix,
    metrics_bundle,
    pack_seg,
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
    SEG_BATCH,
    safe_load,
    safe_print,
)

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OLD_HEAD = RESULT_DIR / "none_presence" / "head_best.pt"
NEW_HEAD = RESULT_DIR / "lesion_presence" / "checkpoints" / "lesion_presence_head_best_constrained.pt"
OUT = RESULT_DIR / "presence_combo"
OLD_TAU = 0.20
# C2 thr from constrained pick; fallback 0.11
NEW_TAU_DEFAULT = 0.11
SENS_DROP = 0.01
NE_DROP = 0.01


@torch.inference_mode()
def collect(vae, old_head, new_head, ds, batch_size: int):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    vae.eval()
    old_head.eval()
    new_head.eval()
    bags = {
        "gt": [], "pred_mass": [], "dices": [],
        "topk50": [], "p_old": [], "p_new": [], "names": [],
    }
    i0 = 0
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        out, mu, _ = seg_forward_det(vae, flair)
        prob = torch.sigmoid(out)
        flat = prob.flatten(1)
        topk = torch.topk(flat, k=TOPK_C1, dim=1).values.mean(1)
        pred = (prob >= PIXEL_THR).float()
        if seg.ndim == 3:
            seg_b = seg.unsqueeze(1)
        else:
            seg_b = seg
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)

        # old head: GAP-only PresenceHead expects mu [B,C,H,W]
        p_old = torch.sigmoid(old_head(mu))
        if p_old.ndim > 1:
            p_old = p_old.view(p_old.size(0), -1).squeeze(1)
        p_new = torch.sigmoid(new_head(mu))

        b = flair.size(0)
        for j in range(b):
            idx = i0 + j
            if idx >= len(ds.filenames):
                break
            bags["names"].append(ds.filenames[idx])
        bags["gt"].append(gs.cpu().numpy())
        bags["pred_mass"].append(ps.cpu().numpy())
        bags["dices"].append(dice.cpu().numpy())
        bags["topk50"].append(topk.cpu().numpy())
        bags["p_old"].append(p_old.detach().cpu().numpy())
        bags["p_new"].append(p_new.detach().cpu().numpy())
        i0 += b
    n = len(bags["names"])
    return {k: (np.concatenate(v, 0)[:n] if k != "names" else v[:n]) for k, v in bags.items()}


def keep_masks(split: dict, new_tau: float):
    """Named boolean keep arrays (True = keep raw SEG)."""
    return {
        "C0": np.ones(len(split["gt"]), dtype=bool),
        "C1_topk": split["topk50"] >= TOPK_THR_C1,
        "C2_new": split["p_new"] >= new_tau,
        "P_old": split["p_old"] >= OLD_TAU,
    }


def and_keep(masks: dict, keys: tuple[str, ...]) -> np.ndarray:
    k = np.ones(len(next(iter(masks.values()))), dtype=bool)
    for name in keys:
        k &= masks[name]
    return k


def or_keep(masks: dict, keys: tuple[str, ...]) -> np.ndarray:
    k = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    for name in keys:
        k |= masks[name]
    return k


def majority_keep(masks: dict, keys: tuple[str, ...], k_min: int) -> np.ndarray:
    votes = np.zeros(len(next(iter(masks.values()))), dtype=np.int32)
    for name in keys:
        votes += masks[name].astype(np.int32)
    return votes >= k_min


def soft_mean(split: dict) -> np.ndarray:
    # all already in [0,1]
    return (split["topk50"] + split["p_new"] + split["p_old"]) / 3.0


def soft_min(split: dict) -> np.ndarray:
    return np.minimum(np.minimum(split["topk50"], split["p_new"]), split["p_old"])


def soft_prod(split: dict) -> np.ndarray:
    return split["topk50"] * split["p_new"] * split["p_old"]


def eval_keep(split, keep, name: str) -> dict:
    m = metrics_bundle(split["gt"], split["pred_mass"], split["dices"], keep)
    return {"name": name, **m}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (NONE_BEST, OLD_HEAD, NEW_HEAD):
        if not p.exists():
            raise FileNotFoundError(p)

    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    vae.to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # old head
    old_head = PresenceHead(32 * BASE)
    ost = safe_load(OLD_HEAD, map_location="cpu")
    old_head.load_state_dict(ost["head"])
    old_head.to(DEVICE).eval()

    # new head
    nst = safe_load(NEW_HEAD, map_location="cpu")
    in_ch = int(nst.get("in_channels", 1024))
    new_tau = float(nst.get("selected_classification_threshold", NEW_TAU_DEFAULT))
    new_head = LesionPresenceHead(in_ch, hidden_dim=int(nst.get("hidden_dim", 128)),
                                  dropout=float(nst.get("dropout", 0.5)))
    new_head.load_state_dict(nst["presence_head"])
    new_head.to(DEVICE).eval()
    safe_print(f"new_tau={new_tau} old_tau={OLD_TAU} topk_thr={TOPK_THR_C1}")

    val = collect(vae, old_head, new_head, data.val, SEG_BATCH)
    test = collect(vae, old_head, new_head, data.test, SEG_BATCH)

    gate_names = ("C1_topk", "C2_new", "P_old")
    rows_val, rows_test = [], []

    def run_all(split, tag: str):
        masks = keep_masks(split, new_tau)
        out = []
        # singles + empty = C0
        for r in range(0, 4):
            for combo in itertools.combinations(gate_names, r):
                keys = combo
                name = "C0" if not keys else "AND(" + "+".join(keys) + ")"
                keep = and_keep(masks, keys) if keys else masks["C0"]
                out.append(eval_keep(split, keep, name))
                if keys and len(keys) >= 2:
                    name_or = "OR(" + "+".join(keys) + ")"
                    out.append(eval_keep(split, or_keep(masks, keys), name_or))
        # majority of 3
        out.append(eval_keep(split, majority_keep(masks, gate_names, 2), "MAJ2of3"))
        out.append(eval_keep(split, majority_keep(masks, gate_names, 3), "MAJ3of3"))
        # soft scores with fixed thr sweep only reported as best later
        for soft_name, score in (
            ("SOFT_mean", soft_mean(split)),
            ("SOFT_min", soft_min(split)),
            ("SOFT_prod", soft_prod(split)),
        ):
            # use thr=0.5 as a simple operating point + also store curve best on VAL later
            out.append(eval_keep(split, score >= 0.5, f"{soft_name}@0.5"))
        for r in out:
            r["split"] = tag
        return out, masks

    val_rows, val_masks = run_all(val, "val")
    test_rows, test_masks = run_all(test, "test")

    # VAL soft thr search under constraints vs C0
    c0 = next(r for r in val_rows if r["name"] == "C0")
    soft_picks = []
    for soft_name, score_fn in (
        ("SOFT_mean", soft_mean),
        ("SOFT_min", soft_min),
        ("SOFT_prod", soft_prod),
    ):
        sv = score_fn(val)
        st_ = score_fn(test)
        best = None
        curve = []
        for i in range(1, 100):
            thr = i / 100.0
            mv = eval_keep(val, sv >= thr, f"{soft_name}@{thr:.2f}")
            ok = (
                mv["sensitivity"] >= c0["sensitivity"] - SENS_DROP
                and mv["nonempty_dice"] >= c0["nonempty_dice"] - NE_DROP
            )
            curve.append({**mv, "thr": thr, "ok": ok})
            if ok and (best is None or mv["empty_fp_rate"] < best["empty_fp_rate"]):
                best = {**mv, "thr": thr}
        if best is not None:
            mt = eval_keep(test, st_ >= best["thr"], f"{soft_name}@VAL{best['thr']:.2f}")
            soft_picks.append({"soft": soft_name, "val": best, "test": mt})
            safe_print(
                f"PICK {soft_name} thr={best['thr']:.2f} "
                f"VAL fp={best['empty_fp_rate']:.3f} ne={best['nonempty_dice']:.4f} "
                f"TEST fp={mt['empty_fp_rate']:.3f} ne={mt['nonempty_dice']:.4f}"
            )
        else:
            soft_picks.append({"soft": soft_name, "val": None, "test": None, "note": "no constrained thr"})
            safe_print(f"PICK {soft_name}: no constrained thr")

    # rank VAL AND/OR/MAJ by empty_fp among those keeping ne within 0.02 of C0
    floor_ne = c0["nonempty_dice"] - 0.02
    floor_sens = c0["sensitivity"] - 0.02
    ranked = []
    for r in val_rows:
        if r["name"].startswith("SOFT_"):
            continue
        ok = r["nonempty_dice"] >= floor_ne and r["sensitivity"] >= floor_sens
        tr = next(x for x in test_rows if x["name"] == r["name"])
        ranked.append({
            "name": r["name"],
            "ok_loose_floor": ok,
            "val": {k: r[k] for k in (
                "overall_dice", "nonempty_dice", "empty_fp_rate",
                "sensitivity", "specificity", "TT", "FT", "TF", "FF",
            )},
            "test": {k: tr[k] for k in (
                "overall_dice", "nonempty_dice", "empty_fp_rate",
                "sensitivity", "specificity", "TT", "FT", "TF", "FF",
            )},
        })
    ranked.sort(key=lambda x: (not x["ok_loose_floor"], x["val"]["empty_fp_rate"], -x["val"]["nonempty_dice"]))

    # case-level agreement table on VAL empty-FP
    gt = val["gt"] > 0
    empty = ~gt
    seg_pos = val["pred_mass"] > 0
    efp = empty & seg_pos
    etn = empty & ~seg_pos
    ne = gt
    agree = {
        "n_empty_fp": int(efp.sum()),
        "n_empty_tn": int(etn.sum()),
        "n_nonempty": int(ne.sum()),
        "empty_fp_rejected_by": {
            "C1_only": int((efp & ~val_masks["C1_topk"] & val_masks["C2_new"] & val_masks["P_old"]).sum()),
            "C2_only": int((efp & val_masks["C1_topk"] & ~val_masks["C2_new"] & val_masks["P_old"]).sum()),
            "P_only": int((efp & val_masks["C1_topk"] & val_masks["C2_new"] & ~val_masks["P_old"]).sum()),
            "C1_and_C2": int((efp & ~val_masks["C1_topk"] & ~val_masks["C2_new"] & val_masks["P_old"]).sum()),
            "C1_and_P": int((efp & ~val_masks["C1_topk"] & val_masks["C2_new"] & ~val_masks["P_old"]).sum()),
            "C2_and_P": int((efp & val_masks["C1_topk"] & ~val_masks["C2_new"] & ~val_masks["P_old"]).sum()),
            "all3": int((efp & ~val_masks["C1_topk"] & ~val_masks["C2_new"] & ~val_masks["P_old"]).sum()),
            "none_reject": int((efp & val_masks["C1_topk"] & val_masks["C2_new"] & val_masks["P_old"]).sum()),
        },
        "nonempty_rejected_by": {
            "C1": int((ne & ~val_masks["C1_topk"]).sum()),
            "C2": int((ne & ~val_masks["C2_new"]).sum()),
            "P": int((ne & ~val_masks["P_old"]).sum()),
            "any": int((ne & ~(val_masks["C1_topk"] & val_masks["C2_new"] & val_masks["P_old"])).sum()),
            "all3": int((ne & ~val_masks["C1_topk"] & ~val_masks["C2_new"] & ~val_masks["P_old"]).sum()),
        },
    }

    # per-case dump VAL
    case_rows = []
    for i, name in enumerate(val["names"]):
        k1, k2, kp = val_masks["C1_topk"][i], val_masks["C2_new"][i], val_masks["P_old"][i]
        votes = int(k1) + int(k2) + int(kp)
        gt_p = bool(gt[i])
        seg_p = bool(seg_pos[i])
        case_rows.append({
            "sample_id": sample_id(name),
            "patient_id": patient_id(name),
            "gt_present": int(gt_p),
            "raw_seg_present": int(seg_p),
            "group": ("nonempty" if gt_p else ("empty_fp" if seg_p else "empty_tn")),
            "topk50": float(val["topk50"][i]),
            "p_new": float(val["p_new"][i]),
            "p_old": float(val["p_old"][i]),
            "keep_C1": int(k1),
            "keep_C2": int(k2),
            "keep_P": int(kp),
            "votes": votes,
            "keep_AND3": int(k1 and k2 and kp),
            "keep_OR3": int(k1 or k2 or kp),
            "keep_MAJ2": int(votes >= 2),
            "soft_mean": float((val["topk50"][i] + val["p_new"][i] + val["p_old"][i]) / 3),
        })
    write_csv(OUT / "val_case_scores.csv", case_rows)

    # flatten val/test combo tables
    def flat(rows):
        return [{
            "name": r["name"],
            "overall_dice": r["overall_dice"],
            "nonempty_dice": r["nonempty_dice"],
            "empty_fp_rate": r["empty_fp_rate"],
            "sensitivity": r["sensitivity"],
            "specificity": r["specificity"],
            "TT": r["TT"], "FT": r["FT"], "TF": r["TF"], "FF": r["FF"],
        } for r in rows]

    write_csv(OUT / "val_combos.csv", flat(val_rows))
    write_csv(OUT / "test_combos.csv", flat(test_rows))

    report = {
        "thresholds": {
            "topk50": TOPK_THR_C1,
            "new_head": new_tau,
            "old_presence": OLD_TAU,
        },
        "val_agreement_empty_fp": agree,
        "ranked_val_loose_floor": ranked,
        "soft_constrained_picks": soft_picks,
        "val_table": flat(val_rows),
        "test_table": flat(test_rows),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    safe_print("\n=== VAL (key rows) ===")
    for name in (
        "C0", "AND(C1_topk)", "AND(C2_new)", "AND(P_old)",
        "AND(C1_topk+C2_new)", "AND(C1_topk+P_old)", "AND(C2_new+P_old)",
        "AND(C1_topk+C2_new+P_old)", "OR(C1_topk+C2_new+P_old)", "MAJ2of3",
    ):
        r = next((x for x in val_rows if x["name"] == name), None)
        if r:
            safe_print(
                f"{r['name']:32s} dice={r['overall_dice']:.4f} ne={r['nonempty_dice']:.4f} "
                f"fp={r['empty_fp_rate']:.3f} sens={r['sensitivity']:.3f} "
                f"TT={r['TT']} FT={r['FT']} TF={r['TF']} FF={r['FF']}"
            )
    safe_print("\n=== TEST (same rules) ===")
    for name in (
        "C0", "AND(C1_topk)", "AND(C2_new)", "AND(P_old)",
        "AND(C1_topk+C2_new)", "AND(C1_topk+P_old)", "AND(C2_new+P_old)",
        "AND(C1_topk+C2_new+P_old)", "OR(C1_topk+C2_new+P_old)", "MAJ2of3",
    ):
        r = next((x for x in test_rows if x["name"] == name), None)
        if r:
            safe_print(
                f"{r['name']:32s} dice={r['overall_dice']:.4f} ne={r['nonempty_dice']:.4f} "
                f"fp={r['empty_fp_rate']:.3f} sens={r['sensitivity']:.3f} "
                f"TT={r['TT']} FT={r['FT']} TF={r['TF']} FF={r['FF']}"
            )
    safe_print("empty-FP reject breakdown VAL", json.dumps(agree["empty_fp_rejected_by"], indent=2))
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
