"""S5 skip SEG + presence gates: hybrid (none-gate / S5-mask) and native S5 gates.

Case-by-case AND/OR/MAJ/soft like eval_presence_combo.py. No training.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead
from models_presence import PresenceHead
from train_lesion_presence import (
    PIXEL_THR,
    TOPK_C1,
    TOPK_THR_C1,
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
    SEG_BATCH,
    safe_load,
    safe_print,
)

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OLD_HEAD = RESULT_DIR / "none_presence" / "head_best.pt"
NEW_HEAD = RESULT_DIR / "lesion_presence" / "checkpoints" / "lesion_presence_head_best_constrained.pt"
OUT = RESULT_DIR / "s5_presence_combo"
OLD_TAU = 0.20
NEW_TAU_DEFAULT = 0.11
SENS_DROP = 0.01
NE_DROP = 0.01
# S5 batch smaller (skip memory)
S5_BATCH = 8


@torch.inference_mode()
def seg_forward_det_s5(vae_s5, flair):
    """Deterministic S5: decode μ with encoder skips (no reparameterize)."""
    flair, _ = _prep(flair, flair)
    mu, logvar, _z, skips = vae_s5.encoder.forward_features(flair)
    out = vae_s5.decoder(mu, skips)
    return out, mu, logvar


@torch.inference_mode()
def collect_paired(vae_none, vae_s5, old_head, new_head, ds, batch_size: int):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    vae_none.eval()
    vae_s5.eval()
    old_head.eval()
    new_head.eval()
    bags = {
        "gt": [], "names": [],
        "pred_mass_none": [], "dices_none": [], "topk_none": [],
        "p_old_none": [], "p_new_none": [],
        "pred_mass_s5": [], "dices_s5": [], "topk_s5": [],
        "p_old_s5": [], "p_new_s5": [],
    }
    i0 = 0

    def _scores(out, mu, seg_b):
        prob = torch.sigmoid(out)
        flat = prob.flatten(1)
        topk = torch.topk(flat, k=TOPK_C1, dim=1).values.mean(1)
        pred = (prob >= PIXEL_THR).float()
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        p_old = torch.sigmoid(old_head(mu))
        if p_old.ndim > 1:
            p_old = p_old.view(p_old.size(0), -1).squeeze(1)
        p_new = torch.sigmoid(new_head(mu))
        return ps, gs, dice, topk, p_old, p_new

    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_b = seg.unsqueeze(1) if seg.ndim == 3 else seg
        out_n, mu_n, _ = seg_forward_det(vae_none, flair)
        out_s, mu_s, _ = seg_forward_det_s5(vae_s5, flair)
        ps_n, gs, dice_n, topk_n, po_n, pn_n = _scores(out_n, mu_n, seg_b)
        ps_s, _, dice_s, topk_s, po_s, pn_s = _scores(out_s, mu_s, seg_b)

        b = flair.size(0)
        for j in range(b):
            idx = i0 + j
            if idx >= len(ds.filenames):
                break
            bags["names"].append(ds.filenames[idx])
        bags["gt"].append(gs.cpu().numpy())
        bags["pred_mass_none"].append(ps_n.cpu().numpy())
        bags["dices_none"].append(dice_n.cpu().numpy())
        bags["topk_none"].append(topk_n.cpu().numpy())
        bags["p_old_none"].append(po_n.detach().cpu().numpy())
        bags["p_new_none"].append(pn_n.detach().cpu().numpy())
        bags["pred_mass_s5"].append(ps_s.cpu().numpy())
        bags["dices_s5"].append(dice_s.cpu().numpy())
        bags["topk_s5"].append(topk_s.cpu().numpy())
        bags["p_old_s5"].append(po_s.detach().cpu().numpy())
        bags["p_new_s5"].append(pn_s.detach().cpu().numpy())
        i0 += b
    n = len(bags["names"])
    return {k: (np.concatenate(v, 0)[:n] if k != "names" else v[:n]) for k, v in bags.items()}


def as_split(bag, prefix: str, use_seg: str):
    """Build metrics split: gates from `prefix`, masks/dice from `use_seg`."""
    return {
        "gt": bag["gt"],
        "names": bag["names"],
        "pred_mass": bag[f"pred_mass_{use_seg}"],
        "dices": bag[f"dices_{use_seg}"],
        "topk50": bag[f"topk_{prefix}"],
        "p_old": bag[f"p_old_{prefix}"],
        "p_new": bag[f"p_new_{prefix}"],
    }


def keep_masks(split: dict, new_tau: float, topk_thr: float):
    return {
        "C0": np.ones(len(split["gt"]), dtype=bool),
        "C1_topk": split["topk50"] >= topk_thr,
        "C2_new": split["p_new"] >= new_tau,
        "P_old": split["p_old"] >= OLD_TAU,
    }


def and_keep(masks, keys):
    k = np.ones(len(next(iter(masks.values()))), dtype=bool)
    for name in keys:
        k &= masks[name]
    return k


def or_keep(masks, keys):
    k = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    for name in keys:
        k |= masks[name]
    return k


def majority_keep(masks, keys, k_min):
    votes = np.zeros(len(next(iter(masks.values()))), dtype=np.int32)
    for name in keys:
        votes += masks[name].astype(np.int32)
    return votes >= k_min


def soft_mean(split):
    return (split["topk50"] + split["p_new"] + split["p_old"]) / 3.0


def soft_min(split):
    return np.minimum(np.minimum(split["topk50"], split["p_new"]), split["p_old"])


def soft_prod(split):
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


def pick_topk_thr(val_split, floor_ne: float, floor_sens: float):
    """VAL thr for topk under ne/sens floor vs raw C0 of that split."""
    best = None
    curve = []
    score = val_split["topk50"]
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_keep(val_split, score >= thr, f"topk@{thr:.2f}")
        ok = m["nonempty_dice"] >= floor_ne and m["sensitivity"] >= floor_sens
        curve.append({**m, "thr": thr, "ok": ok})
        if ok and (best is None or m["empty_fp_rate"] < best["empty_fp_rate"]):
            best = {**m, "thr": thr}
    return best, curve


def run_combo_table(split, new_tau: float, topk_thr: float, tag: str):
    masks = keep_masks(split, new_tau, topk_thr)
    gate_names = ("C1_topk", "C2_new", "P_old")
    out = []
    for r in range(0, 4):
        for combo in itertools.combinations(gate_names, r):
            keys = combo
            name = "C0" if not keys else "AND(" + "+".join(keys) + ")"
            keep = and_keep(masks, keys) if keys else masks["C0"]
            out.append(eval_keep(split, keep, name))
            if keys and len(keys) >= 2:
                out.append(eval_keep(split, or_keep(masks, keys), "OR(" + "+".join(keys) + ")"))
    out.append(eval_keep(split, majority_keep(masks, gate_names, 2), "MAJ2of3"))
    out.append(eval_keep(split, majority_keep(masks, gate_names, 3), "MAJ3of3"))
    for soft_name, score in (
        ("SOFT_mean", soft_mean(split)),
        ("SOFT_min", soft_min(split)),
        ("SOFT_prod", soft_prod(split)),
    ):
        out.append(eval_keep(split, score >= 0.5, f"{soft_name}@0.5"))
    for r in out:
        r["split"] = tag
    return out, masks


def soft_constrained(val, test, c0_val):
    picks = []
    for soft_name, score_fn in (
        ("SOFT_mean", soft_mean),
        ("SOFT_min", soft_min),
        ("SOFT_prod", soft_prod),
    ):
        sv, st = score_fn(val), score_fn(test)
        best = None
        for i in range(1, 100):
            thr = i / 100.0
            mv = eval_keep(val, sv >= thr, f"{soft_name}@{thr:.2f}")
            ok = (
                mv["sensitivity"] >= c0_val["sensitivity"] - SENS_DROP
                and mv["nonempty_dice"] >= c0_val["nonempty_dice"] - NE_DROP
            )
            if ok and (best is None or mv["empty_fp_rate"] < best["empty_fp_rate"]):
                best = {**mv, "thr": thr}
        if best is None:
            picks.append({"soft": soft_name, "val": None, "test": None})
            continue
        mt = eval_keep(test, st >= best["thr"], f"{soft_name}@VAL{best['thr']:.2f}")
        picks.append({"soft": soft_name, "val": best, "test": mt})
        safe_print(
            f"  PICK {soft_name} thr={best['thr']:.2f} "
            f"VAL fp={best['empty_fp_rate']:.3f} ne={best['nonempty_dice']:.4f} "
            f"TEST fp={mt['empty_fp_rate']:.3f} ne={mt['nonempty_dice']:.4f}"
        )
    return picks


def efp_breakdown(split, masks):
    gt = split["gt"] > 0
    empty = ~gt
    seg_pos = split["pred_mass"] > 0
    efp = empty & seg_pos
    ne = gt
    return {
        "n_empty_fp": int(efp.sum()),
        "n_nonempty": int(ne.sum()),
        "empty_fp_rejected_by": {
            "C1_only": int((efp & ~masks["C1_topk"] & masks["C2_new"] & masks["P_old"]).sum()),
            "C2_only": int((efp & masks["C1_topk"] & ~masks["C2_new"] & masks["P_old"]).sum()),
            "P_only": int((efp & masks["C1_topk"] & masks["C2_new"] & ~masks["P_old"]).sum()),
            "C1_and_C2": int((efp & ~masks["C1_topk"] & ~masks["C2_new"] & masks["P_old"]).sum()),
            "C1_and_P": int((efp & ~masks["C1_topk"] & masks["C2_new"] & ~masks["P_old"]).sum()),
            "C2_and_P": int((efp & masks["C1_topk"] & ~masks["C2_new"] & ~masks["P_old"]).sum()),
            "all3": int((efp & ~masks["C1_topk"] & ~masks["C2_new"] & ~masks["P_old"]).sum()),
            "none_reject": int((efp & masks["C1_topk"] & masks["C2_new"] & masks["P_old"]).sum()),
        },
        "nonempty_rejected_by": {
            "C1": int((ne & ~masks["C1_topk"]).sum()),
            "C2": int((ne & ~masks["C2_new"]).sum()),
            "P": int((ne & ~masks["P_old"]).sum()),
            "any": int((ne & ~(masks["C1_topk"] & masks["C2_new"] & masks["P_old"])).sum()),
        },
    }


def dump_cases(path: Path, split, masks, mode: str):
    gt = split["gt"] > 0
    seg_pos = split["pred_mass"] > 0
    rows = []
    for i, name in enumerate(split["names"]):
        k1, k2, kp = masks["C1_topk"][i], masks["C2_new"][i], masks["P_old"][i]
        votes = int(k1) + int(k2) + int(kp)
        gt_p = bool(gt[i])
        seg_p = bool(seg_pos[i])
        rows.append({
            "mode": mode,
            "sample_id": sample_id(name),
            "patient_id": patient_id(name),
            "gt_present": int(gt_p),
            "raw_seg_present": int(seg_p),
            "group": ("nonempty" if gt_p else ("empty_fp" if seg_p else "empty_tn")),
            "dice_raw": float(split["dices"][i]),
            "topk50": float(split["topk50"][i]),
            "p_new": float(split["p_new"][i]),
            "p_old": float(split["p_old"][i]),
            "keep_C1": int(k1),
            "keep_C2": int(k2),
            "keep_P": int(kp),
            "votes": votes,
            "keep_AND3": int(k1 and k2 and kp),
            "keep_OR3": int(k1 or k2 or kp),
            "keep_MAJ2": int(votes >= 2),
        })
    write_csv(path, rows)


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


def print_key(rows, title: str):
    safe_print(f"\n=== {title} ===")
    for name in (
        "C0", "AND(C1_topk)", "AND(C2_new)", "AND(P_old)",
        "AND(C1_topk+C2_new)", "AND(C1_topk+P_old)", "AND(C2_new+P_old)",
        "AND(C1_topk+C2_new+P_old)", "OR(C1_topk+C2_new+P_old)", "MAJ2of3",
    ):
        r = next((x for x in rows if x["name"] == name), None)
        if r:
            safe_print(
                f"{r['name']:32s} dice={r['overall_dice']:.4f} ne={r['nonempty_dice']:.4f} "
                f"fp={r['empty_fp_rate']:.3f} sens={r['sensitivity']:.3f} "
                f"TT={r['TT']} FT={r['FT']} TF={r['TF']} FF={r['FF']}"
            )


def eval_mode(name: str, val_bag, test_bag, gate_src: str, seg_src: str,
              new_tau: float, topk_thr: float, recalibrate_topk: bool):
    safe_print(f"\n######## MODE {name}: gate={gate_src} seg={seg_src} ########")
    val = as_split(val_bag, gate_src, seg_src)
    test = as_split(test_bag, gate_src, seg_src)
    c0v = eval_keep(val, np.ones(len(val["gt"]), dtype=bool), "C0")
    thr = topk_thr
    topk_pick = None
    if recalibrate_topk:
        # keep ne within 0.02 of S5/raw C0, minimize FP
        best, _ = pick_topk_thr(
            val,
            floor_ne=c0v["nonempty_dice"] - 0.02,
            floor_sens=c0v["sensitivity"] - 0.02,
        )
        if best is not None:
            thr = float(best["thr"])
            topk_pick = best
            safe_print(f"  recalibrated topk thr={thr:.2f} VAL fp={best['empty_fp_rate']:.3f} ne={best['nonempty_dice']:.4f}")
        else:
            safe_print("  topk recalibrate: no thr under floor; keep default")

    # For native S5 heads, also optionally retune P/C2? Keep fixed τ first; soft search covers continuous.
    val_rows, val_masks = run_combo_table(val, new_tau, thr, "val")
    test_rows, test_masks = run_combo_table(test, new_tau, thr, "test")
    soft_picks = soft_constrained(val, test, c0v)
    agree = efp_breakdown(val, val_masks)
    dump_cases(OUT / f"val_cases_{name}.csv", val, val_masks, name)
    write_csv(OUT / f"val_combos_{name}.csv", flat(val_rows))
    write_csv(OUT / f"test_combos_{name}.csv", flat(test_rows))
    print_key(val_rows, f"{name} VAL")
    print_key(test_rows, f"{name} TEST")
    safe_print(f"{name} empty-FP reject", json.dumps(agree["empty_fp_rejected_by"]))
    return {
        "mode": name,
        "gate_src": gate_src,
        "seg_src": seg_src,
        "topk_thr": thr,
        "new_tau": new_tau,
        "old_tau": OLD_TAU,
        "topk_recalibrate": topk_pick,
        "soft_constrained_picks": soft_picks,
        "val_agreement": agree,
        "val_table": flat(val_rows),
        "test_table": flat(test_rows),
        "c0_val": {k: c0v[k] for k in ("overall_dice", "nonempty_dice", "empty_fp_rate", "sensitivity")},
        "c0_test": flat([next(r for r in test_rows if r["name"] == "C0")])[0],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (NONE_BEST, S5_BEST, OLD_HEAD, NEW_HEAD):
        if not p.exists():
            raise FileNotFoundError(p)

    data = datasets.bp_train_30(batch_size=S5_BATCH, num_workers=NUM_WORKERS)

    vae_none = OrigVAE(base=BASE)
    vae_none.load_state_dict(safe_load(NONE_BEST, map_location="cpu")["model"])
    vae_none.to(DEVICE).eval()
    for p in vae_none.parameters():
        p.requires_grad = False

    vae_s5 = SkipVAE(base=BASE)
    vae_s5.load_state_dict(safe_load(S5_BEST, map_location="cpu")["model"])
    vae_s5.to(DEVICE).eval()
    for p in vae_s5.parameters():
        p.requires_grad = False

    old_head = PresenceHead(32 * BASE)
    old_head.load_state_dict(safe_load(OLD_HEAD, map_location="cpu")["head"])
    old_head.to(DEVICE).eval()

    nst = safe_load(NEW_HEAD, map_location="cpu")
    new_tau = float(nst.get("selected_classification_threshold", NEW_TAU_DEFAULT))
    new_head = LesionPresenceHead(
        int(nst.get("in_channels", 1024)),
        hidden_dim=int(nst.get("hidden_dim", 128)),
        dropout=float(nst.get("dropout", 0.5)),
    )
    new_head.load_state_dict(nst["presence_head"])
    new_head.to(DEVICE).eval()
    safe_print(f"new_tau={new_tau} old_tau={OLD_TAU} default_topk={TOPK_THR_C1}")

    safe_print("collecting VAL…")
    val_bag = collect_paired(vae_none, vae_s5, old_head, new_head, data.val, S5_BATCH)
    safe_print("collecting TEST…")
    test_bag = collect_paired(vae_none, vae_s5, old_head, new_head, data.test, S5_BATCH)

    # baseline none raw for reference
    none_c0_t = eval_keep(
        as_split(test_bag, "none", "none"),
        np.ones(len(test_bag["gt"]), dtype=bool),
        "none_C0",
    )
    s5_c0_t = eval_keep(
        as_split(test_bag, "s5", "s5"),
        np.ones(len(test_bag["gt"]), dtype=bool),
        "s5_C0",
    )
    safe_print(
        f"REF TEST none C0 ne={none_c0_t['nonempty_dice']:.4f} fp={none_c0_t['empty_fp_rate']:.3f} | "
        f"s5 C0 ne={s5_c0_t['nonempty_dice']:.4f} fp={s5_c0_t['empty_fp_rate']:.3f}"
    )

    modes = []
    # A: hybrid — calibrated none gates, S5 masks (raise ne while keeping gate quality)
    modes.append(eval_mode(
        "hybrid_noneGate_s5Seg", val_bag, test_bag,
        gate_src="none", seg_src="s5",
        new_tau=new_tau, topk_thr=TOPK_THR_C1, recalibrate_topk=False,
    ))
    # B: native S5 gates + S5 seg, recalibrate topk on VAL
    modes.append(eval_mode(
        "native_s5Gate_s5Seg", val_bag, test_bag,
        gate_src="s5", seg_src="s5",
        new_tau=new_tau, topk_thr=TOPK_THR_C1, recalibrate_topk=True,
    ))
    # C: control — none gates + none seg (previous combo baseline)
    modes.append(eval_mode(
        "control_noneGate_noneSeg", val_bag, test_bag,
        gate_src="none", seg_src="none",
        new_tau=new_tau, topk_thr=TOPK_THR_C1, recalibrate_topk=False,
    ))

    # Pareto snapshot: TEST rows with ne >= none_C0 ne and fp <= presence-ish
    summary_rows = []
    for mode in modes:
        for r in mode["test_table"]:
            summary_rows.append({
                "mode": mode["mode"],
                **r,
                "beats_none_ne": r["nonempty_dice"] >= none_c0_t["nonempty_dice"] - 1e-9,
                "beats_s5_fp": r["empty_fp_rate"] <= s5_c0_t["empty_fp_rate"] - 1e-9,
            })
    write_csv(OUT / "test_all_modes.csv", summary_rows)

    report = {
        "ref_test": {
            "none_C0": flat([none_c0_t])[0],
            "s5_C0": flat([s5_c0_t])[0],
        },
        "modes": modes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # short COMPARE
    lines = [
        "# S5 × presence combo",
        "",
        "Hard reject. Hybrid = none μ gates + S5 mask. Native = S5 μ gates (topk recal on VAL) + S5 mask.",
        "",
        "## TEST refs",
        f"- none C0: ne={none_c0_t['nonempty_dice']:.4f} fp={none_c0_t['empty_fp_rate']:.3f}",
        f"- s5 C0: ne={s5_c0_t['nonempty_dice']:.4f} fp={s5_c0_t['empty_fp_rate']:.3f}",
        "",
    ]
    for mode in modes:
        lines.append(f"## {mode['mode']} (topk={mode['topk_thr']:.4f})")
        lines.append("| method | ne Dice | empty FP | sens |")
        lines.append("|--------|--------:|---------:|-----:|")
        for name in (
            "C0", "AND(C1_topk)", "AND(C2_new)", "AND(P_old)",
            "AND(C1_topk+C2_new)", "AND(C1_topk+P_old)", "AND(C2_new+P_old)",
            "AND(C1_topk+C2_new+P_old)", "MAJ2of3",
        ):
            r = next(x for x in mode["test_table"] if x["name"] == name)
            lines.append(
                f"| {name} | {r['nonempty_dice']:.4f} | {r['empty_fp_rate']:.3f} | {r['sensitivity']:.3f} |"
            )
        lines.append("")
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
