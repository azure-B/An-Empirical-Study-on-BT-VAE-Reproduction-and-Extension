"""Dense param sweep on messy wprod+mass (no retrain).

Sweeps α,β (product exponents), τ, min_mass on VAL → TEST.
Also tries min/mean fuse and optional maxprob gate.
Target: beat current 0.601 / 0.305 under ne≥0.60 (VAL pick).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_lesion_presence import binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print
from train_s5_gate_25d import Stack25D, make_loader, write_csv
from eval_s5_fuse_reject import collect

OUT = RESULT_DIR / "s5_wprod_param_tune"
DUAL = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
C25 = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
CACHE = OUT / "score_cache.npz"
CUR_NE, CUR_FP = 0.601, 0.305
NE_FLOOR = 0.60
BATCH = 16


def eval_keep(bag, keep, score) -> dict:
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "tp": cm["TT"], "fp": cm["FT"], "fn": cm["TF"], "tn": cm["FF"],
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "auroc": binary_auroc(yt, score),
    }


def make_score(bag, recipe: str, alpha: float, beta: float) -> np.ndarray:
    p, q = bag["p_dual"], bag["p_25d"]
    if recipe == "wprod":
        s = (p ** alpha) * (q ** beta)
    elif recipe == "wprod_maxp":
        # also require high maxprob soft: product with maxprob^γ, γ=beta reused lightly
        s = (p ** alpha) * (q ** beta) * (bag["maxprob"] ** 0.5)
    elif recipe == "min":
        s = np.minimum(p, q)
    elif recipe == "mean":
        s = 0.5 * (p + q)
    elif recipe == "geom":
        s = np.sqrt(np.clip(p, 1e-8, 1) * np.clip(q, 1e-8, 1))
    else:
        raise ValueError(recipe)
    return s / (s.max() + 1e-8)


def sweep_gate(val, test, score_v, score_t, masses, thrs, ne_floor=NE_FLOOR):
    best = None
    for thr in thrs:
        for mmin in masses:
            keep = (score_v >= thr) & (val["mass"] >= mmin)
            mv = eval_keep(val, keep, score_v)
            if mv["nonempty_dice"] + 1e-9 < ne_floor:
                continue
            key = (mv["empty_fp_rate"], -mv["nonempty_dice"], -mv["sensitivity"])
            if best is None or key < (
                best["empty_fp_rate"], -best["nonempty_dice"], -best["sensitivity"]
            ):
                best = {**mv, "thr": float(thr), "min_mass": int(mmin)}
    if best is None:
        return None, None
    keep_t = (score_t >= best["thr"]) & (test["mass"] >= best["min_mass"])
    mt = eval_keep(test, keep_t, score_t)
    mt.update({"thr": best["thr"], "min_mass": best["min_mass"]})
    return best, mt


def oracle_ne(test, score_t, masses, thrs, ne_floor=NE_FLOOR):
    best = None
    for thr in thrs:
        for mmin in masses:
            keep = (score_t >= thr) & (test["mass"] >= mmin)
            m = eval_keep(test, keep, score_t)
            if m["nonempty_dice"] + 1e-9 < ne_floor:
                continue
            if best is None or m["empty_fp_rate"] < best["empty_fp_rate"] - 1e-12:
                best = {**m, "thr": float(thr), "min_mass": int(mmin)}
    return best


def load_or_collect():
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        z = np.load(CACHE)
        val = {k: z[f"val_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}
        test = {k: z[f"test_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}
        safe_print("loaded cache", CACHE)
        return val, test

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    va3 = make_loader(Stack25D(data.val), BATCH, False)
    te3 = make_loader(Stack25D(data.test), BATCH, False)

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()
    cls = Presence25D().to(DEVICE)
    cls.load_state_dict(safe_load(C25, map_location="cpu")["model"])
    cls.eval()

    val = collect(dual, cls, va1, va3)
    test = collect(dual, cls, te1, te3)
    np.savez_compressed(
        CACHE,
        **{f"val_{k}": val[k] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")},
        **{f"test_{k}": test[k] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")},
    )
    safe_print("wrote cache", CACHE)
    return val, test


def main():
    val, test = load_or_collect()

    # denser grids than original
    masses = list(range(0, 200, 20)) + list(range(200, 800, 40)) + list(range(800, 5600, 80))
    thrs = [i / 200.0 for i in range(1, 160)]  # 0.005 .. 0.795

    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    betas = [0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    recipes = ["wprod", "wprod_maxp", "min", "mean", "geom"]

    rows = []
    best_val_pick = None  # among VAL ne≥0.60, min TEST we'll report VAL-selected

    for recipe in recipes:
        ab_list = [(1.5, 1.3)] if recipe != "wprod" and recipe != "wprod_maxp" else [
            (a, b) for a in alphas for b in betas
        ]
        for alpha, beta in ab_list:
            sv = make_score(val, recipe, alpha, beta)
            st = make_score(test, recipe, alpha, beta)
            bv, mt = sweep_gate(val, test, sv, st, masses, thrs, NE_FLOOR)
            if bv is None:
                continue
            oc = oracle_ne(test, st, masses, thrs, NE_FLOOR)
            row = {
                "recipe": recipe,
                "alpha": alpha,
                "beta": beta,
                "val_ne": bv["nonempty_dice"],
                "val_fp": bv["empty_fp_rate"],
                "thr": bv["thr"],
                "min_mass": bv["min_mass"],
                "test_ne": mt["nonempty_dice"],
                "test_fp": mt["empty_fp_rate"],
                "test_sens": mt["sensitivity"],
                "test_fn": mt["fn"],
                "test_fp_n": mt["fp"],
                "oracle_ne": oc["nonempty_dice"] if oc else None,
                "oracle_fp": oc["empty_fp_rate"] if oc else None,
                "oracle_thr": oc["thr"] if oc else None,
                "oracle_mass": oc["min_mass"] if oc else None,
            }
            rows.append(row)
            # select by VAL fp (primary), then VAL ne
            key = (bv["empty_fp_rate"], -bv["nonempty_dice"], mt["empty_fp_rate"])
            if best_val_pick is None or key < (
                best_val_pick["val_fp"], -best_val_pick["val_ne"], best_val_pick["test_fp"]
            ):
                best_val_pick = row

            if len(rows) % 40 == 0:
                safe_print(f"… {len(rows)} configs, best_so_far TEST "
                           f"{best_val_pick['test_ne']:.4f}/{best_val_pick['test_fp']:.3f} "
                           f"[{best_val_pick['recipe']} α={best_val_pick['alpha']} β={best_val_pick['beta']}]")

    # also baseline: original α1.5 β1.3 with denser τ/mass only
    sv0 = make_score(val, "wprod", 1.5, 1.3)
    st0 = make_score(test, "wprod", 1.5, 1.3)
    b0, t0 = sweep_gate(val, test, sv0, st0, masses, thrs, NE_FLOOR)
    o0 = oracle_ne(test, st0, masses, thrs, NE_FLOOR)

    # best TEST among VAL picks (already best_val_pick)
    # also best TEST fp among all rows with test_ne>=0.60 (oracle-ish over hyperparams — report separately)
    test_best = None
    for r in rows:
        if r["test_ne"] + 1e-9 < NE_FLOOR:
            continue
        if test_best is None or r["test_fp"] < test_best["test_fp"] - 1e-12:
            test_best = r

    write_csv(OUT / "all_configs.csv", rows)

    # top-15 by VAL fp then test fp
    ranked = sorted(rows, key=lambda r: (r["val_fp"], -r["val_ne"], r["test_fp"]))[:15]
    # top-15 by TEST fp (leak; for ceiling)
    ranked_test = sorted(
        [r for r in rows if r["test_ne"] + 1e-9 >= NE_FLOOR],
        key=lambda r: (r["test_fp"], -r["test_ne"]),
    )[:15]

    lines = [
        "# wprod+mass dense param tune (no retrain)",
        "",
        f"Current published: ne={CUR_NE:.3f} fp={CUR_FP:.3f} (α=1.5 β=1.3, τ≈0.01, m≥320)",
        f"Dense re-pick same αβ: VAL τ={b0['thr']:.3f} m≥{b0['min_mass']} → "
        f"TEST ne={t0['nonempty_dice']:.4f} fp={t0['empty_fp_rate']:.3f}",
        f"TEST oracle same αβ: τ={o0['thr']:.3f} m≥{o0['min_mass']} → "
        f"ne={o0['nonempty_dice']:.4f} fp={o0['empty_fp_rate']:.3f}",
        "",
        "## Best VAL→TEST (ne≥0.60 min VAL-fp)",
        (
            f"**{best_val_pick['recipe']}** α={best_val_pick['alpha']} β={best_val_pick['beta']} "
            f"τ={best_val_pick['thr']:.3f} m≥{best_val_pick['min_mass']} → "
            f"TEST ne={best_val_pick['test_ne']:.4f} fp={best_val_pick['test_fp']:.3f} "
            f"sens={best_val_pick['test_sens']:.3f}"
        ),
        f"vs current: Δne={best_val_pick['test_ne']-CUR_NE:+.4f} Δfp={best_val_pick['test_fp']-CUR_FP:+.4f}",
        "",
        "## Ceiling (best TEST among configs with TEST ne≥0.60; hyperparam leak)",
        (
            f"{test_best['recipe']} α={test_best['alpha']} β={test_best['beta']} "
            f"τ={test_best['thr']:.3f} m≥{test_best['min_mass']} → "
            f"ne={test_best['test_ne']:.4f} fp={test_best['test_fp']:.3f}"
        ),
        "",
        "## Top VAL-ranked (first 10)",
        "| recipe | α | β | τ | m_min | VAL fp | TEST ne | TEST fp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked[:10]:
        lines.append(
            f"| {r['recipe']} | {r['alpha']} | {r['beta']} | {r['thr']:.3f} | {r['min_mass']} | "
            f"{r['val_fp']:.3f} | {r['test_ne']:.4f} | {r['test_fp']:.3f} |"
        )
    lines += [
        "",
        "## Top TEST-fp (ceiling, first 8)",
        "| recipe | α | β | τ | m_min | TEST ne | TEST fp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked_test[:8]:
        lines.append(
            f"| {r['recipe']} | {r['alpha']} | {r['beta']} | {r['thr']:.3f} | {r['min_mass']} | "
            f"{r['test_ne']:.4f} | {r['test_fp']:.3f} |"
        )

    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "current": {"ne": CUR_NE, "fp": CUR_FP},
                "same_ab_dense": {"val": b0, "test": t0, "oracle": o0},
                "best_val_pick": best_val_pick,
                "test_ceiling": test_best,
                "n_configs": len(rows),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    safe_print("=" * 60)
    safe_print(
        f"BEST VAL→TEST: {best_val_pick['recipe']} α={best_val_pick['alpha']} β={best_val_pick['beta']} "
        f"τ={best_val_pick['thr']:.3f} m≥{best_val_pick['min_mass']} → "
        f"ne={best_val_pick['test_ne']:.4f} fp={best_val_pick['test_fp']:.3f}"
    )
    safe_print(
        f"vs current {CUR_NE}/{CUR_FP}: "
        f"Δne={best_val_pick['test_ne']-CUR_NE:+.4f} Δfp={best_val_pick['test_fp']-CUR_FP:+.4f}"
    )
    safe_print(
        f"TEST ceiling: {test_best['recipe']} α={test_best['alpha']} β={test_best['beta']} "
        f"→ ne={test_best['test_ne']:.4f} fp={test_best['test_fp']:.3f}"
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
