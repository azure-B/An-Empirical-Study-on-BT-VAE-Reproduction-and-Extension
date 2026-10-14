"""Post-hoc min-mass filter on dual×2.5D wprod gate to kill tiny empty-FP blobs."""
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
from eval_s5_fuse_reject import collect, DUAL_NE, DUAL_FP

OUT = RESULT_DIR / "s5_wprod_mass_filter"
DUAL = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
C25 = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
PREV = 0.335
BATCH = 16


def eval_keep(bag, keep) -> dict:
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    tp, fp, fn, tn = cm["TT"], cm["FT"], cm["TF"], cm["FF"]
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "auroc": binary_auroc(yt, bag["score"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    # wprod score
    for bag in (val, test):
        s = (bag["p_dual"] ** 1.5) * (bag["p_25d"] ** 1.3)
        bag["score"] = s / (s.max() + 1e-8)

    # sweep score thr × min mass
    masses = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    thrs = [i / 100.0 for i in range(1, 50)]
    rows = []
    best_val = None
    for thr in thrs:
        for mmin in masses:
            keep_v = (val["score"] >= thr) & (val["mass"] >= mmin)
            mv = eval_keep(val, keep_v)
            mv.update({"thr": thr, "min_mass": mmin, "split": "val"})
            if mv["nonempty_dice"] + 1e-9 >= DUAL_NE:
                key = (mv["empty_fp_rate"], -mv["nonempty_dice"])
                if best_val is None or key < (best_val["empty_fp_rate"], -best_val["nonempty_dice"]):
                    best_val = dict(mv)

    if best_val is None:
        safe_print("no VAL point with ne>=0.60")
        # relax
        for thr in thrs:
            for mmin in masses:
                keep_v = (val["score"] >= thr) & (val["mass"] >= mmin)
                mv = eval_keep(val, keep_v)
                j = mv["nonempty_dice"] - 0.5 * mv["empty_fp_rate"]
                if best_val is None or j > best_val.get("_j", -1e9):
                    best_val = dict(mv)
                    best_val["_j"] = j
                    best_val["thr"] = thr
                    best_val["min_mass"] = mmin

    thr, mmin = best_val["thr"], best_val["min_mass"]
    keep_t = (test["score"] >= thr) & (test["mass"] >= mmin)
    mt = eval_keep(test, keep_t)
    mt.update({"thr": thr, "min_mass": mmin, "split": "test"})
    rows.append({"name": "VAL_pick", **{k: best_val[k] for k in (
        "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp")}})
    rows.append({"name": "TEST_pick", **{k: mt[k] for k in (
        "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp")}})

    # oracle on TEST among ne>=0.60
    oracle = None
    for thr in thrs:
        for mmin in masses:
            keep = (test["score"] >= thr) & (test["mass"] >= mmin)
            m = eval_keep(test, keep)
            if m["nonempty_dice"] + 1e-9 >= DUAL_NE:
                if oracle is None or m["empty_fp_rate"] < oracle["empty_fp_rate"]:
                    oracle = dict(m)
                    oracle["thr"] = thr
                    oracle["min_mass"] = mmin
    if oracle:
        rows.append({"name": "TEST_oracle_ne60", **{k: oracle[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp")}})

    # also baseline wprod no mass
    base = eval_keep(test, test["score"] >= 0.01)
    rows.append({"name": "TEST_wprod_only_t0.01", "thr": 0.01, "min_mass": 0,
                 **{k: base[k] for k in ("nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp")}})

    safe_print(
        f"VAL pick thr={thr:.2f} min_mass={mmin} → "
        f"ne={best_val['nonempty_dice']:.4f} fp={best_val['empty_fp_rate']:.3f}"
    )
    safe_print(
        f"TEST pick → ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
        f"sens={mt['sensitivity']:.3f} FN={mt['fn']}"
    )
    if oracle:
        safe_print(
            f"TEST oracle ne60 → thr={oracle['thr']:.2f} mass>={oracle['min_mass']} "
            f"ne={oracle['nonempty_dice']:.4f} fp={oracle['empty_fp_rate']:.3f}"
        )
    safe_print(f"wprod only τ=0.01 → ne={base['nonempty_dice']:.4f} fp={base['empty_fp_rate']:.3f}")

    beats = mt["nonempty_dice"] + 1e-9 >= DUAL_NE and mt["empty_fp_rate"] < PREV - 1e-9
    lines = [
        "# wprod + min-mass filter",
        "",
        f"VAL pick: thr={thr:.2f} min_mass={mmin}",
        f"TEST: ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
        f"sens={mt['sensitivity']:.3f} beats_prev={beats}",
        f"wprod-only τ0.01: ne={base['nonempty_dice']:.4f} fp={base['empty_fp_rate']:.3f}",
    ]
    if oracle:
        lines.append(
            f"TEST oracle ne≥0.60: thr={oracle['thr']:.2f} mass>={oracle['min_mass']} "
            f"ne={oracle['nonempty_dice']:.4f} fp={oracle['empty_fp_rate']:.3f}"
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps({"val": best_val, "test": mt, "oracle": oracle, "base": base, "beats_prev": beats}, indent=2),
        encoding="utf-8",
    )
    write_csv(OUT / "final_metrics.csv", rows)
    safe_print("beats_prev", beats)
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
