"""Sweep tau 0.01..0.15 on dual-role ckpts; S5 inference gated by cls P."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from train_dual_role import BATCH, collect_split, eval_gate
from train_lesion_presence import metrics_bundle
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print

OUT = RESULT_DIR / "dual_role_tau_sweep_001_015"
CKPTS = {
    "none": RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt",
    "emptyfp": RESULT_DIR / "dual_role_train_emptyfp" / "checkpoints" / "dual_role_best_auroc.pt",
}
TAUS = [i / 100.0 for i in range(1, 16)]  # 0.01 .. 0.15


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    all_rows = []

    for tag, ckpt in CKPTS.items():
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        safe_print(f"load {tag} {ckpt}")
        model = DualRoleVAE(base=BASE).to(DEVICE)
        model.load_state_dict(safe_load(ckpt, map_location="cpu")["model"])
        model.freeze_skip()
        model.eval()

        val = collect_split(model, data.val_loader)
        test = collect_split(model, data.test_loader)
        ones = np.ones(len(test["gt"]), dtype=bool)
        raw = metrics_bundle(test["gt"], test["mass_s5"], test["dice_s5"], ones)
        safe_print(
            f"{tag} S5_raw TEST ne={raw['nonempty_dice']:.4f} fp={raw['empty_fp_rate']:.3f}"
        )

        for thr in TAUS:
            mv = eval_gate(val, thr)
            mt = eval_gate(test, thr)
            row = {
                "cls_init": tag,
                "tau": thr,
                "split": "test",
                "overall_dice": mt["overall_dice"],
                "nonempty_dice": mt["nonempty_dice"],
                "empty_fp_rate": mt["empty_fp_rate"],
                "sensitivity": mt["sensitivity"],
                "presence_recall": mt["presence_recall"],
                "TT": mt["TT"], "FT": mt["FT"], "TF": mt["TF"], "FF": mt["FF"],
                "val_nonempty_dice": mv["nonempty_dice"],
                "val_empty_fp_rate": mv["empty_fp_rate"],
                "val_sensitivity": mv["sensitivity"],
            }
            all_rows.append(row)
            safe_print(
                f"  τ={thr:.2f} TEST ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
                f"sens={mt['sensitivity']:.3f} TT/FT/TF={mt['TT']}/{mt['FT']}/{mt['TF']}"
            )

    with (OUT / "sweep_test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# τ sweep 0.01–0.15 (cls P → S5 gate)",
        "",
        "## TEST none-cls",
        "| τ | ne | FP | sens | TT | FT | TF |",
        "|--:|---:|---:|-----:|---:|---:|---:|",
    ]
    for r in all_rows:
        if r["cls_init"] != "none":
            continue
        lines.append(
            f"| {r['tau']:.2f} | {r['nonempty_dice']:.4f} | {r['empty_fp_rate']:.3f} | "
            f"{r['sensitivity']:.3f} | {r['TT']} | {r['FT']} | {r['TF']} |"
        )
    lines += [
        "",
        "## TEST emptyfp-cls",
        "| τ | ne | FP | sens | TT | FT | TF |",
        "|--:|---:|---:|-----:|---:|---:|---:|",
    ]
    for r in all_rows:
        if r["cls_init"] != "emptyfp":
            continue
        lines.append(
            f"| {r['tau']:.2f} | {r['nonempty_dice']:.4f} | {r['empty_fp_rate']:.3f} | "
            f"{r['sensitivity']:.3f} | {r['TT']} | {r['FT']} | {r['TF']} |"
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
