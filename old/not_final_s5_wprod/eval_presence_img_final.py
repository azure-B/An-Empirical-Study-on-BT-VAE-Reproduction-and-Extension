"""Write imgcls COMPARE after CSV crash (loads best ckpt)."""
from __future__ import annotations

import csv
import json

import numpy as np

from data_load_BT import datasets
from models_presence_img import PresenceImgCls
from train_original_bp import DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print
from train_presence_img import OUT, collect_scores, pick_thr_val, tf_metrics, write_csv

BEST = OUT / "checkpoints" / "imgcls_best_auroc.pt"


def main():
    data = datasets.bp_train_30(batch_size=16, num_workers=NUM_WORKERS, pin_memory=False)
    model = PresenceImgCls().to(DEVICE)
    st = safe_load(BEST, map_location="cpu")
    model.load_state_dict(st["model"])
    model.eval()
    best_auroc = float(st.get("val_auroc", 0.0))

    yv, sv = collect_scores(model, data.val_loader)
    yt, stt = collect_scores(model, data.test_loader)
    rows = []
    m50 = tf_metrics(yt, stt, 0.5)
    rows.append({
        "name": "TEST_thr0.50", "thr": 0.5,
        "accuracy": m50["accuracy"], "sensitivity": m50["sensitivity"],
        "specificity": m50["specificity"], "precision": m50["precision"],
        "f1": m50["f1"], "auroc": m50["auroc"],
        "tp": m50["tp"], "tn": m50["tn"], "fp": m50["fp"], "fn": m50["fn"],
    })
    for mode in ("acc", "f1", "bal"):
        pv = pick_thr_val(yv, sv, mode)
        mt = tf_metrics(yt, stt, pv["thr"])
        rows.append({
            "name": f"TEST_{mode}", "thr": pv["thr"],
            "accuracy": mt["accuracy"], "sensitivity": mt["sensitivity"],
            "specificity": mt["specificity"], "precision": mt["precision"],
            "f1": mt["f1"], "auroc": mt["auroc"],
            "tp": mt["tp"], "tn": mt["tn"], "fp": mt["fp"], "fn": mt["fn"],
            "val_accuracy": pv["accuracy"],
        })
        safe_print(
            f"{mode} τ={pv['thr']:.2f} acc={mt['accuracy']:.3f} "
            f"sens={mt['sensitivity']:.3f} spec={mt['specificity']:.3f} "
            f"F1={mt['f1']:.3f}"
        )

    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps({"best_val_auroc": best_auroc, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Presence image classifier (step1)",
        "",
        "ResNet18 on FLAIR only · no VAE μ",
        f"best VAL AUROC={best_auroc:.4f}",
        "",
        "## TEST T/F (primary)",
    ]
    for r in rows:
        lines.append(
            f"- {r['name']}: acc={r['accuracy']:.3f} sens={r['sensitivity']:.3f} "
            f"spec={r['specificity']:.3f} F1={r['f1']:.3f} "
            f"TP/TN/FP/FN={r['tp']}/{r['tn']}/{r['fp']}/{r['fn']} τ={r['thr']:.2f}"
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
