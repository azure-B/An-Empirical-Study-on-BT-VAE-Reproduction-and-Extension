"""Finish cascade report after history.csv crash (loads best_tau)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_bt_cascade import BTCascadeVAE
from train_bt_cascade import (
    OUT,
    binary_auroc,
    collect,
    tau_stats,
)
from train_lesion_presence import metrics_bundle
from train_original_bp import BASE, DEVICE, NUM_WORKERS, safe_load, safe_print

CKPT = OUT / "checkpoints" / "bt_cascade_best_tau.pt"


def tf_from_keep(gt, keep):
    yt = gt > 0
    yp = keep.astype(bool)
    tp = int((yp & yt).sum())
    tn = int((~yp & ~yt).sum())
    fp = int((yp & ~yt).sum())
    fn = int((~yp & yt).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-8, prec + sens)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": float(sens), "specificity": float(spec),
        "precision": float(prec), "f1": float(f1), "accuracy": float(acc),
        "n_pos": int(yt.sum()), "n_neg": int((~yt).sum()),
    }


def main():
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    data = datasets.bp_train_30(batch_size=8, num_workers=NUM_WORKERS, pin_memory=False)
    model = BTCascadeVAE(base=BASE).to(DEVICE)
    st = safe_load(CKPT, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_encoder()
    model.eval()
    ts = tau_stats(model)
    safe_print(f"final from {CKPT.name} τ={ts['tau_prob']:.4f} L={ts['tau_logit']:.4f}")

    val = collect(model, data.val_loader)
    test = collect(model, data.test_loader)

    def pack(name, bag):
        keep = bag["keep"].astype(bool)
        m = metrics_bundle(bag["gt"], bag["mass_seg"], bag["dice_seg"], keep)
        m["name"] = name
        m["thr"] = ts["tau_prob"]
        m["tau_logit"] = ts["tau_logit"]
        m["presence_auroc"] = binary_auroc(bag["gt"] > 0, bag["cls_prob"])
        m.update(tf_from_keep(bag["gt"], keep))
        return m

    ones = np.ones(len(test["gt"]), dtype=bool)
    raw_t = metrics_bundle(test["gt"], test["mass_seg"], test["dice_seg"], ones)
    rows = [{
        "name": "TEST_seg_raw",
        "overall_dice": raw_t["overall_dice"],
        "nonempty_dice": raw_t["nonempty_dice"],
        "empty_fp_rate": raw_t["empty_fp_rate"],
        "sensitivity": raw_t["sensitivity"],
        "thr": None,
    }]
    tf_rows = []
    for split_name, bag in [("VAL", val), ("TEST", test)]:
        m = pack(f"{split_name}_learned_tau", bag)
        rows.append({
            "name": m["name"],
            "overall_dice": m["overall_dice"],
            "nonempty_dice": m["nonempty_dice"],
            "empty_fp_rate": m["empty_fp_rate"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
            "thr": m["thr"],
            "tau_logit": m["tau_logit"],
            "presence_auroc": m["presence_auroc"],
        })
        tf_rows.append(m)
        safe_print(f"==== {split_name} TF τ={ts['tau_prob']:.3f} ====")
        safe_print(
            f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} "
            f"(pos={m['n_pos']} neg={m['n_neg']})"
        )
        safe_print(
            f"  sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"prec={m['precision']:.3f} F1={m['f1']:.3f} acc={m['accuracy']:.3f} "
            f"AUROC={m['presence_auroc']:.3f}"
        )
        safe_print(
            f"  SEG@gate ne={m['nonempty_dice']:.4f} empty_fp={m['empty_fp_rate']:.3f}"
        )

    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with (OUT / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    report = {
        "learned_tau": ts,
        "tf_primary": [
            {k: m[k] for k in (
                "name", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
                "precision", "f1", "accuracy", "presence_auroc",
                "nonempty_dice", "empty_fp_rate", "thr",
            )}
            for m in tf_rows
        ],
        "rows": rows,
        "ref_presence_tau020": {"nonempty_dice": 0.533, "empty_fp_rate": 0.269},
        "note": "finished via eval_bt_cascade_final after history.csv crash",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# BT cascade + learnable τ",
        "",
        "Primary view: presence T/F. SEG dice secondary.",
        f"learned τ_prob≈{ts['tau_prob']:.4f} (logit={ts['tau_logit']:.4f})",
        "",
        "## Presence T/F (gate)",
    ]
    for m in tf_rows:
        lines += [
            f"### {m['name']}",
            f"- TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}",
            f"- sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"prec={m['precision']:.3f} F1={m['f1']:.3f} AUROC={m['presence_auroc']:.3f} "
            f"acc={m['accuracy']:.3f}",
            f"- SEG@gate: ne={m['nonempty_dice']:.4f} empty_fp={m['empty_fp_rate']:.3f}",
            "",
        ]
    lines.append("## SEG / other")
    for r in rows:
        lines.append(
            f"- {r['name']}: ne={r['nonempty_dice']:.4f} fp={r['empty_fp_rate']:.3f}"
            + (f" τ={r['thr']:.3f}" if r.get("thr") is not None else "")
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    # mark cascade done for waiter
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write("\nTEST learned τ final report wrote bt_cascade\n")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
