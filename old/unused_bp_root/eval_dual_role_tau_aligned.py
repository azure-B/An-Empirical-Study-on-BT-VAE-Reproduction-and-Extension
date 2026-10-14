"""Re-judge dual-role: pick τ on classification line, apply to S5 inference.

No retrain. Loads dual_role_best_auroc for none / emptyfp.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from train_dual_role import (
    BATCH,
    S5_BEST,
    binary_auroc,
    collect_split,
    eval_gate,
)
from train_lesion_presence import metrics_bundle
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print

OUT = RESULT_DIR / "dual_role_tau_aligned"
CKPTS = {
    "none": RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt",
    "emptyfp": RESULT_DIR / "dual_role_train_emptyfp" / "checkpoints" / "dual_role_best_auroc.pt",
}


def presence_at_thr(bag, thr: float) -> dict:
    yt = bag["gt"] > 0
    yp = bag["cls_prob"] >= thr
    tp = int((yp & yt).sum())
    fp = int((yp & ~yt).sum())
    fn = int((~yp & yt).sum())
    tn = int((~yp & ~yt).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    spec = tn / max(1, tn + fp)
    return {
        "thr": thr,
        "presence_acc": float((yp == yt).mean()),
        "presence_prec": float(prec),
        "presence_recall": float(rec),
        "presence_f1": float(f1),
        "presence_spec": float(spec),
        "youden": float(rec + spec - 1.0),
        "TT": tp, "FT": fp, "TF": fn, "FF": tn,
    }


def pick_cls_tau(val: dict) -> dict:
    """τ chosen on classification scores only (not S5 dice)."""
    best_f1 = None
    best_youden = None
    best_acc = None
    curve = []
    for i in range(1, 100):
        thr = i / 100.0
        c = presence_at_thr(val, thr)
        curve.append(c)
        if best_f1 is None or c["presence_f1"] > best_f1["presence_f1"]:
            best_f1 = dict(c)
        if best_youden is None or c["youden"] > best_youden["youden"]:
            best_youden = dict(c)
        if best_acc is None or c["presence_acc"] > best_acc["presence_acc"]:
            best_acc = dict(c)
    return {
        "f1": best_f1,
        "youden": best_youden,
        "acc": best_acc,
        "curve": curve,
    }


def flat_infer(m: dict, name: str) -> dict:
    return {
        "name": name,
        "thr": m.get("thr"),
        "overall_dice": m["overall_dice"],
        "nonempty_dice": m["nonempty_dice"],
        "empty_fp_rate": m["empty_fp_rate"],
        "sensitivity": m["sensitivity"],
        "TT": m["TT"], "FT": m["FT"], "TF": m["TF"], "FF": m["FF"],
        "presence_acc": m.get("presence_acc"),
        "presence_recall": m.get("presence_recall"),
        "presence_auroc": m.get("presence_auroc"),
    }


def eval_run(tag: str, ckpt: Path, val_loader, test_loader):
    safe_print(f"\n======== {tag} ======== {ckpt}")
    model = DualRoleVAE(base=BASE).to(DEVICE)
    st = safe_load(ckpt, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_skip()
    model.eval()

    val = collect_split(model, val_loader)
    test = collect_split(model, test_loader)
    auroc_v = binary_auroc(val["gt"] > 0, val["cls_prob"])
    auroc_t = binary_auroc(test["gt"] > 0, test["cls_prob"])

    ones_v = np.ones(len(val["gt"]), dtype=bool)
    ones_t = np.ones(len(test["gt"]), dtype=bool)
    s5_v = metrics_bundle(val["gt"], val["mass_s5"], val["dice_s5"], ones_v)
    s5_t = metrics_bundle(test["gt"], test["mass_s5"], test["dice_s5"], ones_t)
    pl_v = metrics_bundle(val["gt"], val["mass_plain"], val["dice_plain"], ones_v)
    pl_t = metrics_bundle(test["gt"], test["mass_plain"], test["dice_plain"], ones_t)

    picks = pick_cls_tau(val)
    rows = [
        flat_infer({**s5_t, "thr": None, "presence_acc": None, "presence_recall": None,
                    "presence_auroc": auroc_t}, f"{tag}|S5_raw"),
        flat_infer({**pl_t, "thr": None, "presence_acc": None, "presence_recall": None,
                    "presence_auroc": auroc_t}, f"{tag}|cls_decoder_raw"),
    ]

    applied = {}
    for key in ("f1", "youden", "acc"):
        thr = float(picks[key]["thr"])
        # classification metrics at thr (VAL pick → TEST cls)
        cls_t = presence_at_thr(test, thr)
        # inference = S5 gated by same thr
        g_v = eval_gate(val, thr)
        g_t = eval_gate(test, thr)
        name = f"{tag}|infer@cls_{key}_τ{thr:.2f}"
        rows.append(flat_infer(g_t, name))
        applied[key] = {
            "thr": thr,
            "val_cls": picks[key],
            "test_cls": cls_t,
            "val_infer": flat_infer(g_v, "val"),
            "test_infer": flat_infer(g_t, "test"),
        }
        safe_print(
            f"  cls-{key} τ={thr:.2f} | TEST cls f1={cls_t['presence_f1']:.3f} "
            f"rec={cls_t['presence_recall']:.3f} | "
            f"infer ne={g_t['nonempty_dice']:.4f} fp={g_t['empty_fp_rate']:.3f} "
            f"sens={g_t['sensitivity']:.3f}"
        )

    return {
        "tag": tag,
        "auroc_val": auroc_v,
        "auroc_test": auroc_t,
        "s5_test": flat_infer({**s5_t, "thr": None}, "s5"),
        "cls_decoder_test": flat_infer({**pl_t, "thr": None}, "cls"),
        "picks": applied,
        "rows": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in CKPTS.values():
        if not p.exists():
            raise FileNotFoundError(p)
    if not S5_BEST.exists():
        raise FileNotFoundError(S5_BEST)

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    all_rows = []
    reports = []
    for tag, ckpt in CKPTS.items():
        r = eval_run(tag, ckpt, data.val_loader, data.test_loader)
        reports.append(r)
        all_rows.extend(r["rows"])

    with (OUT / "test_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    (OUT / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    lines = [
        "# Dual-role: τ from classification → S5 inference",
        "",
        "τ chosen on VAL **classification** (F1 / Youden / Acc). Inference = keep S5 iff P≥τ.",
        "",
        "## TEST",
        "",
        "| run | τ rule | τ | ne | empty FP | sens | cls F1 |",
        "|-----|--------|--:|---:|---------:|-----:|-------:|",
    ]
    for r in reports:
        s5 = r["s5_test"]
        lines.append(
            f"| {r['tag']} | S5 raw | — | {s5['nonempty_dice']:.3f} | {s5['empty_fp_rate']:.3f} | "
            f"{s5['sensitivity']:.3f} | — |"
        )
        for key in ("f1", "youden", "acc"):
            p = r["picks"][key]
            ti = p["test_infer"]
            tc = p["test_cls"]
            lines.append(
                f"| {r['tag']} | cls-{key} | {p['thr']:.2f} | {ti['nonempty_dice']:.3f} | "
                f"{ti['empty_fp_rate']:.3f} | {ti['sensitivity']:.3f} | {tc['presence_f1']:.3f} |"
            )
        lines.append("")
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
