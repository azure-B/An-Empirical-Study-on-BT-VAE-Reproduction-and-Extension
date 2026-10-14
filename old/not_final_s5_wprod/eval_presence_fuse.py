"""Step2: fuse image-cls P with SEG mass/max-prob → gate; report TF + gated ne.

Uses:
  - presence_imgcls best ckpt
  - bt_cascade best_tau (or best_seg) for SEG masks
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from models_bt_cascade import BTCascadeVAE
from models_presence_img import PresenceImgCls
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    safe_load,
    safe_print,
)

OUT = RESULT_DIR / "presence_fuse"
IMGCLS_BEST = RESULT_DIR / "presence_imgcls" / "checkpoints" / "imgcls_best_auroc.pt"
CASCADE_TAU = RESULT_DIR / "bt_cascade" / "checkpoints" / "bt_cascade_best_tau.pt"
CASCADE_SEG = RESULT_DIR / "bt_cascade" / "checkpoints" / "bt_cascade_best_seg.pt"


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


@torch.inference_mode()
def collect(imgcls, cascade, loader):
    imgcls.eval()
    cascade.eval()
    bags = {
        "gt": [], "mass": [], "dice": [],
        "p_img": [], "p_seg": [], "p_cas": [],
    }
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        p_img = torch.sigmoid(imgcls(flair))
        seg_l, _c, plog, _dec, *_ = cascade.forward_parts(flair)
        p_cas = torch.sigmoid(plog)
        prob = torch.sigmoid(seg_l)
        p_seg = prob.flatten(1).amax(dim=1)  # max lesion confidence
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        pred = (prob >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["p_img"].append(p_img.cpu().numpy())
        bags["p_seg"].append(p_seg.cpu().numpy())
        bags["p_cas"].append(p_cas.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def tf_and_seg(bag, score, thr):
    keep = score >= thr
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    tp, fp, fn, tn = cm["TT"], cm["FT"], cm["TF"], cm["FF"]
    prec, sens = cm["precision"], cm["sensitivity"]
    f1 = 2 * prec * sens / max(1e-8, prec + sens)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "thr": float(thr),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(cm["specificity"]),
        "precision": float(prec),
        "f1": float(f1),
        "auroc": binary_auroc(yt, score),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
    }


def pick_thr(bag, score, mode="acc"):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = tf_and_seg(bag, score, thr)
        key = {
            "acc": m["accuracy"],
            "f1": m["f1"],
            "bal": 0.5 * (m["sensitivity"] + m["specificity"]),
            "pareto": m["nonempty_dice"] - 0.5 * m["empty_fp_rate"],
        }[mode]
        if best is None or key > best["_key"]:
            best = dict(m)
            best["_key"] = key
            best["pick"] = mode
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not IMGCLS_BEST.exists():
        raise FileNotFoundError(IMGCLS_BEST)
    cas_path = CASCADE_TAU if CASCADE_TAU.exists() else CASCADE_SEG
    if not cas_path.exists():
        raise FileNotFoundError(cas_path)

    data = datasets.bp_train_30(batch_size=8, num_workers=NUM_WORKERS, pin_memory=False)
    imgcls = PresenceImgCls().to(DEVICE)
    imgcls.load_state_dict(safe_load(IMGCLS_BEST, map_location="cpu")["model"])
    cascade = BTCascadeVAE(base=BASE).to(DEVICE)
    cascade.load_state_dict(safe_load(cas_path, map_location="cpu")["model"])
    cascade.freeze_encoder()
    safe_print(f"fuse: imgcls={IMGCLS_BEST.name} cascade={cas_path.name}")

    val = collect(imgcls, cascade, data.val_loader)
    test = collect(imgcls, cascade, data.test_loader)

    # fuse recipes on VAL scores
    recipes = {
        "img_only": lambda b: b["p_img"],
        "seg_max": lambda b: b["p_seg"],
        "cascade_cls": lambda b: b["p_cas"],
        "mean_img_seg": lambda b: 0.5 * (b["p_img"] + b["p_seg"]),
        "max_img_seg": lambda b: np.maximum(b["p_img"], b["p_seg"]),
        "prod_img_seg": lambda b: b["p_img"] * b["p_seg"],
        "mean_img_cas": lambda b: 0.5 * (b["p_img"] + b["p_cas"]),
        "max_img_cas": lambda b: np.maximum(b["p_img"], b["p_cas"]),
    }

    rows = []
    lines = [
        "# Presence fuse (imgcls × SEG / cascade)",
        "",
        "Primary: T/F accuracy · Secondary: gated ne / empty FP",
        "",
    ]
    for name, fn in recipes.items():
        sv, st = fn(val), fn(test)
        for mode in ("acc", "f1", "pareto"):
            pv = pick_thr(val, sv, mode)
            mt = tf_and_seg(test, st, pv["thr"])
            row = {"recipe": name, "pick": mode, **{k: mt[k] for k in mt if k != "_key"}}
            rows.append(row)
            if mode == "acc":
                safe_print(
                    f"{name}/{mode} τ={mt['thr']:.2f} acc={mt['accuracy']:.3f} "
                    f"sens={mt['sensitivity']:.3f} spec={mt['specificity']:.3f} "
                    f"ne={mt['nonempty_dice']:.3f} fp={mt['empty_fp_rate']:.3f}"
                )
                lines.append(
                    f"- **{name}** @acc τ={mt['thr']:.2f}: acc={mt['accuracy']:.3f} "
                    f"sens={mt['sensitivity']:.3f} spec={mt['specificity']:.3f} "
                    f"F1={mt['f1']:.3f} | ne={mt['nonempty_dice']:.3f} "
                    f"empty_fp={mt['empty_fp_rate']:.3f} "
                    f"TP/TN/FP/FN={mt['tp']}/{mt['tn']}/{mt['fp']}/{mt['fn']}"
                )

    with (OUT / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps({"rows": rows, "imgcls": str(IMGCLS_BEST), "cascade": str(cas_path)}, indent=2),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
