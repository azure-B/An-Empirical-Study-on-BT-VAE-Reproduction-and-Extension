"""Step1: FLAIR image classifier for lesion presence (target ~90% acc).

No VAE μ. Train ResNet18 on bp_train_30. Report TF first.
After this: fuse with cascade SEG (step2).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_presence_img import PresenceImgCls
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix
from train_original_bp import (
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)

OUT = RESULT_DIR / "presence_imgcls"
BATCH = 16
EPOCHS = 60
LR = 1e-4
POS_WEIGHT = 1.5
ES = 12


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


def tf_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score >= thr
    cm = conf_matrix(y, pred)
    tp, fp, fn, tn = cm["TT"], cm["FT"], cm["TF"], cm["FF"]
    prec = cm["precision"]
    sens = cm["sensitivity"]
    f1 = 2 * prec * sens / max(1e-8, prec + sens)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        **cm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": float(f1),
        "accuracy": float(acc),
        "auroc": binary_auroc(y, score),
        "thr": float(thr),
    }


def pick_thr_val(y: np.ndarray, score: np.ndarray, mode: str = "acc") -> dict:
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = tf_metrics(y, score, thr)
        if mode == "acc":
            key = m["accuracy"]
        elif mode == "f1":
            key = m["f1"]
        else:  # bal = 0.5*(sens+spec)
            key = m["balanced_accuracy"]
        if best is None or key > best["_key"]:
            best = dict(m)
            best["_key"] = key
            best["pick"] = mode
    return best


@torch.inference_mode()
def collect_scores(model, loader):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        logit = model(flair)
        ys.append(presence_y(seg).cpu().numpy())
        ss.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False,
    )
    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} "
        f"n={len(data.train)}/{len(data.val)}/{len(data.test)}"
    )
    safe_print("design: ResNet18(FLAIR 1ch) → presence logit · no VAE μ")

    model = PresenceImgCls().to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    best_path = ckpt_dir / "imgcls_best_auroc.pt"
    last_path = ckpt_dir / "imgcls_last.pt"
    history = []
    best_auroc = -1.0
    es = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            y = presence_y(seg)
            opt.zero_grad(set_to_none=True)
            logit = model(flair)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        yv, sv = collect_scores(model, data.val_loader)
        auroc = binary_auroc(yv, sv)
        m05 = tf_metrics(yv, sv, 0.5)
        rec = {
            "epoch": ep,
            "train_bce": tr / max(1, ntr),
            "val_auroc": auroc,
            "val_acc@0.5": m05["accuracy"],
            "val_sens@0.5": m05["sensitivity"],
            "val_spec@0.5": m05["specificity"],
            "val_f1@0.5": m05["f1"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[imgcls {ep}/{args.epochs}] bce={rec['train_bce']:.4f} "
            f"auroc={auroc:.4f} acc={m05['accuracy']:.3f} "
            f"sens={m05['sensitivity']:.3f} spec={m05['specificity']:.3f} "
            f"F1={m05['f1']:.3f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(), "epoch": ep,
            "val_auroc": auroc, "design": "presence_imgcls_resnet18",
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es = 0
            safe_print(f"  * best auroc={auroc:.4f} acc@0.5={m05['accuracy']:.3f}")
        else:
            es += 1
            if es >= ES:
                safe_print(f"early stop @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.eval()

    yv, sv = collect_scores(model, data.val_loader)
    yt, stt = collect_scores(model, data.test_loader)
    picks = {}
    rows = []
    for mode in ("acc", "f1", "bal"):
        pv = pick_thr_val(yv, sv, mode)
        picks[mode] = pv
        mt = tf_metrics(yt, stt, pv["thr"])
        rows.append({
            "name": f"TEST_{mode}",
            "thr": pv["thr"],
            "accuracy": mt["accuracy"],
            "sensitivity": mt["sensitivity"],
            "specificity": mt["specificity"],
            "precision": mt["precision"],
            "f1": mt["f1"],
            "auroc": mt["auroc"],
            "tp": mt["tp"], "tn": mt["tn"], "fp": mt["fp"], "fn": mt["fn"],
            "val_accuracy": pv["accuracy"],
        })
        safe_print(
            f"==== TEST pick={mode} τ={pv['thr']:.2f} ===="
        )
        safe_print(
            f"  TP={mt['tp']} TN={mt['tn']} FP={mt['fp']} FN={mt['fn']}"
        )
        safe_print(
            f"  acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
            f"spec={mt['specificity']:.3f} F1={mt['f1']:.3f} AUROC={mt['auroc']:.3f}"
        )

    # also @0.5
    m50 = tf_metrics(yt, stt, 0.5)
    rows.insert(0, {
        "name": "TEST_thr0.50",
        "thr": 0.5,
        "accuracy": m50["accuracy"],
        "sensitivity": m50["sensitivity"],
        "specificity": m50["specificity"],
        "precision": m50["precision"],
        "f1": m50["f1"],
        "auroc": m50["auroc"],
        "tp": m50["tp"], "tn": m50["tn"], "fp": m50["fp"], "fn": m50["fn"],
    })
    safe_print(
        f"==== TEST τ=0.50 ==== acc={m50['accuracy']:.3f} "
        f"sens={m50['sensitivity']:.3f} spec={m50['specificity']:.3f}"
    )

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": "ResNet18 FLAIR→presence (no VAE μ)",
        "best_val_auroc": best_auroc,
        "target_accuracy": 0.90,
        "rows": rows,
        "ref_cascade_cls_auroc": 0.730,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
