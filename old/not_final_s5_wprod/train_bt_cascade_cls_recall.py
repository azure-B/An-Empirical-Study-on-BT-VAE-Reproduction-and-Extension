"""Upgrade cascade cls only: freeze SEG, train Head(μ)+Head(skips), calibrate τ for recall.

Training change vs old cascade:
  - presence score trained WITHOUT τ in the loss (τ not competing with logit)
  - asymmetric BCE (high pos_weight) → fewer FN (lesion wiped)
  - τ picked on VAL with sensitivity floor (default 0.90), then min empty-FP

SEG = cascade_focal best_seg (frozen).
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
from models_bt_cascade_cls import BTCascadeClsUpgrade, load_from_cascade_focal
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)

FOCAL_SEG = RESULT_DIR / "bt_cascade_focal" / "checkpoints" / "bt_cascade_best_seg.pt"
OUT = RESULT_DIR / "bt_cascade_cls_recall"
BATCH = 8
EPOCHS = 50
LR = 1e-3
POS_WEIGHT = 3.0  # recall-first: punish missing lesions harder
ES = 12
SENS_FLOORS = (0.95, 0.90, 0.85, 0.80)


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, presence, *_ = model.forward_parts(flair)
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(torch.sigmoid(presence).cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def eval_at_thr(bag, thr: float) -> dict:
    keep = bag["score"] >= thr
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    tp, fp, fn, tn = cm["TT"], cm["FT"], cm["TF"], cm["FF"]
    prec, sens, spec = cm["precision"], cm["sensitivity"], cm["specificity"]
    f1 = 2 * prec * sens / max(1e-8, prec + sens)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "thr": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auroc": binary_auroc(yt, bag["score"]),
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "seg_sensitivity": float(m["sensitivity"]),
    }


def pick_tau_sens_floor(val_bag, floor: float):
    """Among τ with VAL sens ≥ floor, pick lowest empty_fp; tie → higher ne."""
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_at_thr(val_bag, thr)
        if m["sensitivity"] + 1e-9 < floor:
            continue
        key = (-m["empty_fp_rate"], m["nonempty_dice"], m["f1"])
        if best is None or key > best["_key"]:
            best = dict(m)
            best["_key"] = key
            best["sens_floor"] = floor
    return best


def pick_tau_f1(val_bag):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_at_thr(val_bag, thr)
        if best is None or m["f1"] > best["f1"]:
            best = dict(m)
            best["sens_floor"] = "f1"
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pos-weight", type=float, default=POS_WEIGHT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seg-ckpt", type=str, default=str(FOCAL_SEG))
    args = ap.parse_args()

    seg_ckpt = Path(args.seg_ckpt)
    if not seg_ckpt.exists():
        raise FileNotFoundError(seg_ckpt)

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False,
    )
    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} "
        f"pos_w={args.pos_weight} seg={seg_ckpt.name}"
    )
    safe_print(
        "design: freeze SEG(focal) | train Head(μ)+Head(skips) NO-τ-in-loss | "
        "τ by VAL sens-floor"
    )

    model = BTCascadeClsUpgrade(base=BASE).to(DEVICE)
    nload = load_from_cascade_focal(model, safe_load(seg_ckpt, map_location="cpu"))
    model.freeze_encoder()
    model.freeze_seg()
    model.set_trainable(cls_heads=True, decoder_cls=False)
    safe_print(f"loaded encoder+seg from focal n={nload}; heads random; τ separate")

    # --- Phase1: train presence score only (no τ) ---
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=DEVICE)
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    best_path = ckpt_dir / "cls_best_auroc.pt"
    last_path = ckpt_dir / "cls_last.pt"
    history = []
    best_auroc = -1.0
    es = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        model.freeze_encoder()
        model.freeze_seg()
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            y = presence_y(seg)
            opt.zero_grad(set_to_none=True)
            _seg, presence, *_ = model.forward_parts(flair)
            loss = bce(presence, y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1

        val = collect(model, data.val_loader)
        auroc = binary_auroc(val["gt"] > 0, val["score"])
        m05 = eval_at_thr(val, 0.5)
        rec = {
            "epoch": ep,
            "train_bce": tr / max(1, ntr),
            "val_auroc": auroc,
            "val_acc@0.5": m05["accuracy"],
            "val_sens@0.5": m05["sensitivity"],
            "val_spec@0.5": m05["specificity"],
            "val_fn@0.5": m05["fn"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[cls {ep}/{args.epochs}] bce={rec['train_bce']:.4f} "
            f"auroc={auroc:.4f} acc={m05['accuracy']:.3f} "
            f"sens={m05['sensitivity']:.3f} spec={m05['specificity']:.3f} "
            f"FN={m05['fn']} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": model.state_dict(),
            "epoch": ep,
            "val_auroc": auroc,
            "design": "cascade_cls_recall_sep_tau",
            "pos_weight": args.pos_weight,
            "seg_ckpt": str(seg_ckpt),
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es = 0
            safe_print(f"  * best auroc={auroc:.4f} sens@0.5={m05['sensitivity']:.3f}")
        else:
            es += 1
            if es >= ES:
                safe_print(f"early stop @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # --- Phase2: calibrate τ (separate from training) ---
    st = safe_load(best_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_encoder()
    model.freeze_seg()
    model.eval()

    val = collect(model, data.val_loader)
    test = collect(model, data.test_loader)
    ones = np.ones(len(test["gt"]), dtype=bool)
    raw = metrics_bundle(test["gt"], test["mass"], test["dice"], ones)

    rows = [{
        "name": "TEST_seg_raw",
        "nonempty_dice": raw["nonempty_dice"],
        "empty_fp_rate": raw["empty_fp_rate"],
        "sensitivity": raw["sensitivity"],
        "thr": None,
    }]
    lines = [
        "# Cascade cls-recall (separate τ)",
        "",
        "SEG frozen from cascade_focal · Head(μ)+Head(skips) · τ NOT in train loss",
        f"best VAL AUROC={best_auroc:.4f} · pos_weight={args.pos_weight}",
        f"TEST raw SEG: ne={raw['nonempty_dice']:.4f} fp={raw['empty_fp_rate']:.3f}",
        "",
        "## τ picks (VAL calibrate → TEST)",
    ]

    picks = {}
    for floor in SENS_FLOORS:
        pv = pick_tau_sens_floor(val, floor)
        if pv is None:
            safe_print(f"no τ with VAL sens≥{floor:.2f}")
            continue
        mt = eval_at_thr(test, pv["thr"])
        picks[f"sens{floor:.2f}"] = {"val": pv, "test": mt}
        rows.append({
            "name": f"TEST_sensFloor_{floor:.2f}",
            **{k: mt[k] for k in (
                "thr", "accuracy", "sensitivity", "specificity", "precision",
                "f1", "auroc", "tp", "tn", "fp", "fn",
                "nonempty_dice", "empty_fp_rate", "seg_sensitivity",
            )},
            "val_sensitivity": pv["sensitivity"],
            "val_empty_fp": pv["empty_fp_rate"],
            "val_ne": pv["nonempty_dice"],
        })
        safe_print(
            f"==== sens≥{floor:.2f} → τ={pv['thr']:.2f} ===="
        )
        safe_print(
            f"  VAL sens={pv['sensitivity']:.3f} fp={pv['empty_fp_rate']:.3f} "
            f"ne={pv['nonempty_dice']:.3f}"
        )
        safe_print(
            f"  TEST TP/TN/FP/FN={mt['tp']}/{mt['tn']}/{mt['fp']}/{mt['fn']} "
            f"acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
            f"spec={mt['specificity']:.3f}"
        )
        safe_print(
            f"  TEST gated ne={mt['nonempty_dice']:.4f} empty_fp={mt['empty_fp_rate']:.3f}"
        )
        lines += [
            f"### sens-floor {floor:.2f} → τ={pv['thr']:.2f}",
            f"- VAL: sens={pv['sensitivity']:.3f} empty_fp={pv['empty_fp_rate']:.3f} "
            f"ne={pv['nonempty_dice']:.3f}",
            f"- TEST TF: TP={mt['tp']} TN={mt['tn']} FP={mt['fp']} FN={mt['fn']} "
            f"acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
            f"spec={mt['specificity']:.3f} F1={mt['f1']:.3f}",
            f"- TEST SEG@gate: ne={mt['nonempty_dice']:.4f} "
            f"empty_fp={mt['empty_fp_rate']:.3f}",
            "",
        ]

    pf = pick_tau_f1(val)
    mt = eval_at_thr(test, pf["thr"])
    rows.append({
        "name": "TEST_f1",
        **{k: mt[k] for k in (
            "thr", "accuracy", "sensitivity", "specificity", "precision",
            "f1", "auroc", "tp", "tn", "fp", "fn",
            "nonempty_dice", "empty_fp_rate", "seg_sensitivity",
        )},
    })
    safe_print(
        f"==== f1 → τ={pf['thr']:.2f} TEST ne={mt['nonempty_dice']:.4f} "
        f"fp={mt['empty_fp_rate']:.3f} sens={mt['sensitivity']:.3f}"
    )
    lines += [
        f"### F1 → τ={pf['thr']:.2f}",
        f"- TEST: acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
        f"spec={mt['specificity']:.3f} ne={mt['nonempty_dice']:.4f} "
        f"empty_fp={mt['empty_fp_rate']:.3f}",
        "",
        "## Refs",
        "- cascade_focal gated: ne=0.521 fp=0.281 acc=0.695 sens=0.683",
        "- presence τ=0.20: ne=0.533 fp=0.269",
        "- dual-role @0.22: ne=0.600 fp=0.383",
    ]

    # save primary τ = first available sens 0.90 else 0.85
    primary = picks.get("sens0.90") or picks.get("sens0.85") or picks.get("sens0.80")
    if primary:
        thr = float(primary["val"]["thr"])
        model.tau_prob.fill_(thr)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "tau_prob": thr,
                "val_auroc": best_auroc,
                "pick": primary["val"].get("sens_floor"),
                "design": "cascade_cls_recall_sep_tau",
                "pos_weight": args.pos_weight,
                "seg_ckpt": str(seg_ckpt),
            },
            ckpt_dir / "cls_best_tau.pt",
        )

    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "best_val_auroc": best_auroc,
                "pos_weight": args.pos_weight,
                "seg_ckpt": str(seg_ckpt),
                "picks": {
                    k: {"val_thr": v["val"]["thr"], "test": v["test"]}
                    for k, v in picks.items()
                },
                "rows": rows,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# Cascade cls-recall",
            "",
            "```",
            "x → BT Enc (freeze) → μ, skips",
            "      ├─ Head(μ) + Head(skips) → P   # train: asymmetric BCE, NO τ",
            "      └─ Decoder_seg (freeze, from cascade_focal) → mask",
            "",
            "τ calibrated on VAL: max empty-FP↓ among sens ≥ floor",
            "infer: keep SEG iff P ≥ τ",
            "```",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
