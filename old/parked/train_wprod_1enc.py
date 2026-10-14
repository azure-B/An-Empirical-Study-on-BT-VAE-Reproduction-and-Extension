"""Compress messy wprod+mass into 1 S5 encoder + Head_μ × Head_ctx + min-mass.

- Freeze S5; train two heads with BCE (τ ∉ loss).
- Gate: s ∝ p_μ^1.5 · p_ctx^1.3, keep if s≥τ ∧ m≥m_min (VAL→TEST).
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
from models_wprod_1enc import Wprod1Enc
from train_lesion_presence import PIXEL_THR, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)
from train_s5_gate_25d import Stack25D, make_loader, presence_y, write_csv

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OUT = RESULT_DIR / "wprod_1enc"
WPROD_NE, WPROD_FP = 0.601, 0.305
ALPHA, BETA = 1.5, 1.3
BATCH = 8
EPOCHS = 40
LR = 1e-3
POS_WEIGHT = 2.0
ES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def collect(model: Wprod1Enc, loader):
    model.eval()
    bags = {
        "gt": [], "mass": [], "dice": [],
        "p_mu": [], "p_ctx": [],
    }
    for batch in loader:
        x3 = batch[0].to(DEVICE)
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        xp, xc, xn = x3[:, 0:1], x3[:, 1:2], x3[:, 2:3]
        seg_logits = model.seg_logits(xc)
        l_mu, l_ctx = model.presence_logits(xp, xc, xn)
        pred = (torch.sigmoid(seg_logits) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["p_mu"].append(torch.sigmoid(l_mu).cpu().numpy())
        bags["p_ctx"].append(torch.sigmoid(l_ctx).cpu().numpy())
    out = {k: np.concatenate(v) for k, v in bags.items()}
    s = (out["p_mu"] ** ALPHA) * (out["p_ctx"] ** BETA)
    out["score"] = s / (s.max() + 1e-8)
    return out


def eval_keep(bag, keep) -> dict:
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "fn": cm["TF"],
        "fp": cm["FT"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "auroc": binary_auroc(yt, bag["score"]),
    }


def pick_gate(val, test, ne_floor=0.60):
    masses = [0, 80, 160, 320, 640, 1280, 2560, 5120]
    thrs = [i / 100.0 for i in range(1, 80)]
    best = None
    for thr in thrs:
        for mmin in masses:
            keep = (val["score"] >= thr) & (val["mass"] >= mmin)
            mv = eval_keep(val, keep)
            if mv["nonempty_dice"] + 1e-9 < ne_floor:
                continue
            key = (mv["empty_fp_rate"], -mv["nonempty_dice"])
            if best is None or key < (best["empty_fp_rate"], -best["nonempty_dice"]):
                best = {**mv, "thr": thr, "min_mass": mmin}
    if best is None:
        # joint fallback
        for thr in thrs:
            for mmin in masses:
                keep = (val["score"] >= thr) & (val["mass"] >= mmin)
                mv = eval_keep(val, keep)
                j = mv["nonempty_dice"] - 0.5 * mv["empty_fp_rate"]
                if best is None or j > best.get("_j", -1e9):
                    best = {**mv, "thr": thr, "min_mass": mmin, "_j": j}
    keep_t = (test["score"] >= best["thr"]) & (test["mass"] >= best["min_mass"])
    mt = eval_keep(test, keep_t)
    mt.update({"thr": best["thr"], "min_mass": best["min_mass"]})
    return best, mt


def train_epoch(model, loader, opt, bce):
    model.head_mu.train()
    model.head_ctx.train()
    model.s5.eval()
    total, n = 0.0, 0
    for batch in loader:
        x3 = batch[0].to(DEVICE)
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        y = presence_y(seg)
        xp, xc, xn = x3[:, 0:1], x3[:, 1:2], x3[:, 2:3]
        with torch.no_grad():
            mu_p, _, _ = model.encode_mu(xp)
            mu_c, _, _ = model.encode_mu(xc)
            mu_n, _, _ = model.encode_mu(xn)
        l_mu = model.head_mu(mu_c)
        l_ctx = model.head_ctx(mu_p, mu_c, mu_n)
        loss = bce(l_mu, y) + bce(l_ctx, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def run_eval(model, va, te, out: Path):
    val, test = collect(model, va), collect(model, te)
    a_mu = binary_auroc(val["gt"] > 0, val["p_mu"])
    a_ctx = binary_auroc(val["gt"] > 0, val["p_ctx"])
    a_prod = binary_auroc(val["gt"] > 0, val["score"])
    safe_print(f"VAL AUROC μ={a_mu:.4f} ctx={a_ctx:.4f} wprod={a_prod:.4f}")

    best60, mt60 = pick_gate(val, test, 0.60)
    best57, mt57 = pick_gate(val, test, 0.57)
    # score-only soft
    base = eval_keep(test, test["score"] >= 0.01)

    rows = [
        {"name": "VAL_ne60", **{k: best60[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn")}},
        {"name": "TEST_ne60", **{k: mt60[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn")}},
        {"name": "TEST_ne57", **{k: mt57[k] for k in (
            "thr", "min_mass", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn")}},
        {"name": "TEST_wprod_only_t0.01", "thr": 0.01, "min_mass": 0,
         **{k: base[k] for k in ("nonempty_dice", "empty_fp_rate", "sensitivity", "fn")}},
        {"name": "messy_wprod_mass", "nonempty_dice": WPROD_NE, "empty_fp_rate": WPROD_FP},
    ]
    write_csv(out / "final_metrics.csv", rows)

    lines = [
        "# wprod 1-encoder (S5 μ × ctx μ) + min-mass",
        "",
        f"VAL AUROC: μ={a_mu:.4f} ctx={a_ctx:.4f} wprod={a_prod:.4f}",
        "",
        "| system | ne | empty FP | note |",
        "|---|---:|---:|---|",
        f"| messy wprod+mass | {WPROD_NE:.3f} | {WPROD_FP:.3f} | S5+dual+ResNet25d |",
        (
            f"| **1enc wprod+mass (ne≥0.60 pick)** | **{mt60['nonempty_dice']:.3f}** | "
            f"**{mt60['empty_fp_rate']:.3f}** | τ={mt60['thr']:.2f} m≥{mt60['min_mass']} |"
        ),
        (
            f"| 1enc wprod+mass (ne≥0.57 pick) | {mt57['nonempty_dice']:.3f} | "
            f"{mt57['empty_fp_rate']:.3f} | τ={mt57['thr']:.2f} m≥{mt57['min_mass']} |"
        ),
        f"| 1enc wprod-only τ0.01 | {base['nonempty_dice']:.3f} | {base['empty_fp_rate']:.3f} | no mass |",
        "",
        "Encoder count: **1** (frozen S5). Heads only for presence.",
    ]
    (out / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "val_auroc": {"mu": a_mu, "ctx": a_ctx, "wprod": a_prod},
        "test_ne60": mt60,
        "test_ne57": mt57,
        "test_wprod_only": base,
        "messy": {"ne": WPROD_NE, "fp": WPROD_FP},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print(
        f"TEST ne≥0.60 pick → ne={mt60['nonempty_dice']:.4f} fp={mt60['empty_fp_rate']:.3f} "
        f"(messy {WPROD_NE}/{WPROD_FP})"
    )
    return a_prod, mt60


def main():
    args = parse_args()
    out = OUT
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "heads_best_auroc.pt"

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False
    )
    tr = make_loader(Stack25D(data.train), args.batch, True)
    va = make_loader(Stack25D(data.val), args.batch, False)
    te = make_loader(Stack25D(data.test), args.batch, False)

    model = Wprod1Enc(base=BASE).to(DEVICE)
    sd = safe_load(S5_BEST, map_location="cpu")
    model.s5.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
    model.freeze_s5()

    if args.eval_only:
        if not best_path.exists():
            raise SystemExit(f"missing {best_path}")
        h = safe_load(best_path, map_location="cpu")
        model.head_mu.load_state_dict(h["head_mu"])
        model.head_ctx.load_state_dict(h["head_ctx"])
        run_eval(model, va, te, out)
        return

    pw = torch.tensor([POS_WEIGHT], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(
        list(model.head_mu.parameters()) + list(model.head_ctx.parameters()),
        lr=args.lr,
    )
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_auroc, bad = -1.0, 0
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr, opt, bce)
        val = collect(model, va)
        auroc = binary_auroc(val["gt"] > 0, val["score"])
        sched.step(auroc)
        hist.append({"ep": ep, "train_bce": tr_loss, "val_wprod_auroc": auroc})
        safe_print(
            f"[wprod1enc {ep}/{args.epochs}] bce={tr_loss:.4f} val_wprod_auroc={auroc:.4f}"
        )
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            bad = 0
            atomic_torch_save(
                {
                    "head_mu": model.head_mu.state_dict(),
                    "head_ctx": model.head_ctx.state_dict(),
                    "ep": ep,
                    "val_wprod_auroc": auroc,
                    "design": "wprod_1enc",
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= ES:
                safe_print(f"early stop @ {ep}")
                break

    h = safe_load(best_path, map_location="cpu")
    model.head_mu.load_state_dict(h["head_mu"])
    model.head_ctx.load_state_dict(h["head_ctx"])
    run_eval(model, va, te, out)
    write_csv(out / "train_hist.csv", hist)
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "best_val_wprod_auroc": best_auroc,
                "seconds": time.time() - t0,
                "alpha": ALPHA,
                "beta": BETA,
                "s5": str(S5_BEST),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    safe_print("wrote", out)


if __name__ == "__main__":
    main()
