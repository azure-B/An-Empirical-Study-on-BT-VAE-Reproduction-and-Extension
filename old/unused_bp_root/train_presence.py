"""Train a slice-presence head on frozen none VAE. original/ is not modified.

none SEG weights stay as-is. Only the head learns empty vs nonempty.
At test, slices with P(present) < tau are forced to empty.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_presence import PresenceVAE
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RESULT_DIR,
    SEG_BATCH,
    atomic_torch_save,
    evaluate_test,
    safe_load,
    safe_print,
    split_dice_stats,
    batch_dice,
    batch_pred_gt_sums,
)

ROOT = Path(__file__).resolve().parent
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT_DIR = ROOT / "results" / "bp_original" / "none_presence"
PRESENCE_EPOCHS = 60
PRESENCE_LR = 1e-3
ES_PATIENCE = 12
POS_WEIGHT = 1.5
TAU_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def eval_gated(model: PresenceVAE, loader, tau: float):
    model.eval()
    model.gate_thr = tau
    dices, gt_sums, pred_sums = [], [], []
    y_true, y_score = [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        out, mu, _ = model.vae(flair)
        logit = model.presence_logit(mu)
        present = torch.sigmoid(logit) >= tau
        out = out.masked_fill(~present.view(out.size(0), 1, 1, 1), -20.0)
        dices.append(batch_dice(out, seg))
        pred_s, gt_s = batch_pred_gt_sums(out, seg)
        pred_sums.append(pred_s)
        gt_sums.append(gt_s)
        y_true.append((gt_s > 0).astype(np.float32))
        y_score.append(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1))
    model.gate_thr = None
    stats = split_dice_stats(
        np.concatenate(dices), np.concatenate(gt_sums), np.concatenate(pred_sums)
    )
    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    pred_pos = ys >= tau
    stats["presence_acc"] = float((pred_pos == (yt > 0.5)).mean())
    stats["presence_recall"] = float(((pred_pos) & (yt > 0.5)).sum() / max(1.0, (yt > 0.5).sum()))
    stats["presence_auroc"] = binary_auroc(yt > 0.5, ys)
    stats["tau"] = tau
    return stats


def pick_tau(model: PresenceVAE, val_loader, nonempty_floor: float):
    rows = []
    best = None
    for tau in TAU_GRID:
        s = eval_gated(model, val_loader, tau)
        s["ok_nonempty"] = (
            s["mean_dice_nonempty"] is not None
            and s["mean_dice_nonempty"] >= nonempty_floor
        )
        rows.append(s)
        safe_print(
            f"  tau={tau:.2f} dice={s['dice_mean']:.4f} nonempty={s['mean_dice_nonempty']:.4f} "
            f"empty_fp={s['empty_fp_rate']:.3f} pres_rec={s['presence_recall']:.3f}"
        )
        if not s["ok_nonempty"]:
            continue
        key = (s["dice_mean"], -s["empty_fp_rate"])
        if best is None or key > (best["dice_mean"], -best["empty_fp_rate"]):
            best = s
    if best is None:
        best = max(rows, key=lambda r: r["dice_mean"])
        safe_print("no tau kept nonempty floor; fallback to max overall dice")
    return best, rows


def main():
    if not NONE_BEST.exists():
        raise FileNotFoundError(f"none best missing: {NONE_BEST}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    safe_print(
        "DEVICE:", DEVICE, "presence on frozen none:",
        len(data.train), len(data.val), len(data.test), f"batch={SEG_BATCH}",
    )

    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    none_best_ep = int(st.get("best_ep", -1))
    del st
    for p in vae.parameters():
        p.requires_grad = False
    vae.to(DEVICE).eval()

    model = PresenceVAE(vae, base=BASE).to(DEVICE)
    opt = torch.optim.AdamW(model.head.parameters(), lr=PRESENCE_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-6)  # AUROC
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

    ungated_val = evaluate_test(vae, data.val_loader)
    floor = float(ungated_val["mean_dice_nonempty"]) - 0.02
    safe_print(
        "ungated VAL:",
        f"dice={ungated_val['dice_mean']:.4f}",
        f"nonempty={ungated_val['mean_dice_nonempty']:.4f}",
        f"empty_fp={ungated_val['empty_fp_rate']:.3f}",
        f"tau_floor_nonempty={floor:.4f}",
    )

    best_score = -1.0
    best_ep = 0
    history = []
    for ep in range(1, PRESENCE_EPOCHS + 1):
        model.head.train()
        model.vae.eval()
        tr_loss = tr_acc = 0.0
        ntr = 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            with torch.no_grad():
                _, mu, _ = model.vae(flair)
            y = (seg.flatten(1).sum(dim=1) > 0).float().unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logit = model.presence_logit(mu)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            pred = (torch.sigmoid(logit) >= 0.5).float()
            tr_acc += float((pred == y).float().mean().item())
            ntr += 1
        ntr = max(1, ntr)
        val_s = eval_gated(model, data.val_loader, 0.5)
        sched.step(val_s["presence_auroc"])
        safe_print(
            f"[presence] ep {ep}/{PRESENCE_EPOCHS} loss={tr_loss/ntr:.4f} acc={tr_acc/ntr:.3f} "
            f"auroc={val_s['presence_auroc']:.3f} val_dice={val_s['dice_mean']:.4f} "
            f"nonempty={val_s['mean_dice_nonempty']:.4f} empty_fp={val_s['empty_fp_rate']:.3f} "
            f"pres_rec={val_s['presence_recall']:.3f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr_loss / ntr, "val": val_s})
        payload = {
            "ep": ep,
            "head": model.head.state_dict(),
            "best_score": best_score,
            "best_ep": best_ep,
            "none_ckpt": str(NONE_BEST),
            "none_best_ep": none_best_ep,
        }
        if val_s["presence_auroc"] > best_score:
            best_score = float(val_s["presence_auroc"])
            best_ep = ep
            payload["best_score"] = best_score
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, OUT_DIR / "head_best.pt")
            safe_print(f"  saved HEAD_BEST auroc={best_score:.4f}")
        atomic_torch_save(payload, OUT_DIR / "head_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_PATIENCE:
            safe_print(f"early stop: no val AUROC best for {ES_PATIENCE} epochs (best={best_score:.4f} @{best_ep})")
            break

    st = safe_load(OUT_DIR / "head_best.pt", map_location="cpu")
    model.head.load_state_dict(st["head"])
    model.to(DEVICE).eval()
    safe_print("tau sweep on VAL (keep nonempty >= none-0.02)")
    best_tau_row, tau_rows = pick_tau(model, data.val_loader, floor)
    tau = float(best_tau_row["tau"])
    safe_print(f"chosen tau={tau:.2f}")

    val_g = eval_gated(model, data.val_loader, tau)
    test_g = eval_gated(model, data.test_loader, tau)
    test_ungated = evaluate_test(vae, data.test_loader)
    report = {
        "mode": "none_presence",
        "none_ckpt": str(NONE_BEST),
        "none_best_ep": none_best_ep,
        "head_best_ep": int(st.get("best_ep", best_ep)),
        "tau": tau,
        "pos_weight": POS_WEIGHT,
        "ungated_val": ungated_val,
        "ungated_test": test_ungated,
        "gated_val": val_g,
        "gated_test": test_g,
        "tau_sweep_val": tau_rows,
        "protocol": {
            "vae": "frozen none seg_best",
            "train": "presence head only",
            "selection": "val gated dice @0.5 then tau sweep",
            "nonempty_floor": floor,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST ungated:", json.dumps(test_ungated, indent=2))
    safe_print("TEST gated:", json.dumps(test_g, indent=2))


if __name__ == "__main__":
    main()
