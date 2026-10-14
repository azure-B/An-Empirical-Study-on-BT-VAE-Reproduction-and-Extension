"""Frozen S5 + bottleneck skip on/off gate.

S5 encoder/decoder stay fixed. Only a small head on μ (8×8 bottleneck)
learns empty vs nonempty. High pos_weight so lesion slices are almost
never turned off. At eval, P(present) < τ zeros the S5 mask.

original/ and s5_concat/ are not overwritten.
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
from models_bp_skip import VAE as SkipVAE
from models_presence import PresenceVAE
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    evaluate_test,
    safe_load,
    safe_print,
    split_dice_stats,
)
from train_presence import eval_gated

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OUT_DIR = RESULT_DIR / "s5_skipgate"
BATCH = 8
EPOCHS = 60
LR = 1e-3
ES_PATIENCE = 12
POS_WEIGHT = 8.0
TAU_GRID = tuple(round(x, 2) for x in np.linspace(0.05, 0.90, 18))


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def pick_tau_min_fp(model, val_loader, nonempty_floor: float):
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
            f"empty_fp={s['empty_fp_rate']:.3f} pres_rec={s['presence_recall']:.3f} "
            f"{'OK' if s['ok_nonempty'] else 'drop'}"
        )
        if not s["ok_nonempty"]:
            continue
        if best is None or s["empty_fp_rate"] < best["empty_fp_rate"]:
            best = s
    if best is None:
        best = max(rows, key=lambda r: r["mean_dice_nonempty"] or -1)
        safe_print("no tau kept nonempty floor; fallback to max nonempty")
    return best, rows


def main():
    if not S5_BEST.exists():
        raise FileNotFoundError(S5_BEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    safe_print("DEVICE", DEVICE, "S5 skip-gate frozen S5, batch", BATCH, "pos_w", POS_WEIGHT)

    vae = SkipVAE(base=BASE)
    st = safe_load(S5_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    s5_ep = int(st.get("best_ep", -1))
    del st
    for p in vae.parameters():
        p.requires_grad = False
    vae.to(DEVICE).eval()

    model = PresenceVAE(vae, base=BASE).to(DEVICE)
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

    ungated_val = evaluate_test(vae, data.val_loader)
    floor = float(ungated_val["mean_dice_nonempty"]) - 0.02
    safe_print(
        "ungated VAL",
        f"dice={ungated_val['dice_mean']:.4f} nonempty={ungated_val['mean_dice_nonempty']:.4f} "
        f"empty_fp={ungated_val['empty_fp_rate']:.3f} floor={floor:.4f}",
    )

    best_score = -1.0
    best_ep = 0
    history = []
    for ep in range(1, EPOCHS + 1):
        model.head.train()
        model.vae.eval()
        tr_loss = tr_acc = tr_rec = 0.0
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
            rec_den = float(y.sum().item())
            tr_rec += float(((pred * y).sum() / rec_den).item()) if rec_den > 0 else 1.0
            ntr += 1
        ntr = max(1, ntr)
        val_s = eval_gated(model, data.val_loader, 0.5)
        sched.step(val_s["presence_auroc"])
        safe_print(
            f"[s5-gate] ep {ep}/{EPOCHS} loss={tr_loss/ntr:.4f} acc={tr_acc/ntr:.3f} "
            f"tr_rec={tr_rec/ntr:.3f} auroc={val_s['presence_auroc']:.3f} "
            f"val_dice={val_s['dice_mean']:.4f} nonempty={val_s['mean_dice_nonempty']:.4f} "
            f"empty_fp={val_s['empty_fp_rate']:.3f} pres_rec={val_s['presence_recall']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr_loss / ntr, "val": val_s})
        payload = {
            "ep": ep,
            "head": model.head.state_dict(),
            "best_score": best_score,
            "best_ep": best_ep,
            "s5_ckpt": str(S5_BEST),
            "s5_ep": s5_ep,
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
            safe_print(f"early stop @{best_ep} auroc={best_score:.4f}")
            break

    st = safe_load(OUT_DIR / "head_best.pt", map_location="cpu")
    model.head.load_state_dict(st["head"])
    model.to(DEVICE).eval()
    safe_print("tau sweep VAL, keep nonempty >= S5-0.02, then min empty_fp")
    best_tau_row, tau_rows = pick_tau_min_fp(model, data.val_loader, floor)
    tau = float(best_tau_row["tau"])
    safe_print(f"chosen tau={tau:.2f}")

    val_g = eval_gated(model, data.val_loader, tau)
    test_g = eval_gated(model, data.test_loader, tau)
    test_ungated = evaluate_test(vae, data.test_loader)
    report = {
        "mode": "s5_skipgate",
        "s5_ckpt": str(S5_BEST),
        "s5_ep": s5_ep,
        "head_best_ep": int(st.get("best_ep", best_ep)),
        "tau": tau,
        "pos_weight": POS_WEIGHT,
        "ungated_val": ungated_val,
        "ungated_test": test_ungated,
        "gated_val": val_g,
        "gated_test": test_g,
        "tau_sweep_val": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
            for r in tau_rows
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST ungated", {k: test_ungated[k] for k in ("dice_mean", "mean_dice_nonempty", "empty_fp_rate")})
    safe_print("TEST gated  ", {k: test_g[k] for k in ("dice_mean", "mean_dice_nonempty", "empty_fp_rate")})


if __name__ == "__main__":
    main()
