"""Fine-tune none SEG with an extra loss on empty-slice false positives.

original/ and none/seg_best.pt are not overwritten.
If GT is empty but the model paints a lesion, that slice's loss spikes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RECON_WEIGHT,
    RESULT_DIR,
    SEG_BATCH,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    combined_binary_loss_sigmoid,
    safe_load,
    safe_print,
    split_dice_stats,
)

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT_DIR = RESULT_DIR / "none_emptyfp"
EPOCHS = 25
FT_LR = 3e-5
ES_PATIENCE = 8
EMPTY_FP_WEIGHT = 6.0


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def empty_fp_penalty(logits, seg):
    """Large when GT is empty and predicted mass/peak is high."""
    prob = torch.sigmoid(logits)
    empty = (seg.flatten(1).sum(dim=1) == 0).float()
    mass = prob.mean(dim=(1, 2, 3))
    peak = prob.amax(dim=(1, 2, 3))
    return (empty * (0.7 * mass + 0.3 * peak)).mean()


def step_loss(model, flair, seg, empty_fp_weight: float | None = None):
    flair, seg = _prep(flair, seg)
    output, mu, logvar = model(flair)
    dice_bce, _, _ = combined_binary_loss_sigmoid(output, seg)
    kl = 1 + logvar - torch.square(mu) - torch.exp(logvar)
    kl = (-0.5 * kl).sum(dim=-1).mean()
    extra = empty_fp_penalty(output, seg)
    weight = EMPTY_FP_WEIGHT if empty_fp_weight is None else float(empty_fp_weight)
    loss = RECON_WEIGHT * dice_bce + BETA_LOCAL * kl + weight * extra
    return loss, float(dice_bce.item()), float(extra.item())


@torch.no_grad()
def eval_split(model, loader):
    model.eval()
    dices, gt_s, pred_s = [], [], []
    extra = 0.0
    n = 0
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        out, _, _ = model(flair)
        extra += float(empty_fp_penalty(out, seg).item())
        n += 1
        dices.append(batch_dice(out, seg))
        p, g = batch_pred_gt_sums(out, seg)
        pred_s.append(p)
        gt_s.append(g)
    stats = split_dice_stats(np.concatenate(dices), np.concatenate(gt_s), np.concatenate(pred_s))
    stats["empty_fp_penalty"] = extra / max(1, n)
    return stats


def main():
    if not NONE_BEST.exists():
        raise FileNotFoundError(NONE_BEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    safe_print(
        "DEVICE:", DEVICE, "empty-FP FT from", NONE_BEST,
        f"batch={SEG_BATCH} lr={FT_LR} λ={EMPTY_FP_WEIGHT}",
    )

    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    none_ep = int(st.get("best_ep", -1))
    del st
    vae.to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=FT_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    base_val = eval_split(vae, data.val_loader)
    floor = float(base_val["mean_dice_nonempty"]) - 0.02
    safe_print(
        "none VAL ",
        f"dice={base_val['dice_mean']:.4f} nonempty={base_val['mean_dice_nonempty']:.4f} "
        f"empty_fp={base_val['empty_fp_rate']:.3f} floor={floor:.4f}",
    )

    best_fp = 2.0
    best_ep = 0
    history = [{"init_val": base_val, "none_ep": none_ep}]
    for ep in range(1, EPOCHS + 1):
        vae.train()
        tr = tr_extra = 0.0
        ntr = 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, _, extra = step_loss(vae, flair, seg)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            tr_extra += extra
            ntr += 1
        val = eval_split(vae, data.val_loader)
        sched.step(val["empty_fp_rate"])
        safe_print(
            f"[emptyfp] ep {ep}/{EPOCHS} loss={tr/max(1,ntr):.4f} extra={tr_extra/max(1,ntr):.4f} "
            f"val_dice={val['dice_mean']:.4f} nonempty={val['mean_dice_nonempty']:.4f} "
            f"empty_fp={val['empty_fp_rate']:.3f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr / max(1, ntr), "val": val})
        ok = val["mean_dice_nonempty"] is not None and val["mean_dice_nonempty"] >= floor
        payload = {
            "ep": ep,
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "best_fp": best_fp,
            "best_ep": best_ep,
            "none_ckpt": str(NONE_BEST),
        }
        if ok and val["empty_fp_rate"] < best_fp:
            best_fp = float(val["empty_fp_rate"])
            best_ep = ep
            payload["best_fp"] = best_fp
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, OUT_DIR / "seg_best.pt")
            safe_print(f"  saved BEST empty_fp={best_fp:.3f} nonempty={val['mean_dice_nonempty']:.4f}")
        atomic_torch_save(payload, OUT_DIR / "seg_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_PATIENCE:
            safe_print(f"early stop @{best_ep} empty_fp={best_fp:.3f}")
            break

    if (OUT_DIR / "seg_best.pt").exists():
        st = safe_load(OUT_DIR / "seg_best.pt", map_location="cpu")
        vae.load_state_dict(st["model"])
        best_ep = int(st.get("best_ep", best_ep))
    else:
        safe_print("no ckpt kept nonempty floor; evaluating last")
    vae.to(DEVICE)
    val_b = eval_split(vae, data.val_loader)
    test_b = eval_split(vae, data.test_loader)
    report = {
        "mode": "none_emptyfp_ft",
        "none_ckpt": str(NONE_BEST),
        "none_ep": none_ep,
        "best_ep": best_ep,
        "lambda": EMPTY_FP_WEIGHT,
        "lr": FT_LR,
        "init_val": base_val,
        "nonempty_floor": floor,
        "val": val_b,
        "test": test_b,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST emptyfp:", json.dumps(test_b, indent=2))


if __name__ == "__main__":
    main()
