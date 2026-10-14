"""Fine-tune dual decoder: skip tail (TT) + plain tail (empty-FP), gated merge.

Does not overwrite none/ or s5_concat/. Batch 8 for 8GB.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_dual_dec import DualVAE
from train_emptyfp import empty_fp_penalty
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RECON_WEIGHT,
    RESULT_DIR,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    combined_binary_loss_sigmoid,
    safe_load,
    safe_print,
    split_dice_stats,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT_DIR = RESULT_DIR / "dual_dec"
BATCH = 4
LR = 3e-5
EPOCHS = 80
ES_PATIENCE = 20
PLAIN_FP_W = 1.5
W_SKIP = 0.4
W_PLAIN = 0.3
W_MERGED = 0.3


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def load_s5_none(dual: DualVAE) -> None:
    st = safe_load(S5_BEST, map_location="cpu")
    dual.skip_net.load_state_dict(st["model"])
    del st
    st = safe_load(NONE_BEST, map_location="cpu")
    dual.plain_net.load_state_dict(st["model"])
    del st
    safe_print("loaded skip_net from s5, plain_net from none")


def step_loss(vae, flair, seg, plain_fp_w: float):
    flair, seg = _prep(flair, seg)
    merged, skip_l, plain_l, mu, logvar = vae(flair, return_heads=True)
    dice_m, _, _ = combined_binary_loss_sigmoid(merged, seg)
    dice_s, _, _ = combined_binary_loss_sigmoid(skip_l, seg)
    dice_p, _, _ = combined_binary_loss_sigmoid(plain_l, seg)
    extra = empty_fp_penalty(plain_l, seg)
    kl = 1 + logvar - torch.square(mu) - torch.exp(logvar)
    kl = (-0.5 * kl).sum(dim=-1).mean()
    dice = W_SKIP * dice_s + W_PLAIN * dice_p + W_MERGED * dice_m
    loss = RECON_WEIGHT * dice + BETA_LOCAL * kl + plain_fp_w * extra
    return loss, float(dice.item()), float(extra.item())


@torch.no_grad()
def eval_heads(vae, loader):
    vae.eval()
    bags = {k: {"d": [], "g": [], "p": []} for k in ("merged", "skip", "plain")}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        merged, skip_l, plain_l, _, _ = vae(flair, return_heads=True)
        for name, logits in (("merged", merged), ("skip", skip_l), ("plain", plain_l)):
            bags[name]["d"].append(batch_dice(logits, seg))
            ps, gs = batch_pred_gt_sums(logits, seg)
            bags[name]["p"].append(ps)
            bags[name]["g"].append(gs)
    out = {}
    for name, b in bags.items():
        out[name] = split_dice_stats(
            np.concatenate(b["d"]), np.concatenate(b["g"]), np.concatenate(b["p"])
        )
    return out


def pack(st):
    return {
        "dice": st["dice_mean"],
        "nonempty": st["mean_dice_nonempty"],
        "empty_fp": st["empty_fp_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epoch", type=int, default=EPOCHS)
    ap.add_argument("--lambda-fp", type=float, default=PLAIN_FP_W)
    args = ap.parse_args()
    plain_fp_w = float(args.lambda_fp)
    limit = int(args.max_epoch)

    if not S5_BEST.exists() or not NONE_BEST.exists():
        raise FileNotFoundError(f"need {S5_BEST} and {NONE_BEST}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    safe_print("DEVICE", DEVICE, "dual-dec batch", BATCH, "lr", LR, "plain λ", plain_fp_w)

    vae = DualVAE(base=BASE)
    load_s5_none(vae)
    vae.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    init = eval_heads(vae, data.val_loader)
    floor = float(init["skip"]["mean_dice_nonempty"]) - 0.02
    safe_print("INIT VAL merged", pack(init["merged"]), "floor", f"{floor:.4f}")
    safe_print("INIT VAL skip  ", pack(init["skip"]))
    safe_print("INIT VAL plain ", pack(init["plain"]))

    opt = torch.optim.AdamW(vae.parameters(), lr=LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-7)

    best_fp = 2.0
    best_ep = 0
    best_ne = -1.0
    history = [{"init": {k: pack(v) for k, v in init.items()}, "floor": floor}]

    for ep in range(1, limit + 1):
        vae.train()
        tr = tr_extra = 0.0
        ntr = 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, _, extra = step_loss(vae, flair, seg, plain_fp_w)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            tr_extra += extra
            ntr += 1
        heads = eval_heads(vae, data.val_loader)
        m = heads["merged"]
        sched.step(m["empty_fp_rate"])
        safe_print(
            f"[dual] ep {ep}/{limit} loss={tr/max(1,ntr):.4f} extra={tr_extra/max(1,ntr):.4f} "
            f"merged dice={m['dice_mean']:.4f} nonempty={m['mean_dice_nonempty']:.4f} "
            f"empty_fp={m['empty_fp_rate']:.3f} | skip ne={heads['skip']['mean_dice_nonempty']:.4f} "
            f"fp={heads['skip']['empty_fp_rate']:.3f} plain ne={heads['plain']['mean_dice_nonempty']:.4f} "
            f"fp={heads['plain']['empty_fp_rate']:.3f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr / max(1, ntr), "val": {k: pack(v) for k, v in heads.items()}})

        ok = m["mean_dice_nonempty"] is not None and m["mean_dice_nonempty"] >= floor
        payload = {
            "ep": ep,
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "best_fp": best_fp,
            "best_ep": best_ep,
            "best_ne": best_ne,
        }
        if ok and m["empty_fp_rate"] < best_fp:
            best_fp = float(m["empty_fp_rate"])
            best_ep = ep
            best_ne = float(m["mean_dice_nonempty"])
            payload.update({"best_fp": best_fp, "best_ep": best_ep, "best_ne": best_ne})
            atomic_torch_save(payload, OUT_DIR / "seg_best.pt")
            safe_print(f"  saved BEST empty_fp={best_fp:.3f} nonempty={best_ne:.4f}")
        elif (not ok) and m["mean_dice_nonempty"] is not None and m["mean_dice_nonempty"] > best_ne:
            # keep last-best nonempty if floor never met
            best_ne = float(m["mean_dice_nonempty"])
        atomic_torch_save(payload, OUT_DIR / "seg_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_PATIENCE:
            safe_print(f"early stop @{best_ep} empty_fp={best_fp:.3f}")
            break

    if (OUT_DIR / "seg_best.pt").exists():
        st = safe_load(OUT_DIR / "seg_best.pt", map_location="cpu")
        vae.load_state_dict(st["model"])
        best_ep = int(st.get("best_ep", best_ep))
        vae.to(DEVICE)
    else:
        safe_print("no ckpt kept nonempty floor; evaluating last")
    val_h = eval_heads(vae, data.val_loader)
    test_h = eval_heads(vae, data.test_loader)
    report = {
        "mode": "dual_dec",
        "s5_ckpt": str(S5_BEST),
        "none_ckpt": str(NONE_BEST),
        "best_ep": best_ep,
        "lambda_plain_fp": plain_fp_w,
        "lr": LR,
        "batch": BATCH,
        "init_val": {k: pack(v) for k, v in init.items()},
        "nonempty_floor": floor,
        "val": {k: pack(v) for k, v in val_h.items()},
        "test": {k: pack(v) for k, v in test_h.items()},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST merged", pack(test_h["merged"]))
    safe_print("TEST skip  ", pack(test_h["skip"]))
    safe_print("TEST plain ", pack(test_h["plain"]))


if __name__ == "__main__":
    main()
