"""Few-label spatial prototypes on frozen BT μ (+16×16 map). original/ untouched.

1) No-train prototype on raw spatial μ (GAP 대신 8×8).
2) Train a small projector + FG/BG proto (encoder stays BT).
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
from models_proto import ProtoHead, encode_spatial
from train_original_bp import (
    BASE,
    BT_BEST,
    BT_LAST,
    DEVICE,
    NUM_WORKERS,
    OrigEncoder,
    SEG_BATCH,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    combined_binary_loss_sigmoid,
    load_bt_into_encoder,
    safe_load,
    safe_print,
    split_dice_stats,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "bp_original" / "bt_proto"
EPOCHS = 80
LR = 1e-3
ES_PATIENCE = 15


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


@torch.no_grad()
def raw_mu_prototypes(encoder, loader):
    fg = bg = None
    n_fg = n_bg = 0.0
    encoder.eval()
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        mu, _ = encode_spatial(encoder, flair)
        feat = F.normalize(mu, dim=1)
        m8 = F.interpolate(seg, size=feat.shape[-2:], mode="area")
        b, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(-1, c)
        w8 = m8.reshape(-1)
        fg_m = w8 > 0.25
        bg_m = w8 <= 0.25
        if fg_m.any():
            s = feat[fg_m].sum(0)
            fg = s if fg is None else fg + s
            n_fg += float(fg_m.sum())
        if bg_m.any():
            s = feat[bg_m].sum(0)
            bg = s if bg is None else bg + s
            n_bg += float(bg_m.sum())
    if fg is None or bg is None or n_fg < 1 or n_bg < 1:
        raise RuntimeError("failed to build raw μ prototypes")
    return F.normalize(fg / n_fg, dim=0), F.normalize(bg / n_bg, dim=0)


@torch.no_grad()
def eval_raw_mu(encoder, loader, fg, bg, thr: float = 0.0):
    encoder.eval()
    dices, gt_sums, pred_sums = [], [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        mu, _ = encode_spatial(encoder, flair)
        feat = F.normalize(mu, dim=1)
        logit = (feat * fg.view(1, -1, 1, 1)).sum(1, keepdim=True) - (
            feat * bg.view(1, -1, 1, 1)
        ).sum(1, keepdim=True)
        logit = F.interpolate(logit, size=seg.shape[-2:], mode="bilinear", align_corners=False)
        pred = (logit > thr).float()
        inter = (pred * seg).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        dices.append(((2 * inter + 1e-6) / (ps + gs + 1e-6)).cpu().numpy())
        pred_sums.append(ps.cpu().numpy())
        gt_sums.append(gs.cpu().numpy())
    return split_dice_stats(
        np.concatenate(dices), np.concatenate(gt_sums), np.concatenate(pred_sums)
    )


@torch.no_grad()
def eval_head(encoder, head, loader):
    encoder.eval()
    head.eval()
    dices, gt_s, pred_s = [], [], []
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        mu, mid = encode_spatial(encoder, flair)
        logits = head(mu, mid, out_size=seg.shape[-2:])
        dices.append(batch_dice(logits, seg))
        p, g = batch_pred_gt_sums(logits, seg)
        pred_s.append(p)
        gt_s.append(g)
    return split_dice_stats(np.concatenate(dices), np.concatenate(gt_s), np.concatenate(pred_s))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
    if not bt_ckpt.exists():
        raise FileNotFoundError(bt_ckpt)

    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    safe_print("DEVICE:", DEVICE, "BT proto", bt_ckpt, "batch", SEG_BATCH)

    encoder = OrigEncoder(base=BASE)
    load_bt_into_encoder(encoder, bt_ckpt)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(DEVICE).eval()

    safe_print("building raw spatial-μ prototypes on train")
    fg, bg = raw_mu_prototypes(encoder, data.train_loader)
    raw_val = eval_raw_mu(encoder, data.val_loader, fg, bg)
    raw_test = eval_raw_mu(encoder, data.test_loader, fg, bg)
    safe_print(
        "raw μ VAL ",
        f"dice={raw_val['dice_mean']:.4f} nonempty={raw_val['mean_dice_nonempty']:.4f} "
        f"empty_fp={raw_val['empty_fp_rate']:.3f}",
    )
    safe_print(
        "raw μ TEST",
        f"dice={raw_test['dice_mean']:.4f} nonempty={raw_test['mean_dice_nonempty']:.4f} "
        f"empty_fp={raw_test['empty_fp_rate']:.3f}",
    )

    head = ProtoHead().to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    best = -1.0
    best_ep = 0
    # overall Dice rewards all-empty; select on lesion overlap.
    history = [{"raw_mu_val": raw_val, "raw_mu_test": raw_test}]
    for ep in range(1, EPOCHS + 1):
        head.train()
        encoder.eval()
        tr = 0.0
        ntr = 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            with torch.no_grad():
                mu, mid = encode_spatial(encoder, flair)
            opt.zero_grad(set_to_none=True)
            logits = head(mu, mid, out_size=seg.shape[-2:])
            loss, _, _ = combined_binary_loss_sigmoid(logits, seg)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        val = eval_head(encoder, head, data.val_loader)
        score = float(val["mean_dice_nonempty"] or 0.0)
        sched.step(score)
        safe_print(
            f"[proto] ep {ep}/{EPOCHS} loss={tr/max(1,ntr):.4f} "
            f"val_dice={val['dice_mean']:.4f} nonempty={val['mean_dice_nonempty']:.4f} "
            f"empty_fp={val['empty_fp_rate']:.3f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr / max(1, ntr), "val": val})
        payload = {"ep": ep, "head": head.state_dict(), "best": best, "best_ep": best_ep}
        if score > best:
            best = score
            best_ep = ep
            payload["best"] = best
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, OUT_DIR / "head_best.pt")
            safe_print(f"  saved HEAD_BEST nonempty={best:.4f}")
        atomic_torch_save(payload, OUT_DIR / "head_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_PATIENCE:
            safe_print(f"early stop @{best_ep} best={best:.4f}")
            break

    st = safe_load(OUT_DIR / "head_best.pt", map_location="cpu")
    head.load_state_dict(st["head"])
    val_b = eval_head(encoder, head, data.val_loader)
    test_b = eval_head(encoder, head, data.test_loader)
    report = {
        "mode": "bt_spatial_proto",
        "bt_ckpt": str(bt_ckpt),
        "best_ep": int(st.get("best_ep", best_ep)),
        "raw_mu": {"val": raw_val, "test": raw_test},
        "learned": {"val": val_b, "test": test_b},
        "protocol": {
            "encoder": "frozen BT",
            "features": "spatial μ 8x8 + encoder 16x16, no GAP",
            "head": "1x1 projector + FG/BG prototype",
            "labels": "bp_train_30 only",
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST raw μ:", json.dumps(raw_test, indent=2))
    safe_print("TEST learned proto:", json.dumps(test_b, indent=2))


if __name__ == "__main__":
    main()
