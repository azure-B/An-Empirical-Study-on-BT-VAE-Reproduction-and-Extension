"""
BP original none → S5 concat skip comparison.

- Does NOT modify original/
- Imports original.models1 for none
- Uses root lazy data_load_BT (batch / on-disk)
- Protocol from original/BT-VAE.ipynb (not paper table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.data import datasets  # noqa: E402
from pipeline.models_skip import VAE as SkipVAE  # noqa: E402

# Import original models1 by path without mutating sys.path / without editing original/
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "original_models1", ROOT / "original" / "models1.py"
)
_orig = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_orig)
OrigEncoder = _orig.Encoder
OrigVAE = _orig.VAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = 32
BT_EPOCHS = 200
BT_LR = 3e-4
# Notebook uses 64, but [B,1024,1024] KL peaks ~12.5GB on 8GB GPU → BSOD.
# Measured peak: B16≈3.3GB (safe margin), B32≈6.4GB (thin). Use 16.
BT_BATCH = 16
SEG_EPOCHS = 400
SEG_LR = 1e-4
SEG_BATCH = 16
SEG_ES_PATIENCE = 40  # stop if val best not updated
BETA_LOCAL = 0.01
RECON_WEIGHT = 0.99
NUM_WORKERS = 2

CKPT_DIR = ROOT / "old" / "checkpoints"
RESULT_DIR = ROOT / "old" / "results" / "bp_original"
BT_BEST = CKPT_DIR / "bt_encoder_best.pt"
BT_LAST = CKPT_DIR / "bt_encoder_last.pt"


def safe_print(*a, **k):
    print(*a, **k, flush=True)


def safe_load(path: Path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----- BT losses (from original notebook) -----
def kl_gaussian(mu1, sigma1, mu2, sigma2, eps: float = 1e-12):
    var1 = (sigma1 ** 2).clamp_min(eps)
    var2 = (sigma2 ** 2).clamp_min(eps)
    diff2 = (mu1 - mu2) ** 2
    return 0.5 * (torch.log(var2) - torch.log(var1) + (var1 + diff2) / var2 - 1.0)


def kl_matrix_loss_weighted(K, w_diag=1.0, w_off=1.0, target_off=3.0):
    C = K.size(0)
    eye = torch.eye(C, device=K.device, dtype=K.dtype).bool()
    diag_loss = (K.diagonal() ** 2).mean()
    off_loss = ((K[~eye] - target_off) ** 2).mean()
    return w_diag * diag_loss + w_off * off_loss


def cross_corr_mu_loss(m1v, m2v, eps=1e-12, lambda_off=1.0):
    B = m1v.size(0)
    z1n = (m1v - m1v.mean(0)) / (m1v.std(0) + eps)
    z2n = (m2v - m2v.mean(0)) / (m2v.std(0) + eps)
    C = (z1n.T @ z2n) / float(B)
    D = C.size(0)
    diag = torch.diag(C)
    off = C - torch.diag(diag)
    diag_loss = (1.0 - diag).pow(2).mean()
    off_loss = off.pow(2).sum() / (D * (D - 1) + eps)
    return diag_loss + lambda_off * off_loss, C


def bt_batch_loss(encoder, view1, view2, ep: int):
    if view1.ndim == 3:
        view1 = view1.unsqueeze(1)
    if view2.ndim == 3:
        view2 = view2.unsqueeze(1)
    m1, v1_, z1 = encoder(view1)
    m2, v2_, z2 = encoder(view2)
    v1_ = v1_.clamp(-5, 2)
    v2_ = v2_.clamp(-5, 2)
    gap = nn.AdaptiveAvgPool2d(1)
    s = lambda x: gap(x).flatten(1) if x.dim() == 4 else x
    m1v, v1v = s(m1), s(v1_)
    m2v, v2v = s(m2), s(v2_)
    sig1 = (0.5 * v1v).exp()
    sig2 = (0.5 * v2v).exp()
    K12_b = kl_gaussian(
        m1v.unsqueeze(2), sig1.unsqueeze(2), m2v.unsqueeze(1), sig2.unsqueeze(1)
    )
    K21_b = kl_gaussian(
        m2v.unsqueeze(2), sig2.unsqueeze(2), m1v.unsqueeze(1), sig1.unsqueeze(1)
    )
    K = 0.5 * (K12_b + K21_b).mean(dim=0)
    tgt = 1.0 + min(1.0, ep / 100.0)
    loss_kl = kl_matrix_loss_weighted(K, w_diag=1.0, w_off=1.0, target_off=tgt)
    loss_cmu, _ = cross_corr_mu_loss(m1v, m2v, lambda_off=1.0)
    return loss_kl + 0.2 * loss_cmu


def freeze_bn(encoder: nn.Module):
    for m in encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def run_bt():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_split(batch_size=BT_BATCH, num_workers=NUM_WORKERS)
    safe_print("BT BP train_split:", len(data.train), len(data.val), len(data.test))

    encoder = OrigEncoder(base=BASE).to(DEVICE)
    conv1x1 = nn.Conv2d(2, 1, 1).to(DEVICE)  # notebook includes this in opt
    freeze_bn(encoder)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(conv1x1.parameters()),
        lr=BT_LR,
        weight_decay=1e-4,
    )
    start_ep = 1
    best_val = float("inf")
    if BT_LAST.exists():
        st = safe_load(BT_LAST)
        encoder.load_state_dict(st["encoder"])
        if "conv1x1" in st:
            conv1x1.load_state_dict(st["conv1x1"])
        if "optimizer" in st:
            opt.load_state_dict(st["optimizer"])
        start_ep = int(st.get("ep", 0)) + 1
        best_val = float(st.get("best_val", best_val))
        safe_print(f"BT resume from ep {start_ep}, best_val={best_val:.6f}")

    amp = False  # notebook cell_05
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for ep in range(start_ep, BT_EPOCHS + 1):
        encoder.train()
        freeze_bn(encoder)
        running = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            view1 = batch[2].to(DEVICE, non_blocking=True)
            view2 = batch[3].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = bt_batch_loss(encoder, view1, view2, ep)
            if not torch.isfinite(loss):
                raise ValueError("BT loss NaN/Inf")
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
        train_avg = running / max(1, len(data.train_loader))

        encoder.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in data.val_loader:
                view1 = batch[2].to(DEVICE, non_blocking=True)
                view2 = batch[3].to(DEVICE, non_blocking=True)
                val_running += float(bt_batch_loss(encoder, view1, view2, ep).item())
        val_avg = val_running / max(1, len(data.val_loader))
        safe_print(
            f"[BT] ep {ep}/{BT_EPOCHS} train={train_avg:.6f} val={val_avg:.6f} "
            f"({time.time() - t0:.1f}s)"
        )

        payload = {
            "ep": ep,
            "encoder": encoder.state_dict(),
            "conv1x1": conv1x1.state_dict(),
            "optimizer": opt.state_dict(),
            "best_val": best_val,
        }
        if val_avg < best_val:
            best_val = val_avg
            payload["best_val"] = best_val
            atomic_torch_save(payload, BT_BEST)
            safe_print(f"  saved BT_BEST val={best_val:.6f}")
        atomic_torch_save(payload, BT_LAST)

    safe_print("BT done. best:", BT_BEST if BT_BEST.exists() else BT_LAST)
    return BT_BEST if BT_BEST.exists() else BT_LAST


# ----- SEG losses -----
def binary_dice_loss_sigmoid(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_binary_loss_sigmoid(pred, target, dice_weight=0.5, ce_weight=0.5):
    dice = binary_dice_loss_sigmoid(pred, target)
    bce = nn.BCEWithLogitsLoss()
    ce = bce(pred.view(pred.size(0), -1), target.view(target.size(0), -1))
    return dice_weight * dice + ce_weight * ce, float(dice.item()), float(ce.item())


def seg_step_loss(model, flair, seg, return_out=False):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    output, mu_local, logvar_local = model(flair)
    dice_loss_combined, _, _ = combined_binary_loss_sigmoid(output, seg)
    kl_loss_local = 1 + logvar_local - torch.square(mu_local) - torch.exp(logvar_local)
    kl_loss_local = torch.sum(kl_loss_local, dim=-1)
    kl_loss_local *= -0.5
    loss = RECON_WEIGHT * dice_loss_combined + BETA_LOCAL * torch.mean(kl_loss_local)
    stats = (loss, float(dice_loss_combined.item()), float(torch.mean(kl_loss_local).item()))
    if return_out:
        return stats + (output, seg)
    return stats


@torch.no_grad()
def batch_dice(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + 1e-6) / (union + 1e-6)).detach().cpu().numpy()


@torch.no_grad()
def batch_pred_gt_sums(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    return (
        pred.sum(dim=(1, 2, 3)).detach().cpu().numpy(),
        target.sum(dim=(1, 2, 3)).detach().cpu().numpy(),
    )


@torch.no_grad()
def split_dice_stats(dices, gt_sums, pred_sums):
    nonempty = gt_sums > 0
    empty = ~nonempty
    ne = dices[nonempty]
    return {
        "dice_mean": float(dices.mean()),
        "dice_median": float(np.median(dices)),
        "mean_dice_nonempty": float(ne.mean()) if ne.size else None,
        "median_dice_nonempty": float(np.median(ne)) if ne.size else None,
        "n": int(dices.size),
        "n_nonempty": int(nonempty.sum()),
        "n_empty": int(empty.sum()),
        "empty_fp_rate": float((pred_sums[empty] > 0).mean()) if empty.any() else None,
    }


@torch.no_grad()
def evaluate_test(model, loader):
    model.eval()
    dices = []
    gt_sums = []
    pred_sums = []
    for batch in loader:
        flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
        if flair.ndim == 3:
            flair = flair.unsqueeze(1)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        out, _, _ = model(flair)
        dices.append(batch_dice(out, seg))
        pred_s, gt_s = batch_pred_gt_sums(out, seg)
        pred_sums.append(pred_s)
        gt_sums.append(gt_s)
    return split_dice_stats(
        np.concatenate(dices), np.concatenate(gt_sums), np.concatenate(pred_sums)
    )


def freeze_encoder(vae: nn.Module):
    for p in vae.encoder.parameters():
        p.requires_grad = False
    vae.encoder.eval()
    n_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in vae.parameters())
    safe_print(f"encoder frozen: trainable {n_train}/{n_all} params")


def encoder_eval_if_frozen(vae: nn.Module, frozen: bool):
    if frozen:
        vae.encoder.eval()


def load_bt_into_encoder(encoder: nn.Module, bt_ckpt: Path):
    st = safe_load(bt_ckpt, map_location="cpu")
    enc_state = st["encoder"] if isinstance(st, dict) and "encoder" in st else st
    enc_state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in enc_state.items()
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    safe_print(f"encoder load missing={len(missing)} unexpected={len(unexpected)}")


def run_seg(mode: str, bt_ckpt: Path, max_ep: int | None = None):
    assert mode in {"none", "s5", "none_freeze"}
    out_name = {"none": "none", "s5": "s5_concat", "none_freeze": "none_freeze"}[mode]
    out_dir = RESULT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_best = out_dir / "seg_best.pt"
    seg_last = out_dir / "seg_last.pt"
    frozen = mode == "none_freeze"

    # S5 concat peaks ~12GB at batch 16 (WDDM overcommit). 8GB card: use 8.
    seg_bs = 8 if mode == "s5" else SEG_BATCH
    data = datasets.bp_train_30(batch_size=seg_bs, num_workers=NUM_WORKERS)
    safe_print(
        f"SEG({mode}) BP train_30:", len(data.train), len(data.val), len(data.test),
        f"batch={seg_bs}",
    )

    if mode in {"none", "none_freeze"}:
        vae = OrigVAE(base=BASE).to(DEVICE)
    else:
        vae = SkipVAE(base=BASE).to(DEVICE)

    if frozen:
        freeze_encoder(vae)
        opt = torch.optim.AdamW(
            [p for p in vae.parameters() if p.requires_grad],
            lr=SEG_LR,
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(vae.parameters(), lr=SEG_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.95, patience=3, min_lr=1e-8)

    start_ep = 1
    best_loss = float("inf")
    best_ep = 0
    if seg_last.exists():
        # Load to CPU first: map_location=cuda keeps a full extra copy on GPU → OOM on resume.
        st = safe_load(seg_last, map_location="cpu")
        vae.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            try:
                sched.load_state_dict(st["scheduler"])
            except Exception:
                pass
        start_ep = int(st.get("ep", 0)) + 1
        best_loss = float(st.get("best_loss", best_loss))
        best_ep = int(st.get("best_ep", 0))
        del st
        safe_print(f"SEG resume {mode} from ep {start_ep}, best_loss={best_loss:.4f}")
    else:
        load_bt_into_encoder(vae.encoder, bt_ckpt)
    if frozen:
        freeze_encoder(vae)

    history = []
    limit = SEG_EPOCHS if max_ep is None else min(int(max_ep), SEG_EPOCHS)
    if start_ep > limit:
        safe_print(f"SEG {mode}: already at ep {start_ep-1} >= max {limit}, skip train")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    for ep in range(start_ep, limit + 1):
        vae.train()
        encoder_eval_if_frozen(vae, frozen)
        tr_loss = tr_dice = tr_kl = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            flair = batch[0].to(DEVICE)
            seg = batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, dice_v, kl_v = seg_step_loss(vae, flair, seg)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            tr_dice += dice_v
            tr_kl += kl_v
        ntr = max(1, len(data.train_loader))
        train_loss = tr_loss / ntr

        vae.eval()
        va_loss = 0.0
        va_dices, va_gt, va_pred = [], [], []
        with torch.no_grad():
            for batch in data.val_loader:
                flair = batch[0].to(DEVICE)
                seg = batch[1].to(DEVICE)
                if frozen:
                    loss, _, _, out, seg_b = seg_step_loss(vae, flair, seg, return_out=True)
                    va_dices.append(batch_dice(out, seg_b))
                    pred_s, gt_s = batch_pred_gt_sums(out, seg_b)
                    va_pred.append(pred_s)
                    va_gt.append(gt_s)
                else:
                    loss, _, _ = seg_step_loss(vae, flair, seg)
                va_loss += float(loss.item())
        val_loss = va_loss / max(1, len(data.val_loader))
        sched.step(val_loss)

        extra = ""
        val_split = None
        if frozen and va_dices:
            val_split = split_dice_stats(
                np.concatenate(va_dices), np.concatenate(va_gt), np.concatenate(va_pred)
            )
            extra = (
                f" nonempty={val_split['mean_dice_nonempty']:.4f}"
                f" empty_fp={val_split['empty_fp_rate']:.3f}"
            )
        safe_print(
            f"[SEG {mode}] ep {ep}/{SEG_EPOCHS} train={train_loss:.4f} "
            f"val={val_loss:.4f} dice_term={tr_dice/ntr:.4f}{extra} ({time.time()-t0:.1f}s)"
        )
        rec = {"ep": ep, "train_loss": train_loss, "val_loss": val_loss}
        if val_split:
            rec["val_nonempty_dice"] = val_split["mean_dice_nonempty"]
            rec["val_empty_fp"] = val_split["empty_fp_rate"]
        history.append(rec)

        payload = {
            "ep": ep,
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_loss": best_loss,
            "best_ep": best_ep,
            "mode": mode,
            "freeze_encoder": frozen,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            best_ep = ep
            payload["best_loss"] = best_loss
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, seg_best)
            safe_print(f"  saved SEG_BEST loss={best_loss:.4f}")
        atomic_torch_save(payload, seg_last)

        if best_ep > 0 and (ep - best_ep) >= SEG_ES_PATIENCE:
            safe_print(
                f"early stop {mode}: no val best for {SEG_ES_PATIENCE} epochs "
                f"(best={best_loss:.4f} @ {best_ep})"
            )
            break

    # test best
    if seg_best.exists():
        st = safe_load(seg_best)
        vae.load_state_dict(st["model"])
        best_ep = int(st.get("best_ep", best_ep))
        best_loss = float(st.get("best_loss", best_loss))
    metrics = evaluate_test(vae, data.test_loader)
    report = {
        "mode": mode,
        "dataset": "bp_train_30",
        "bt_ckpt": str(bt_ckpt),
        "best_val_loss": best_loss,
        "best_ep": best_ep,
        "freeze_encoder": frozen,
        "test": metrics,
        "protocol": {
            "bt_epochs": BT_EPOCHS,
            "bt_batch": BT_BATCH,
            "bt_lr": BT_LR,
            "seg_epochs": SEG_EPOCHS,
            "seg_batch": SEG_BATCH,
            "seg_lr": SEG_LR,
            "beta_local": BETA_LOCAL,
            "selection": "best_val_loss",
            "early_stop_patience": SEG_ES_PATIENCE,
            "freeze_encoder": frozen,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print(f"TEST {mode}:", json.dumps(metrics, indent=2))
    return report


def write_summary():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, folder in [
        ("none", "none"),
        ("s5_concat", "s5_concat"),
        ("none_freeze", "none_freeze"),
    ]:
        rp = RESULT_DIR / folder / "report.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        t = r.get("test", {})
        rows.append(
            (
                name,
                t.get("dice_mean"),
                t.get("dice_median"),
                t.get("mean_dice_nonempty"),
                t.get("median_dice_nonempty"),
                r.get("best_val_loss"),
                r.get("best_ep"),
            )
        )
    lines = [
        "# BP original none vs S5 concat",
        "",
        "- Protocol: `original/BT-VAE.ipynb` (not paper table)",
        "- `original/` code untouched; lazy loader = root `data_load_BT.py`",
        "- S5: B0–B3 concat+3×3, B4 off (`models_bp_skip.py`)",
        "",
        "| Mode | TEST mean | TEST median | nonempty mean | nonempty median | best val loss | best ep |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m, md, nm, nmd, bl, be in rows:
        lines.append(
            f"| {name} | {m:.4f} | {md:.4f} | {nm:.4f} | {nmd:.4f} | {bl:.4f} | {be} |"
            if m is not None
            else f"| {name} | - | - | - | - | - | - |"
        )
    (RESULT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print("wrote", RESULT_DIR / "SUMMARY.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["all", "bt", "none", "s5", "none_freeze", "summary"],
        default="all",
    )
    ap.add_argument("--max-seg-epoch", type=int, default=None)
    args = ap.parse_args()
    safe_print("DEVICE:", DEVICE, "ROOT:", ROOT)

    if args.phase in {"all", "bt"}:
        bt_ckpt = run_bt()
    else:
        bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
        if not bt_ckpt.exists():
            raise FileNotFoundError(f"BT ckpt missing: {bt_ckpt}")

    if args.phase in {"all", "none"}:
        run_seg("none", bt_ckpt, max_ep=args.max_seg_epoch)

    if args.phase in {"all", "s5"}:
        run_seg("s5", bt_ckpt, max_ep=args.max_seg_epoch)

    if args.phase == "none_freeze":
        run_seg("none_freeze", bt_ckpt, max_ep=args.max_seg_epoch)

    if args.phase in {"all", "summary", "none", "s5", "none_freeze"}:
        write_summary()


if __name__ == "__main__":
    main()
