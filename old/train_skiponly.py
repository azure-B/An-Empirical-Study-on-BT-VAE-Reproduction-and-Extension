"""
BT-VAE + Attention skips with learnable gamma blend (COVID).
Decoder: x += gamma*A + (1-gamma)*B  (A=AG skip, B=ungated skip).
SEG early-stops on val Dice patience.
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_load_BT import datasets
from models1_skip import Encoder, VAE

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# restored original-scale backbone
BASE = 32
LATENT_CH = 1024
DROPOUT = 0.1  # original/models1.py default

BT_BATCH = 8
BT_ACCUM = 4
SEG_BATCH = 4
SEG_ACCUM = 4
NUM_WORKERS = 0

BT_EPOCHS = 200
SEG_EPOCHS = 400
BT_PATIENCE = 9999
# Stop SEG when val Dice stalls (prior attn peak ~ep105; avoid 300 wasted epochs)
SEG_PATIENCE = 40
PRETRAIN_MAX_HOURS = 96.0
MAX_TOTAL_HOURS = 240.0

# KL warmup: start near 0, ramp to BETA_KL over WARMUP epochs
BETA_KL = 1e-3
KL_WARMUP_EPOCHS = 40

VRAM_LADDER = [
    (1024, 8, 4),
    (1024, 4, 2),
    (1024, 2, 1),
    (768, 4, 2),
    (512, 4, 2),
]

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints_skip_covid_attn_gamma01"
RESULT_DIR = ROOT / "results" / "skiponly_covid_attn_gamma01"
# Reuse BT encoder from prior attn run (encoder unchanged; gamma is decoder-only)
BT_SOURCE_DIR = ROOT / "checkpoints_skip_covid_attn"
# Direct gamma param init (not logit); mostly ungated skip at start
GAMMA_INIT = 0.01
GAMMA_LR = 1e-2  # gamma is 5 scalars; needs higher LR than backbone 1e-4
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BT_BEST = CKPT_DIR / "bt_encoder_best.pt"
BT_LAST = CKPT_DIR / "bt_encoder_last.pt"
SEG_BEST = RESULT_DIR / "seg_best.pt"
SEG_LAST = RESULT_DIR / "seg_last.pt"
STATUS_JSON = RESULT_DIR / "run_status.json"

CONFIG = {
    "base": BASE,
    "latent_ch": LATENT_CH,
    "dropout": DROPOUT,
    "bt_batch": BT_BATCH,
    "seg_batch": SEG_BATCH,
    "model": "attn_skip_gamma",
    "skip_blend": "gamma*AG + (1-gamma)*ungated",
    "gamma_init": GAMMA_INIT,
    "gamma_param": "direct_clamp",
    "gamma_lr": GAMMA_LR,
    "seg_patience": SEG_PATIENCE,
    "seg_best_metric": "val_dice",
    "bt_dataset": "covid",
    "seg_dataset": "covid",
}

RUN_START = time.time()


def safe_print(*args, **kwargs) -> None:
    """Windows Tee/pipe can raise OSError(22) mid-run; never kill training for logs."""
    try:
        print(*args, **kwargs)
        try:
            import sys
            if hasattr(sys.stdout, "flush"):
                sys.stdout.flush()
        except Exception:
            pass
    except OSError:
        pass


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_status(**kwargs) -> None:
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    if STATUS_JSON.exists():
        try:
            old = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
            old.update(payload)
            payload = old
        except Exception:
            pass
    # clear stale crash marker on successful progress updates
    if payload.get("phase") in ("bt", "seg", "done") and "traceback" not in kwargs:
        payload.pop("traceback", None)
    tmp = STATUS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS_JSON)
    try:
        from update_results_md import main as refresh_results_md

        refresh_results_md()
    except Exception:
        pass


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    last_err = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.25 * (attempt + 1))
    # last resort: direct overwrite if replace still locked
    try:
        torch.save(obj, path)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    except Exception:
        if last_err is not None:
            raise last_err
        raise


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict | None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        safe_print(f"[warn] RNG restore skipped: {e}")


def check_config(saved: dict | None, where: str, keys: tuple[str, ...] | None = None) -> None:
    if not saved:
        return
    keys = keys or ("base", "latent_ch", "dropout", "model", "bt_dataset", "seg_dataset")
    for k in keys:
        if k in saved and k in CONFIG and saved[k] != CONFIG[k]:
            raise RuntimeError(
                f"Config mismatch in {where}: saved {k}={saved[k]} vs now {CONFIG[k]}"
            )


def safe_load(path: Path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def as_bchw(x):
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW/BHW, got {tuple(x.shape)}")
    return x


def make_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=USE_AMP)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=USE_AMP)


def amp_context():
    if not USE_AMP:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1, z2, lambd=5e-3, eps=1e-4):
    z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + eps)
    z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + eps)
    c = z1.T @ z2 / z1.size(0)
    on = (torch.diagonal(c) - 1).pow(2).mean()
    off = off_diagonal(c).pow(2).mean()
    return on + lambd * off


def symmetric_diag_kl(mu1, lv1, mu2, lv2):
    mu1 = F.adaptive_avg_pool2d(mu1, 1).flatten(1).float()
    mu2 = F.adaptive_avg_pool2d(mu2, 1).flatten(1).float()
    lv1 = F.adaptive_avg_pool2d(lv1, 1).flatten(1).float().clamp(-6, 2)
    lv2 = F.adaptive_avg_pool2d(lv2, 1).flatten(1).float().clamp(-6, 2)
    v1, v2 = lv1.exp(), lv2.exp()
    d2 = (mu1 - mu2).pow(2)
    kl12 = 0.5 * (lv2 - lv1 + (v1 + d2) / v2 - 1)
    kl21 = 0.5 * (lv1 - lv2 + (v2 + d2) / v1 - 1)
    return 0.5 * (kl12.mean() + kl21.mean())


def kl_standard_normal(mu, logvar):
    """Mean-reduced KL to N(0,I). Stable for large spatial latents."""
    logvar = logvar.clamp(-6.0, 2.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def dice_bce(logits, target):
    probs = torch.sigmoid(logits.float())
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    return 0.5 * dice_loss + 0.5 * bce


def batch_metrics(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits.float()) >= thr).float()
    target = (target >= 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    ps = pred.sum(dim=(1, 2, 3))
    ts = target.sum(dim=(1, 2, 3))
    dice = ((2 * inter + 1e-6) / (ps + ts + 1e-6)).mean()
    iou = ((inter + 1e-6) / (ps + ts - inter + 1e-6)).mean()
    return dice.item(), iou.item()


@torch.no_grad()
def per_image_test_report(model, loader, thr=0.5, min_gt_px=1):
    """Per-image Dice/IoU on TEST for epoch-400 baseline KPIs."""
    model.eval()
    dices_all, ious_all, dices_ne = [], [], []
    empty_gt = 0
    for batch in loader:
        x = as_bchw(batch[0]).to(DEVICE, non_blocking=True)
        y = as_bchw(batch[1]).to(DEVICE, non_blocking=True)
        logits, _, _ = model(x, sample=False)
        pred = (torch.sigmoid(logits.float()) >= thr).float()
        target = (y >= 0.5).float()
        inter = (pred * target).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        ts = target.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + ts + 1e-6)
        iou = (inter + 1e-6) / (ps + ts - inter + 1e-6)
        for i in range(x.size(0)):
            d = float(dice[i].item())
            o = float(iou[i].item())
            gt_px = int(ts[i].item())
            dices_all.append(d)
            ious_all.append(o)
            if gt_px < min_gt_px:
                empty_gt += 1
            else:
                dices_ne.append(d)
    dices_all = np.asarray(dices_all, dtype=np.float64)
    ious_all = np.asarray(ious_all, dtype=np.float64)
    dices_ne = np.asarray(dices_ne, dtype=np.float64)
    return {
        "n": int(dices_all.size),
        "empty_gt": int(empty_gt),
        "mean_dice_all": float(dices_all.mean()) if dices_all.size else None,
        "median_dice_all": float(np.median(dices_all)) if dices_all.size else None,
        "mean_dice_nonempty": float(dices_ne.mean()) if dices_ne.size else None,
        "median_dice_nonempty": float(np.median(dices_ne)) if dices_ne.size else None,
        "mean_iou_all": float(ious_all.mean()) if ious_all.size else None,
    }


def beta_at(epoch: int) -> float:
    if KL_WARMUP_EPOCHS <= 0:
        return BETA_KL
    return BETA_KL * min(1.0, epoch / float(KL_WARMUP_EPOCHS))


def time_exceeded(hours: float) -> bool:
    return (time.time() - RUN_START) >= hours * 3600


def _oom_probe(latent: int, bt_bs: int, seg_bs: int) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    enc = Encoder(BASE, DROPOUT, latent).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-4)
    sc = torch.cuda.amp.GradScaler(enabled=True)
    enc.train()
    v1 = torch.randn(bt_bs, 1, 256, 256, device=DEVICE)
    v2 = torch.randn(bt_bs, 1, 256, 256, device=DEVICE)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        p1, mu1, lv1 = enc.forward_bt(v1)
        p2, mu2, lv2 = enc.forward_bt(v2)
        loss = barlow_loss(p1.float(), p2.float()) + 0.05 * symmetric_diag_kl(mu1, lv1, mu2, lv2)
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
    del enc, opt, sc, v1, v2, mu1, mu2, lv1, lv2, p1, p2, loss
    torch.cuda.empty_cache()

    model = VAE(BASE, DROPOUT, latent).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    x = torch.randn(seg_bs, 1, 256, 256, device=DEVICE)
    y = (torch.rand_like(x) > 0.9).float()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits, mu, lv = model(x)
        loss = dice_bce(logits, y) + BETA_KL * kl_standard_normal(mu, lv)
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, sc, x, y, logits, mu, lv, loss
    torch.cuda.empty_cache()
    return peak


def auto_fit_vram(headroom_gb: float = 1.2) -> None:
    global LATENT_CH, BT_BATCH, SEG_BATCH, BT_ACCUM, SEG_ACCUM, CONFIG
    if DEVICE.type != "cuda":
        safe_print("CPU mode - keep configured latent/batch")
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = max(1.0, total - headroom_gb)
    safe_print(f"VRAM total={total:.2f}G budget={budget:.2f}G - probing ladder...")
    chosen = None
    for latent, bt_bs, seg_bs in VRAM_LADDER:
        try:
            peak = _oom_probe(latent, bt_bs, seg_bs)
            ok = peak <= budget
            safe_print(f"  [{'OK' if ok else 'TIGHT'}] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs} peak={peak:.2f}G")
            if ok and chosen is None:
                chosen = (latent, bt_bs, seg_bs, peak)
                break
        except torch.cuda.OutOfMemoryError:
            safe_print(f"  [OOM] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs}")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                safe_print(f"  [OOM] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs}")
                torch.cuda.empty_cache()
            else:
                raise
    if chosen is None:
        raise RuntimeError("No VRAM_LADDER entry fits")
    LATENT_CH, BT_BATCH, SEG_BATCH, peak = chosen
    BT_ACCUM = max(1, 32 // max(1, BT_BATCH))
    SEG_ACCUM = max(1, 16 // max(1, SEG_BATCH))
    CONFIG = {
        "base": BASE,
        "latent_ch": LATENT_CH,
        "dropout": DROPOUT,
        "bt_batch": BT_BATCH,
        "seg_batch": SEG_BATCH,
        "model": "attn_skip_gamma",
        "skip_blend": "gamma*AG + (1-gamma)*ungated",
        "gamma_init": GAMMA_INIT,
        "gamma_param": "direct_clamp",
        "gamma_lr": GAMMA_LR,
        "seg_patience": SEG_PATIENCE,
        "seg_best_metric": "val_dice",
        "bt_dataset": "covid",
        "seg_dataset": "covid",
    }
    n = sum(p.numel() for p in VAE(BASE, DROPOUT, LATENT_CH).parameters()) / 1e6
    safe_print(
        f"selected latent={LATENT_CH} BT={BT_BATCH}x{BT_ACCUM} "
        f"SEG={SEG_BATCH}x{SEG_ACCUM} peak~{peak:.2f}G params={n:.1f}M"
    )


# --------------------- BT ---------------------
def run_bt_epoch(encoder, loader, optimizer=None, scaler=None, accum_steps=1):
    training = optimizer is not None
    encoder.train(training)
    total, total_bt, total_dist = 0.0, 0.0, 0.0
    n_ok = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        v1 = as_bchw(batch[2]).to(DEVICE, non_blocking=True)
        v2 = as_bchw(batch[3]).to(DEVICE, non_blocking=True)
        ctx = amp_context() if training else torch.no_grad()
        with ctx:
            p1, mu1, lv1 = encoder.forward_bt(v1)
            p2, mu2, lv2 = encoder.forward_bt(v2)
            loss_bt = barlow_loss(p1.float(), p2.float())
            loss_dist = symmetric_diag_kl(mu1, lv1, mu2, lv2)
            loss = loss_bt + 0.05 * loss_dist

        if not torch.isfinite(loss):
            safe_print(f"[warn] non-finite BT loss at step {step} - skip")
            if training:
                optimizer.zero_grad(set_to_none=True)
            continue

        if training:
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total += loss.detach().item()
        total_bt += loss_bt.detach().item()
        total_dist += loss_dist.detach().item()
        n_ok += 1

    n = max(1, n_ok)
    return {"loss": total / n, "bt": total_bt / n, "dist": total_dist / n}


def train_bt(bt_train_loader, bt_val_loader) -> Path:
    encoder = Encoder(BASE, DROPOUT, LATENT_CH).to(DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=6, min_lr=1e-6
    )
    scaler = make_scaler()
    best_bt, bad, start = float("inf"), 0, 1

    if BT_LAST.exists():
        st = safe_load(BT_LAST)
        check_config(st.get("config"), str(BT_LAST))
        encoder.load_state_dict(st["encoder"])
        opt.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            sched.load_state_dict(st["scheduler"])
        start = int(st["epoch"]) + 1
        best_bt = float(st.get("best", best_bt))
        bad = int(st.get("bad_epochs", 0))
        restore_rng(st.get("rng"))
        safe_print(f"BT resume epoch={start} best={best_bt:.5f}")

    if start > BT_EPOCHS:
        return BT_BEST if BT_BEST.exists() else BT_LAST

    t0 = time.time()
    try:
        for epoch in range(start, BT_EPOCHS + 1):
            tr = run_bt_epoch(encoder, bt_train_loader, opt, scaler, BT_ACCUM)
            va = run_bt_epoch(encoder, bt_val_loader)
            sched.step(va["loss"])
            payload = {
                "phase": "bt",
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best": best_bt,
                "bad_epochs": bad,
                "config": CONFIG,
                "rng": capture_rng(),
            }
            if va["loss"] < best_bt:
                best_bt, bad = va["loss"], 0
                payload["best"] = best_bt
                payload["bad_epochs"] = bad
                atomic_torch_save(payload, BT_BEST)
            else:
                bad += 1
                payload["bad_epochs"] = bad
            atomic_torch_save(payload, BT_LAST)
            write_status(phase="bt", epoch=epoch, best=best_bt, bad_epochs=bad, val=va)
            safe_print(
                f"[BT {epoch:03d}] train={tr['loss']:.5f} val={va['loss']:.5f} "
                f"bt={va['bt']:.5f} dist={va['dist']:.5f} bad={bad}"
            )
            if bad >= BT_PATIENCE:
                safe_print("BT early stopping")
                break
            if time.time() - t0 >= PRETRAIN_MAX_HOURS * 3600:
                safe_print("BT time budget")
                break
    except KeyboardInterrupt:
        safe_print("\n[BT] interrupted - saved LAST")
        atomic_torch_save(
            {
                "phase": "bt",
                "epoch": max(start - 1, 1),
                "encoder": encoder.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best": best_bt,
                "bad_epochs": bad,
                "config": CONFIG,
                "rng": capture_rng(),
            },
            BT_LAST,
        )
        raise
    return BT_BEST if BT_BEST.exists() else BT_LAST


# --------------------- SEG ---------------------
def run_seg_epoch(model, loader, epoch, optimizer=None, scaler=None, accum_steps=1):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "main": 0.0, "kl": 0.0, "dice": 0.0, "iou": 0.0}
    n_ok = 0
    beta = beta_at(epoch)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        x = as_bchw(batch[0]).to(DEVICE, non_blocking=True)
        y = as_bchw(batch[1]).to(DEVICE, non_blocking=True)
        ctx = amp_context() if training else torch.no_grad()
        with ctx:
            logits, mu, logvar = model(x, sample=training)
            main = dice_bce(logits, y)
            kl = kl_standard_normal(mu, logvar)
            loss = main + beta * kl

        if not torch.isfinite(loss):
            safe_print(f"[warn] non-finite SEG loss ep{epoch} step{step} - skip")
            if training:
                optimizer.zero_grad(set_to_none=True)
            continue

        if training:
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name.split(".")[-1].startswith("gamma"):
                            p.clamp_(0.0, 1.0)

        dice, iou = batch_metrics(logits.detach(), y)
        sums["loss"] += loss.detach().item()
        sums["main"] += main.detach().item()
        sums["kl"] += kl.detach().item()
        sums["dice"] += dice
        sums["iou"] += iou
        n_ok += 1

    n = max(1, n_ok)
    return {k: v / n for k, v in sums.items()}


def train_seg(seg_train_loader, seg_val_loader, seg_test_loader, bt_ckpt: Path):
    vae = VAE(BASE, DROPOUT, LATENT_CH).to(DEVICE)
    gamma_params = [p for n, p in vae.named_parameters() if n.split(".")[-1].startswith("gamma")]
    other_params = [p for n, p in vae.named_parameters() if not n.split(".")[-1].startswith("gamma")]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": 1e-4, "weight_decay": 1e-4},
            {"params": gamma_params, "lr": GAMMA_LR, "weight_decay": 0.0},
        ]
    )
    safe_print(
        f"SEG opt: backbone lr=1e-4 (n={len(other_params)}), "
        f"gamma lr={GAMMA_LR} wd=0 (n={len(gamma_params)})"
    )
    # LR still tracks val loss; BEST ckpt tracks val Dice
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=12, min_lr=1e-7
    )
    scaler = make_scaler()
    best_dice, bad, history, start = -1.0, 0, [], 1

    if SEG_LAST.exists():
        st = safe_load(SEG_LAST)
        check_config(st.get("config"), str(SEG_LAST))
        vae.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            sched.load_state_dict(st["scheduler"])
        start = int(st["epoch"]) + 1
        # backward/forward compatible: prefer best_dice, else treat old 'best' as dice if metric says so
        if "best_dice" in st:
            best_dice = float(st["best_dice"])
        elif st.get("config", {}).get("seg_best_metric") == "val_dice":
            best_dice = float(st.get("best", -1.0))
        else:
            best_dice = -1.0
        bad = int(st.get("bad_epochs", 0))
        history = st.get("history", [])
        restore_rng(st.get("rng"))
        safe_print(f"SEG resume epoch={start} best_dice={best_dice:.5f}")
    else:
        if not Path(bt_ckpt).exists():
            raise FileNotFoundError(bt_ckpt)
        bt = safe_load(bt_ckpt)
        # Encoder-only: allow prior attn_skip BT ckpt (gamma is decoder-only)
        check_config(
            bt.get("config"),
            str(bt_ckpt),
            keys=("base", "latent_ch", "dropout", "bt_dataset", "seg_dataset"),
        )
        missing, unexpected = vae.encoder.load_state_dict(bt["encoder"], strict=True)
        safe_print("encoder loaded; missing:", missing, "unexpected:", unexpected)

    try:
        for epoch in range(start, SEG_EPOCHS + 1):
            tr = run_seg_epoch(vae, seg_train_loader, epoch, opt, scaler, SEG_ACCUM)
            va = run_seg_epoch(vae, seg_val_loader, epoch)
            sched.step(va["loss"])
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
                "lr": opt.param_groups[0]["lr"],
                "beta": beta_at(epoch),
            }
            history.append(row)
            gammas = vae.decoder.gamma_values()
            row.update({f"gamma_{k}": v for k, v in gammas.items()})
            payload = {
                "phase": "seg",
                "epoch": epoch,
                "model": vae.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best": best_dice,
                "best_dice": best_dice,
                "bad_epochs": bad,
                "history": history,
                "config": CONFIG,
                "rng": capture_rng(),
            }
            improved = (
                torch.isfinite(torch.tensor(va["dice"]))
                and va["dice"] > best_dice
            )
            if improved:
                best_dice, bad = float(va["dice"]), 0
                payload["best"] = best_dice
                payload["best_dice"] = best_dice
                payload["bad_epochs"] = bad
                atomic_torch_save(payload, SEG_BEST)
            else:
                bad += 1
                payload["bad_epochs"] = bad
            atomic_torch_save(payload, SEG_LAST)
            with open(RESULT_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=history[0].keys())
                w.writeheader()
                w.writerows(history)
            write_status(
                phase="seg",
                epoch=epoch,
                best=best_dice,
                best_dice=best_dice,
                bad_epochs=bad,
                val=va,
                gammas=gammas,
            )
            safe_print(
                f"[SEG {epoch:03d}] train loss={tr['loss']:.5f} D={tr['dice']:.4f} IoU={tr['iou']:.4f} | "
                f"val loss={va['loss']:.5f} D={va['dice']:.4f} IoU={va['iou']:.4f} "
                f"bad={bad} bestD={best_dice:.4f} beta={beta_at(epoch):.4g} "
                f"g=[{gammas['g4']:.5f},{gammas['g3']:.5f},{gammas['g2']:.5f},"
                f"{gammas['g1']:.5f},{gammas['g0']:.5f}]"
            )
            if bad >= SEG_PATIENCE:
                safe_print(f"SEG early stopping (patience={SEG_PATIENCE})")
                break
            if time_exceeded(MAX_TOTAL_HOURS):
                safe_print("Total time budget")
                break
    except KeyboardInterrupt:
        safe_print("\n[SEG] interrupted - saved LAST")
        atomic_torch_save(
            {
                "phase": "seg",
                "epoch": max(start - 1, 1),
                "model": vae.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best": best_dice,
                "best_dice": best_dice,
                "bad_epochs": bad,
                "history": history,
                "config": CONFIG,
                "rng": capture_rng(),
            },
            SEG_LAST,
        )
        raise

    # test with Dice-best (early-stop or epoch cap)
    best_epoch = None
    final_gammas = None
    if SEG_BEST.exists():
        st = safe_load(SEG_BEST)
        vae.load_state_dict(st["model"])
        best_epoch = int(st.get("epoch", -1))
        final_gammas = vae.decoder.gamma_values()
        safe_print(f"loaded SEG_BEST dice={st.get('best_dice', st.get('best'))} epoch={best_epoch}")
        safe_print(f"SEG_BEST gammas (1=AG,0=additive): {final_gammas}")
    test_m = run_seg_epoch(vae, seg_test_loader, epoch=SEG_EPOCHS)
    per_img = per_image_test_report(vae, seg_test_loader)
    safe_print("TEST:", test_m)
    safe_print("TEST per-image:", per_img)
    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(test_m, indent=2), encoding="utf-8")

    hist_rows = [h for h in history if int(h.get("epoch", 0)) <= SEG_EPOCHS]
    best_val_row = max(hist_rows, key=lambda h: float(h.get("val_dice", -1.0)), default=None)
    last_row = hist_rows[-1] if hist_rows else None
    report = {
        "cutoff_note": (
            f"SEG early-stop patience={SEG_PATIENCE}; best ckpt used for TEST; "
            f"max epochs={SEG_EPOCHS}"
        ),
        "history": {
            "best_val_dice": float(best_val_row["val_dice"]) if best_val_row else None,
            "best_val_epoch": int(best_val_row["epoch"]) if best_val_row else None,
            "last_epoch": int(last_row["epoch"]) if last_row else None,
            "last_val_dice": float(last_row["val_dice"]) if last_row else None,
        },
        "seg_best_ckpt": {
            "epoch": best_epoch,
            "best_dice": float(best_dice),
            "gammas": final_gammas,
        },
        "test_batch_avg": test_m,
        "test_per_image": per_img,
        "original_ref": {"mean": 0.76, "median": 0.84},
        "additive_skip_ref": {
            "mean_dice_nonempty": 0.7828528351253934,
            "median_dice_nonempty": 0.8391492366790771,
        },
        "attn_ref": {
            "test_dice": 0.7729,
            "mean_dice_nonempty": 0.7787,
            "median_dice_nonempty": 0.8299,
        },
    }
    (RESULT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_status(
        phase="done",
        best_dice=best_dice,
        test=test_m,
        test_per_image=per_img,
        gammas=final_gammas,
    )
    return test_m


def ensure_bt_from_prior() -> bool:
    """Copy prior attn BT encoder into this run dir if missing (encoder is identical)."""
    if BT_BEST.exists() or BT_LAST.exists():
        return True
    src_best = BT_SOURCE_DIR / "bt_encoder_best.pt"
    src_last = BT_SOURCE_DIR / "bt_encoder_last.pt"
    copied = False
    if src_best.exists():
        shutil.copy2(src_best, BT_BEST)
        safe_print(f"reused BT_BEST from {src_best}")
        copied = True
    if src_last.exists():
        shutil.copy2(src_last, BT_LAST)
        safe_print(f"reused BT_LAST from {src_last}")
        copied = True
    return copied


def main():
    seed_everything(SEED)
    safe_print("device:", DEVICE)
    if DEVICE.type == "cuda":
        safe_print("GPU:", torch.cuda.get_device_name(0))
    auto_fit_vram()

    safe_print("Loading datasets (COVID only)...")
    bt_data = datasets.covid(batch_size=BT_BATCH, num_workers=NUM_WORKERS)
    seg_data = datasets.covid(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    safe_print("BT(covid):", len(bt_data.train), len(bt_data.val), len(bt_data.test))
    safe_print("SEG(covid):", len(seg_data.train), len(seg_data.val), len(seg_data.test))

    m = VAE(BASE, DROPOUT, LATENT_CH).to(DEVICE).eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        out, mu, lv = m(x, sample=False)
    g0 = m.decoder.gamma_values()
    safe_print("smoke out:", tuple(out.shape), "mu:", tuple(mu.shape))
    safe_print(f"init gammas (expect ~{GAMMA_INIT}): {g0}")
    n = sum(p.numel() for p in m.parameters()) / 1e6
    safe_print(
        f"params: {n:.2f}M  base={BASE} latent={LATENT_CH} "
        f"model={CONFIG['model']} gamma_init={GAMMA_INIT} SEG_patience={SEG_PATIENCE}"
    )
    del m, x, out, mu, lv
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    ensure_bt_from_prior()

    if SEG_LAST.exists():
        safe_print("Found SEG_LAST - resume SEG")
        bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
        train_seg(seg_data.train_loader, seg_data.val_loader, seg_data.test_loader, bt_ckpt)
    elif BT_BEST.exists() or BT_LAST.exists():
        bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
        safe_print(f"Skip BT training; load encoder from {bt_ckpt}")
        train_seg(seg_data.train_loader, seg_data.val_loader, seg_data.test_loader, bt_ckpt)
    else:
        bt_ckpt = train_bt(bt_data.train_loader, bt_data.val_loader)
        train_seg(seg_data.train_loader, seg_data.val_loader, seg_data.test_loader, bt_ckpt)
    safe_print("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("Stopped by user - rerun to resume.")
    except Exception:
        write_status(phase="error", traceback=traceback.format_exc())
        traceback.print_exc()
        raise
