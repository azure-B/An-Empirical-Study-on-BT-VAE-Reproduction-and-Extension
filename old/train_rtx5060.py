"""
MS-GSF-BT-VAE training (Windows RTX 5060).
Interrupt-safe: atomic checkpoints, resume BT/SEG, scheduler+RNG restore.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_load_BT import datasets
from models1 import Encoder, VAE

# --------------------- config ---------------------
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# target: latent 1024급 (OOM 시 auto_fit_vram이 ladder로 내려감)
BASE = 64
LATENT_CH = 1024
DROPOUT = 0.05

# preferred batches; auto_fit may reduce these
BT_BATCH = 8
BT_ACCUM = 4
SEG_BATCH = 4
SEG_ACCUM = 4
NUM_WORKERS = 0  # Windows spawn 안정성

BT_EPOCHS = 400
SEG_EPOCHS = 400
BT_PATIENCE = 40
SEG_PATIENCE = 50
PRETRAIN_MAX_HOURS = 96.0
MAX_TOTAL_HOURS = 240.0

# (latent, bt_batch, seg_batch) - 위에서부터 시도
VRAM_LADDER = [
    (1024, 8, 4),
    (1024, 4, 2),
    (896, 4, 2),
    (768, 4, 2),
    (640, 4, 2),
    (512, 4, 2),
    (512, 2, 1),
]

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results" / "ms_gsf_bt_vae_covid"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BT_BEST = CKPT_DIR / "ms_gsf_bt_encoder_best.pt"
BT_LAST = CKPT_DIR / "ms_gsf_bt_encoder_last.pt"
SEG_BEST = RESULT_DIR / "seg_best.pt"
SEG_LAST = RESULT_DIR / "seg_last.pt"
STATUS_JSON = RESULT_DIR / "run_status.json"

CONFIG = {
    "base": BASE,
    "latent_ch": LATENT_CH,
    "dropout": DROPOUT,
    "bt_batch": BT_BATCH,
    "seg_batch": SEG_BATCH,
    "bt_dataset": "covid",
    "seg_dataset": "covid",
}

RUN_START = time.time()


def _oom_probe(latent: int, bt_bs: int, seg_bs: int) -> float:
    """Return peak GB for one BT + one SEG train step; raises on OOM."""
    if DEVICE.type != "cuda":
        return 0.0
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
        loss = p1.float().pow(2).mean() + p2.float().pow(2).mean() + mu1.float().abs().mean()
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
    del enc, opt, sc, v1, v2, p1, p2, mu1, mu2, lv1, lv2, loss
    torch.cuda.empty_cache()

    model = VAE(BASE, DROPOUT, latent).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    x = torch.randn(seg_bs, 1, 256, 256, device=DEVICE)
    y = (torch.rand(seg_bs, 1, 256, 256, device=DEVICE) > 0.9).float()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits, mu, lv, aux = model(x, sample=True, return_aux=True)
        loss = (
            F.binary_cross_entropy_with_logits(logits.float(), y)
            + 0.1 * mu.float().pow(2).mean()
            + 0.3 * F.binary_cross_entropy_with_logits(aux[0].float(), y)
        )
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, sc, x, y, logits, mu, lv, aux, loss
    torch.cuda.empty_cache()
    return peak


def auto_fit_vram(headroom_gb: float = 1.2) -> None:
    """Pick largest latent/batch from VRAM_LADDER that fits with headroom."""
    global LATENT_CH, BT_BATCH, SEG_BATCH, BT_ACCUM, SEG_ACCUM, CONFIG
    if DEVICE.type != "cuda":
        print("CPU mode - keep configured latent/batch")
        return

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = max(1.0, total - headroom_gb)
    print(f"VRAM total={total:.2f}G budget={budget:.2f}G - probing ladder...")

    chosen = None
    for latent, bt_bs, seg_bs in VRAM_LADDER:
        try:
            peak = _oom_probe(latent, bt_bs, seg_bs)
            ok = peak <= budget
            mark = "OK" if ok else "TIGHT"
            print(f"  [{mark}] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs} peak={peak:.2f}G")
            if ok and chosen is None:
                chosen = (latent, bt_bs, seg_bs, peak)
                break
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs}")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] latent={latent} BT_bs={bt_bs} SEG_bs={seg_bs}")
                torch.cuda.empty_cache()
            else:
                raise

    if chosen is None:
        raise RuntimeError("No VRAM_LADDER entry fits this GPU - lower BASE or ladder further.")

    LATENT_CH, BT_BATCH, SEG_BATCH, peak = chosen
    # keep effective batch roughly stable when micro-batch shrinks
    target_bt_eff = 32
    target_seg_eff = 16
    BT_ACCUM = max(1, target_bt_eff // max(1, BT_BATCH))
    SEG_ACCUM = max(1, target_seg_eff // max(1, SEG_BATCH))
    CONFIG = {
        "base": BASE,
        "latent_ch": LATENT_CH,
        "dropout": DROPOUT,
        "bt_batch": BT_BATCH,
        "seg_batch": SEG_BATCH,
        "bt_dataset": "covid",
        "seg_dataset": "covid",
    }
    nparams = sum(p.numel() for p in VAE(BASE, DROPOUT, LATENT_CH).parameters()) / 1e6
    print(
        f"selected latent={LATENT_CH} BT_batch={BT_BATCH}x{BT_ACCUM} "
        f"SEG_batch={SEG_BATCH}x{SEG_ACCUM} peak~{peak:.2f}G params={nparams:.1f}M"
    )
    torch.cuda.empty_cache()


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
    tmp = STATUS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS_JSON)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
        print(f"[warn] RNG restore skipped: {e}")


def check_config(saved: dict | None, where: str) -> None:
    if not saved:
        return
    for k in ("base", "latent_ch", "dropout", "bt_dataset", "seg_dataset"):
        if k in saved and k in CONFIG and saved[k] != CONFIG[k]:
            raise RuntimeError(
                f"Config mismatch in {where}: saved {k}={saved[k]} vs now {CONFIG[k]}. "
                "Delete checkpoints or match config."
            )


def safe_load(path: Path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# --------------------- utils ---------------------
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
    if n != m:
        raise ValueError("cross-correlation matrix must be square")
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


def time_exceeded(hours: float) -> bool:
    return (time.time() - RUN_START) >= hours * 3600


# --------------------- BT ---------------------
def run_bt_epoch(encoder, loader, optimizer=None, scaler=None, accum_steps=1):
    training = optimizer is not None
    encoder.train(training)
    total, total_bt, total_dist = 0.0, 0.0, 0.0
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
    n = max(1, len(loader))
    return {"loss": total / n, "bt": total_bt / n, "dist": total_dist / n}


def save_bt_state(encoder, optimizer, scheduler, epoch, best_bt, bad_epochs, path: Path):
    atomic_torch_save(
        {
            "phase": "bt",
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best_bt,
            "bad_epochs": bad_epochs,
            "config": CONFIG,
            "rng": capture_rng(),
        },
        path,
    )


def train_bt(bt_train_loader, bt_val_loader) -> Path:
    encoder = Encoder(BASE, DROPOUT, LATENT_CH).to(DEVICE)
    bt_optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    bt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bt_optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    bt_scaler = make_scaler()
    best_bt, bad_epochs = float("inf"), 0
    start_epoch = 1

    if BT_LAST.exists():
        state = safe_load(BT_LAST)
        check_config(state.get("config"), str(BT_LAST))
        encoder.load_state_dict(state["encoder"])
        bt_optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            bt_scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_bt = float(state.get("best", best_bt))
        bad_epochs = int(state.get("bad_epochs", 0))
        restore_rng(state.get("rng"))
        print(f"BT resume from epoch {start_epoch} (best={best_bt:.5f}, bad={bad_epochs})")

    if start_epoch > BT_EPOCHS:
        print("BT already finished according to checkpoint")
        return BT_BEST if BT_BEST.exists() else BT_LAST

    bt_phase_start = time.time()
    write_status(phase="bt", epoch=start_epoch - 1, best=best_bt)

    try:
        for epoch in range(start_epoch, BT_EPOCHS + 1):
            train_m = run_bt_epoch(encoder, bt_train_loader, bt_optimizer, bt_scaler, BT_ACCUM)
            val_m = run_bt_epoch(encoder, bt_val_loader)
            bt_scheduler.step(val_m["loss"])

            if val_m["loss"] < best_bt:
                best_bt, bad_epochs = val_m["loss"], 0
                save_bt_state(encoder, bt_optimizer, bt_scheduler, epoch, best_bt, bad_epochs, BT_BEST)
            else:
                bad_epochs += 1

            save_bt_state(encoder, bt_optimizer, bt_scheduler, epoch, best_bt, bad_epochs, BT_LAST)
            write_status(phase="bt", epoch=epoch, best=best_bt, bad_epochs=bad_epochs, val=val_m)

            print(
                f"[BT {epoch:03d}] train={train_m['loss']:.5f} val={val_m['loss']:.5f} "
                f"bt={val_m['bt']:.5f} dist={val_m['dist']:.5f} bad={bad_epochs}"
            )
            if bad_epochs >= BT_PATIENCE:
                print("BT early stopping")
                break
            if time.time() - bt_phase_start >= PRETRAIN_MAX_HOURS * 3600:
                print("BT time budget reached")
                break
    except KeyboardInterrupt:
        print("\n[BT] KeyboardInterrupt - saving LAST checkpoint")
        save_bt_state(
            encoder, bt_optimizer, bt_scheduler, max(start_epoch - 1, 1), best_bt, bad_epochs, BT_LAST
        )
        write_status(phase="bt", interrupted=True)
        raise

    return BT_BEST if BT_BEST.exists() else BT_LAST


# --------------------- SEG ---------------------
def dice_bce(logits, target):
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice_loss = 1 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * dice_loss + 0.5 * bce


def segmentation_loss(logits, target, mu, logvar, aux, epoch):
    main = dice_bce(logits, target)
    aux_loss = 0.3 * dice_bce(aux[0], target) + 0.15 * dice_bce(aux[1], target)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).flatten(1).mean()
    beta = 1e-3 * min(1.0, epoch / 25.0)
    return main + aux_loss + beta * kl, main.detach(), kl.detach()


@torch.no_grad()
def batch_metrics(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = (target >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    pred_sum, target_sum = pred.sum(dims), target.sum(dims)
    union = pred_sum + target_sum - inter
    dice = ((2 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)).mean()
    iou = ((inter + 1e-6) / (union + 1e-6)).mean()
    return dice.item(), iou.item()


def run_seg_epoch(model, loader, epoch, optimizer=None, scaler=None, accum_steps=1):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "main": 0.0, "kl": 0.0, "dice": 0.0, "iou": 0.0}
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        x = as_bchw(batch[0]).to(DEVICE, non_blocking=True)
        y = as_bchw(batch[1]).to(DEVICE, non_blocking=True)
        ctx = amp_context() if training else torch.no_grad()
        with ctx:
            logits, mu, logvar, aux = model(x, sample=training, return_aux=True)
            loss, main, kl = segmentation_loss(logits, y, mu, logvar, aux, epoch)

        if training:
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        dice, iou = batch_metrics(logits.detach(), y)
        sums["loss"] += loss.detach().item()
        sums["main"] += main.item()
        sums["kl"] += kl.item()
        sums["dice"] += dice
        sums["iou"] += iou

    n = max(1, len(loader))
    return {k: v / n for k, v in sums.items()}


def save_seg_state(model, optimizer, scheduler, epoch, best_val, bad_epochs, history, path: Path):
    atomic_torch_save(
        {
            "phase": "seg",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best_val,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": CONFIG,
            "rng": capture_rng(),
        },
        path,
    )


def train_seg(seg_train_loader, seg_val_loader, seg_test_loader, bt_ckpt: Path):
    vae = VAE(BASE, DROPOUT, LATENT_CH).to(DEVICE)
    seg_optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
    seg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        seg_optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-7
    )
    seg_scaler = make_scaler()
    best_val, bad_epochs, history = float("inf"), 0, []
    start_epoch = 1

    if SEG_LAST.exists():
        state = safe_load(SEG_LAST)
        check_config(state.get("config"), str(SEG_LAST))
        vae.load_state_dict(state["model"])
        seg_optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            seg_scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best", best_val))
        bad_epochs = int(state.get("bad_epochs", 0))
        history = state.get("history", [])
        restore_rng(state.get("rng"))
        print(f"SEG resume from epoch {start_epoch} (best={best_val:.5f}, bad={bad_epochs})")
    else:
        if not bt_ckpt.exists():
            raise FileNotFoundError(f"BT checkpoint not found: {bt_ckpt}")
        bt_state = safe_load(bt_ckpt)
        check_config(bt_state.get("config"), str(bt_ckpt))
        missing, unexpected = vae.encoder.load_state_dict(bt_state["encoder"], strict=True)
        print("encoder loaded; missing:", missing, "unexpected:", unexpected)

    if start_epoch > SEG_EPOCHS:
        print("SEG already finished according to checkpoint")
    else:
        write_status(phase="seg", epoch=start_epoch - 1, best=best_val)
        try:
            for epoch in range(start_epoch, SEG_EPOCHS + 1):
                tr = run_seg_epoch(vae, seg_train_loader, epoch, seg_optimizer, seg_scaler, SEG_ACCUM)
                va = run_seg_epoch(vae, seg_val_loader, epoch)
                seg_scheduler.step(va["loss"])
                row = {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in tr.items()},
                    **{f"val_{k}": v for k, v in va.items()},
                    "lr": seg_optimizer.param_groups[0]["lr"],
                }
                history.append(row)

                if va["loss"] < best_val:
                    best_val, bad_epochs = va["loss"], 0
                    save_seg_state(
                        vae, seg_optimizer, seg_scheduler, epoch, best_val, bad_epochs, history, SEG_BEST
                    )
                else:
                    bad_epochs += 1

                save_seg_state(
                    vae, seg_optimizer, seg_scheduler, epoch, best_val, bad_epochs, history, SEG_LAST
                )
                write_status(phase="seg", epoch=epoch, best=best_val, bad_epochs=bad_epochs, val=va)

                with open(RESULT_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)

                print(
                    f"[SEG {epoch:03d}] train loss={tr['loss']:.5f} D={tr['dice']:.4f} IoU={tr['iou']:.4f} | "
                    f"val loss={va['loss']:.5f} D={va['dice']:.4f} IoU={va['iou']:.4f} bad={bad_epochs}"
                )
                if bad_epochs >= SEG_PATIENCE:
                    print("SEG early stopping")
                    break
                if time_exceeded(MAX_TOTAL_HOURS):
                    print("Total time budget reached")
                    break
        except KeyboardInterrupt:
            print("\n[SEG] KeyboardInterrupt - saving LAST checkpoint")
            save_seg_state(
                vae,
                seg_optimizer,
                seg_scheduler,
                max(start_epoch - 1, 1),
                best_val,
                bad_epochs,
                history,
                SEG_LAST,
            )
            write_status(phase="seg", interrupted=True)
            raise

    # test
    ckpt = SEG_BEST if SEG_BEST.exists() else SEG_LAST
    state = safe_load(ckpt)
    vae.load_state_dict(state["model"])
    test_m = run_seg_epoch(vae, seg_test_loader, epoch=SEG_EPOCHS)
    print("TEST:", test_m)
    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(test_m, indent=2), encoding="utf-8")
    write_status(phase="done", test=test_m)
    return test_m


def main():
    seed_everything(SEED)
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    auto_fit_vram()

    print("Loading datasets (COVID only)...")
    bt_data = datasets.covid(batch_size=BT_BATCH, num_workers=NUM_WORKERS)
    seg_data = datasets.covid(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    print("BT(covid):", len(bt_data.train), len(bt_data.val), len(bt_data.test))
    print("SEG(covid):", len(seg_data.train), len(seg_data.val), len(seg_data.test))

    # smoke
    _m = VAE(BASE, DROPOUT, LATENT_CH).to(DEVICE).eval()
    with torch.no_grad():
        _x = torch.randn(1, 1, 256, 256, device=DEVICE)
        _out, _mu, _lv, _aux = _m(_x, sample=False, return_aux=True)
    print("smoke out:", tuple(_out.shape), "mu:", tuple(_mu.shape))
    del _m, _x, _out, _mu, _lv, _aux
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Simpler rule: if SEG_LAST exists, resume SEG only; else run/resume BT then SEG
    if SEG_LAST.exists():
        print("Found SEG_LAST - resume segmentation phase")
        bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
        train_seg(seg_data.train_loader, seg_data.val_loader, seg_data.test_loader, bt_ckpt)
    else:
        bt_ckpt = train_bt(bt_data.train_loader, bt_data.val_loader)
        train_seg(seg_data.train_loader, seg_data.val_loader, seg_data.test_loader, bt_ckpt)

    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user - rerun the same command to resume.")
    except Exception:
        write_status(phase="error", traceback=traceback.format_exc())
        traceback.print_exc()
        raise
