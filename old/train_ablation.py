"""
5-stage COVID skip ablation (no Attention Gate).

  S1 none         — original, no skips
  S2 full         — additive B0..B4 (reuse prior archive if present)
  S3 b0b3         — additive B0..B3 (B4 off only)
  S4 b0b3_learn   — B0..B3 zero-init 1x1 + learnable scale
  S5 b0b3_concat  — B0..B3 concat + 3x3 conv fuse

Usage:
  python train_ablation.py --stage 1
  python train_ablation.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
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
from models_ablation import (
    SKIP_B0B3,
    SKIP_B0B3_CONCAT,
    SKIP_B0B3_LEARN,
    SKIP_FULL,
    SKIP_NONE,
    Encoder,
    VAE,
)

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

BASE = 32
LATENT_CH = 1024
DROPOUT = 0.1
BT_BATCH, BT_ACCUM = 8, 4
SEG_BATCH, SEG_ACCUM = 4, 4
NUM_WORKERS = 0
BT_EPOCHS, SEG_EPOCHS = 200, 400
BT_PATIENCE, SEG_PATIENCE = 9999, 40
BETA_KL, KL_WARMUP_EPOCHS = 1e-3, 40
SCALE_LR = 1e-2
PRETRAIN_MAX_HOURS, MAX_TOTAL_HOURS = 96.0, 240.0

ROOT = Path(__file__).resolve().parent
SHARED_BT_DIR = ROOT / "checkpoints_ablation_bt"
BT_SOURCE = ROOT / "checkpoints_skip_covid_attn"

STAGES = {
    1: {"mode": SKIP_NONE, "name": "s1_original_none", "desc": "original no skip"},
    2: {
        "mode": SKIP_FULL,
        "name": "s2_full_additive",
        "desc": "additive B0..B4 (reuse prior)",
        "reuse_from": "results/archive/skiponly_covid_base32",
    },
    3: {"mode": SKIP_B0B3, "name": "s3_b0b3_additive", "desc": "additive B0..B3 (B4 off)"},
    4: {"mode": SKIP_B0B3_LEARN, "name": "s4_b0b3_learn", "desc": "learnable scale B0..B3 (B4 off)"},
    5: {"mode": SKIP_B0B3_CONCAT, "name": "s5_b0b3_concat", "desc": "concat+3x3 B0..B3 (B4 off)"},
}

RUN_START = time.time()


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
        import sys
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
    except OSError:
        pass


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_load(path: Path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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
    return (torch.diagonal(c) - 1).pow(2).mean() + lambd * off_diagonal(c).pow(2).mean()


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
    logvar = logvar.clamp(-6.0, 2.0)
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


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
    ps, ts = pred.sum(dim=(1, 2, 3)), target.sum(dim=(1, 2, 3))
    dice = ((2 * inter + 1e-6) / (ps + ts + 1e-6)).mean()
    iou = ((inter + 1e-6) / (ps + ts - inter + 1e-6)).mean()
    return dice.item(), iou.item()


@torch.no_grad()
def per_image_test_report(model, loader, thr=0.5, min_gt_px=1):
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
        ps, ts = pred.sum(dim=(1, 2, 3)), target.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + ts + 1e-6)
        iou = (inter + 1e-6) / (ps + ts - inter + 1e-6)
        for i in range(x.size(0)):
            d, o = float(dice[i]), float(iou[i])
            dices_all.append(d)
            ious_all.append(o)
            if int(ts[i]) < min_gt_px:
                empty_gt += 1
            else:
                dices_ne.append(d)
    dices_all = np.asarray(dices_all)
    ious_all = np.asarray(ious_all)
    dices_ne = np.asarray(dices_ne)
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


def write_status(status_json: Path, **kwargs) -> None:
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    if status_json.exists():
        try:
            old = json.loads(status_json.read_text(encoding="utf-8"))
            old.update(payload)
            payload = old
        except Exception:
            pass
    status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_shared_bt() -> Path:
    SHARED_BT_DIR.mkdir(parents=True, exist_ok=True)
    bt_best = SHARED_BT_DIR / "bt_encoder_best.pt"
    bt_last = SHARED_BT_DIR / "bt_encoder_last.pt"
    if bt_best.exists() or bt_last.exists():
        return bt_best if bt_best.exists() else bt_last
    src_best = BT_SOURCE / "bt_encoder_best.pt"
    src_last = BT_SOURCE / "bt_encoder_last.pt"
    if src_best.exists():
        shutil.copy2(src_best, bt_best)
        safe_print(f"reused BT_BEST from {src_best}")
    if src_last.exists():
        shutil.copy2(src_last, bt_last)
        safe_print(f"reused BT_LAST from {src_last}")
    if not bt_best.exists() and not bt_last.exists():
        raise FileNotFoundError("No BT encoder found; place bt_encoder_best.pt under checkpoints_skip_covid_attn")
    return bt_best if bt_best.exists() else bt_last


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
                if model.skip_mode == SKIP_B0B3_LEARN:
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if name.split(".")[-1].startswith("scale"):
                                p.clamp_(-2.0, 2.0)

        dice, iou = batch_metrics(logits.detach(), y)
        sums["loss"] += loss.detach().item()
        sums["main"] += main.detach().item()
        sums["kl"] += kl.detach().item()
        sums["dice"] += dice
        sums["iou"] += iou
        n_ok += 1

    n = max(1, n_ok)
    return {k: v / n for k, v in sums.items()}


def train_stage(stage: int, seg_train, seg_val, seg_test, bt_ckpt: Path) -> dict:
    meta = STAGES[stage]
    mode = meta["mode"]
    result_dir = ROOT / "results" / "ablation" / meta["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    seg_best = result_dir / "seg_best.pt"
    seg_last = result_dir / "seg_last.pt"
    status_json = result_dir / "run_status.json"
    config = {
        "stage": stage,
        "skip_mode": mode,
        "desc": meta["desc"],
        "base": BASE,
        "latent_ch": LATENT_CH,
        "dropout": DROPOUT,
        "seg_patience": SEG_PATIENCE,
        "seg_best_metric": "val_dice",
        "bt_dataset": "covid",
        "seg_dataset": "covid",
    }

    safe_print("=" * 60)
    safe_print(f"STAGE {stage}: {meta['desc']}  mode={mode}")
    safe_print(f"result_dir={result_dir}")
    safe_print("=" * 60)

    vae = VAE(BASE, DROPOUT, LATENT_CH, skip_mode=mode).to(DEVICE)
    if mode == SKIP_B0B3_LEARN:
        scale_params = [p for n, p in vae.named_parameters() if n.split(".")[-1].startswith("scale")]
        other = [p for n, p in vae.named_parameters() if not n.split(".")[-1].startswith("scale")]
        opt = torch.optim.AdamW(
            [
                {"params": other, "lr": 1e-4, "weight_decay": 1e-4},
                {"params": scale_params, "lr": SCALE_LR, "weight_decay": 0.0},
            ]
        )
        safe_print(f"learnable scales n={len(scale_params)} lr={SCALE_LR}")
    else:
        opt = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12, min_lr=1e-7)
    scaler = make_scaler()
    best_dice, bad, history, start = -1.0, 0, [], 1

    if seg_last.exists():
        st = safe_load(seg_last)
        # allow resume across minor decoder module adds (e.g. fuse3)
        missing, unexpected = vae.load_state_dict(st["model"], strict=False)
        if missing or unexpected:
            safe_print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        try:
            opt.load_state_dict(st["optimizer"])
        except Exception as e:
            safe_print(f"[resume] optimizer reset ({e})")
        if "scheduler" in st:
            try:
                sched.load_state_dict(st["scheduler"])
            except Exception:
                pass
        start = int(st["epoch"]) + 1
        best_dice = float(st.get("best_dice", st.get("best", -1.0)))
        bad = int(st.get("bad_epochs", 0))
        history = st.get("history", [])
        safe_print(f"SEG resume epoch={start} best_dice={best_dice:.5f}")
    else:
        bt = safe_load(bt_ckpt)
        missing, unexpected = vae.encoder.load_state_dict(bt["encoder"], strict=False)
        safe_print("encoder loaded; missing:", len(missing), "unexpected:", len(unexpected))

    t0 = time.time()
    for epoch in range(start, SEG_EPOCHS + 1):
        tr = run_seg_epoch(vae, seg_train, epoch, opt, scaler, SEG_ACCUM)
        va = run_seg_epoch(vae, seg_val, epoch)
        sched.step(va["loss"])
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "lr": opt.param_groups[0]["lr"],
            "beta": beta_at(epoch),
        }
        scales = vae.decoder.scale_values()
        if scales:
            row.update({f"scale_{k}": v for k, v in scales.items()})
        history.append(row)

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
            "config": config,
        }
        if torch.isfinite(torch.tensor(va["dice"])) and va["dice"] > best_dice:
            best_dice, bad = float(va["dice"]), 0
            payload["best"] = best_dice
            payload["best_dice"] = best_dice
            payload["bad_epochs"] = bad
            atomic_torch_save(payload, seg_best)
        else:
            bad += 1
            payload["bad_epochs"] = bad
        atomic_torch_save(payload, seg_last)

        with open(result_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)

        write_status(
            status_json,
            phase="seg",
            stage=stage,
            skip_mode=mode,
            epoch=epoch,
            best_dice=best_dice,
            bad_epochs=bad,
            val=va,
            scales=scales,
        )
        extra = ""
        if scales:
            extra = (
                f" s=[{scales['s3']:.4f},{scales['s2']:.4f},"
                f"{scales['s1']:.4f},{scales['s0']:.4f}]"
            )
        safe_print(
            f"[S{stage} SEG {epoch:03d}] train D={tr['dice']:.4f} | "
            f"val D={va['dice']:.4f} IoU={va['iou']:.4f} "
            f"bad={bad} bestD={best_dice:.4f} beta={beta_at(epoch):.4g}{extra}"
        )
        if bad >= SEG_PATIENCE:
            safe_print(f"SEG early stopping (patience={SEG_PATIENCE})")
            break
        if (time.time() - t0) >= MAX_TOTAL_HOURS * 3600:
            safe_print("time budget")
            break

    best_epoch = None
    final_scales = None
    if seg_best.exists():
        st = safe_load(seg_best)
        vae.load_state_dict(st["model"])
        best_epoch = int(st.get("epoch", -1))
        final_scales = vae.decoder.scale_values()
        safe_print(f"loaded SEG_BEST dice={st.get('best_dice')} epoch={best_epoch}")

    test_m = run_seg_epoch(vae, seg_test, epoch=SEG_EPOCHS)
    per_img = per_image_test_report(vae, seg_test)
    safe_print("TEST:", test_m)
    safe_print("TEST per-image:", per_img)

    report = {
        "stage": stage,
        "skip_mode": mode,
        "desc": meta["desc"],
        "seg_best_ckpt": {"epoch": best_epoch, "best_dice": float(best_dice), "scales": final_scales},
        "test_batch_avg": test_m,
        "test_per_image": per_img,
    }
    (result_dir / "test_metrics.json").write_text(json.dumps(test_m, indent=2), encoding="utf-8")
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_status(
        status_json,
        phase="done",
        stage=stage,
        skip_mode=mode,
        best_dice=best_dice,
        test=test_m,
        test_per_image=per_img,
        scales=final_scales,
    )
    safe_print(f"STAGE {stage} DONE")
    return report


def reuse_stage_report(stage: int, src_dir: Path) -> dict:
    """Copy prior experiment metrics into ablation folder; do not retrain."""
    meta = STAGES[stage]
    result_dir = ROOT / "results" / "ablation" / meta["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    test_m = json.loads((src_dir / "test_metrics.json").read_text(encoding="utf-8"))
    src_report = src_dir / "report_ep400.json"
    if not src_report.exists():
        src_report = src_dir / "report.json"
    src = json.loads(src_report.read_text(encoding="utf-8")) if src_report.exists() else {}
    per_img = src.get("test_per_image") or {}
    sb = src.get("seg_best_ckpt") or {}
    if not sb and src.get("history_le400"):
        h = src["history_le400"]
        sb = {"epoch": h.get("best_val_epoch"), "best_dice": h.get("best_val_dice")}
    report = {
        "stage": stage,
        "skip_mode": meta["mode"],
        "desc": meta["desc"],
        "reused_from": str(src_dir).replace("\\", "/"),
        "seg_best_ckpt": {
            "epoch": sb.get("epoch"),
            "best_dice": sb.get("best_dice"),
            "scales": None,
        },
        "test_batch_avg": test_m,
        "test_per_image": per_img,
    }
    (result_dir / "test_metrics.json").write_text(json.dumps(test_m, indent=2), encoding="utf-8")
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_status(
        result_dir / "run_status.json",
        phase="done",
        stage=stage,
        skip_mode=meta["mode"],
        reused_from=report["reused_from"],
        best_dice=sb.get("best_dice"),
        test=test_m,
        test_per_image=per_img,
    )
    safe_print("=" * 60)
    safe_print(f"STAGE {stage}: REUSE only - {report['reused_from']}")
    safe_print(
        f"TEST Dice={test_m.get('dice'):.4f} nonempty="
        f"{per_img.get('mean_dice_nonempty'):.4f}/{per_img.get('median_dice_nonempty'):.4f}"
    )
    safe_print("=" * 60)
    return report


def write_summary(reports: list[dict]) -> None:
    out = ROOT / "results" / "ablation" / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Skip Ablation Summary (COVID)",
        "",
        "| Stage | Mode | TEST Dice | nonempty mean | nonempty median | best val |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in reports:
        t = r.get("test_batch_avg") or {}
        p = r.get("test_per_image") or {}
        sb = r.get("seg_best_ckpt") or {}
        lines.append(
            f"| S{r['stage']} | `{r['skip_mode']}` | "
            f"{t.get('dice', float('nan')):.4f} | "
            f"{p.get('mean_dice_nonempty', float('nan')):.4f} | "
            f"{p.get('median_dice_nonempty', float('nan')):.4f} | "
            f"{sb.get('best_dice', float('nan')):.4f} @"
            f"{sb.get('epoch', '-')} |"
        )
    lines += [
        "",
        "## Mapping",
        "",
        "- B0=e0(256), B1=e1(128), B2=e2(64), B3=e3(32), B4=e4(16)",
        "- S3–S5: B0..B3 on, **B4 off only**",
        "- S2: reused from `archive/skiponly_covid_base32` (full additive)",
        "- S4: zero-init 1×1 + learnable scale (no Attention)",
        "- S5: concat(decoder, skip) → 3×3 Conv+BN+LeakyReLU (no Attention, no learnable γ)",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    safe_print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=list(STAGES), help="run one stage")
    ap.add_argument("--all", action="store_true", help="run stages 1→5 sequentially")
    ap.add_argument("--from-stage", type=int, default=1, help="with --all, start from this stage")
    args = ap.parse_args()
    if not args.stage and not args.all:
        ap.error("pass --stage N or --all")

    seed_everything(SEED)
    safe_print("device:", DEVICE)
    if DEVICE.type == "cuda":
        safe_print("GPU:", torch.cuda.get_device_name(0))

    seg_data = datasets.covid(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    safe_print("SEG(covid):", len(seg_data.train), len(seg_data.val), len(seg_data.test))

    for mode in (SKIP_NONE, SKIP_FULL, SKIP_B0B3, SKIP_B0B3_LEARN, SKIP_B0B3_CONCAT):
        m = VAE(BASE, DROPOUT, LATENT_CH, skip_mode=mode).to(DEVICE).eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 256, 256, device=DEVICE)
            out, _, _ = m(x, sample=False)
        assert out.shape == (1, 1, 256, 256), mode
        del m, x, out
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    safe_print("smoke OK for all 5 skip modes")

    bt_ckpt = ensure_shared_bt()
    safe_print("BT encoder:", bt_ckpt)

    stages = list(range(args.from_stage, 6)) if args.all else [args.stage]
    reports = []
    for s in range(1, stages[0]):
        rp = ROOT / "results" / "ablation" / STAGES[s]["name"] / "report.json"
        if rp.exists():
            reports.append(json.loads(rp.read_text(encoding="utf-8")))

    for s in stages:
        meta = STAGES[s]
        reuse = meta.get("reuse_from")
        if reuse:
            reports.append(reuse_stage_report(s, ROOT / reuse))
        else:
            reports.append(
                train_stage(
                    s,
                    seg_data.train_loader,
                    seg_data.val_loader,
                    seg_data.test_loader,
                    bt_ckpt,
                )
            )
        write_summary(reports)

    safe_print("ALL REQUESTED STAGES DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("Stopped by user - rerun to resume.")
    except Exception:
        traceback.print_exc()
        raise
