"""SkipOnly result compare: Image | GT | Pred | Overlay + Dice."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_load_BT import datasets
from models1_skip import VAE

ROOT = Path(__file__).resolve().parent
SEG_BEST = ROOT / "results" / "skiponly_covid_attn" / "seg_best.pt"
OUT_DIR = ROOT / "results" / "skiponly_covid_attn" / "compare"

BASE, LATENT_CH, DROPOUT = 32, 1024, 0.1


def as_bchw(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def to_np2d(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    return a


def dice_iou(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> tuple[float, float]:
    p = (pred >= thr).astype(np.float32)
    g = (gt >= 0.5).astype(np.float32)
    inter = (p * g).sum()
    ps, gs = p.sum(), g.sum()
    dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
    iou = (inter + 1e-6) / (ps + gs - inter + 1e-6)
    return float(dice), float(iou)


def make_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> np.ndarray:
    g = img.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    rgb = np.stack([g, g, g], axis=-1)
    gt_m = gt >= 0.5
    pr_m = pred >= thr
    both = gt_m & pr_m
    rgb[gt_m & ~pr_m] = [0.15, 0.85, 0.25]
    rgb[pr_m & ~gt_m] = [0.95, 0.20, 0.20]
    rgb[both] = [0.95, 0.90, 0.15]
    return np.clip(rgb, 0, 1)


def load_model(device: torch.device) -> VAE:
    if not SEG_BEST.exists():
        raise FileNotFoundError(SEG_BEST)
    model = VAE(BASE, DROPOUT, LATENT_CH).to(device)
    try:
        state = torch.load(SEG_BEST, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(SEG_BEST, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(
        f"loaded {SEG_BEST.name} epoch={state.get('epoch')} "
        f"best_dice={state.get('best_dice', state.get('best'))}"
    )
    return model


@torch.no_grad()
def collect(model, loader, device, thr, min_gt_px, scan_limit):
    rows, seen = [], 0
    for batch in loader:
        x = as_bchw(batch[0]).to(device)
        y = as_bchw(batch[1]).to(device)
        logits, _, _ = model(x, sample=False)
        prob = torch.sigmoid(logits)
        for i in range(x.size(0)):
            gt = to_np2d(y[i])
            gt_px = int((gt >= 0.5).sum())
            seen += 1
            if gt_px < min_gt_px:
                continue
            img, pr = to_np2d(x[i]), to_np2d(prob[i])
            d, iou = dice_iou(pr, gt, thr)
            rows.append({"img": img, "gt": gt, "pred": pr, "dice": d, "iou": iou, "gt_px": gt_px})
            if seen >= scan_limit:
                return rows
        if seen >= scan_limit:
            break
    return rows


def pick_diverse(rows, n):
    if len(rows) <= n:
        return sorted(rows, key=lambda r: r["dice"], reverse=True)
    ranked = sorted(rows, key=lambda r: r["dice"], reverse=True)
    idxs = np.linspace(0, len(ranked) - 1, n, dtype=int)
    out, used = [], set()
    for i in idxs:
        if i not in used:
            out.append(ranked[i])
            used.add(i)
    return out


def save_single(row, path, thr, idx):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    panels = [row["img"], row["gt"], row["pred"] >= thr, make_overlay(row["img"], row["gt"], row["pred"], thr)]
    titles = ["Image", "GT", "Pred (result)", "Overlay\n(G=GT R=Pred Y=both)"]
    for ax, title, panel, cmap in zip(axes, titles, panels, ["gray", "gray", "gray", None]):
        if cmap:
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"#{idx:02d}  Dice={row['dice']:.3f}  IoU={row['iou']:.3f}  GT_px={row['gt_px']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_grid(rows, path, thr):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.5 * n))
    if n == 1:
        axes = np.array([axes])
    headers = ["Image", "GT", "Pred", "Overlay"]
    for r, row in enumerate(rows):
        panels = [
            row["img"],
            row["gt"],
            row["pred"] >= thr,
            make_overlay(row["img"], row["gt"], row["pred"], thr),
        ]
        for c, (panel, header) in enumerate(zip(panels, headers)):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            if r == 0:
                ax.set_title(header)
            if c == 0:
                ax.set_ylabel(f"#{r:02d}\nD={row['dice']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("SkipOnly base32 lat1024  |  green=GT  red=Pred  yellow=both", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-gt-px", type=int, default=200)
    ap.add_argument("--scan", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.covid(batch_size=4, num_workers=0)
    loader = data.test_loader if args.split == "test" else data.val_loader
    model = load_model(device)
    cands = collect(model, loader, device, args.thr, args.min_gt_px, args.scan)
    if not cands:
        raise RuntimeError("no samples with GT")
    rows = pick_diverse(cands, args.n)
    dices = [r["dice"] for r in cands]
    print(
        f"candidates={len(cands)} show={len(rows)} "
        f"mean_dice={np.mean(dices):.3f} median={np.median(dices):.3f} "
        f"best={max(dices):.3f} worst={min(dices):.3f}"
    )

    grid = OUT_DIR / f"compare_grid_{args.split}.png"
    save_grid(rows, grid, args.thr)
    print("grid ->", grid)
    for i, row in enumerate(rows):
        p = OUT_DIR / f"compare_{args.split}_{i:02d}_d{row['dice']:.2f}.png"
        save_single(row, p, args.thr, i)
        print(" ", p.name)

    ranked = sorted(cands, key=lambda r: r["dice"], reverse=True)
    save_single(ranked[0], OUT_DIR / f"best_{args.split}.png", args.thr, 0)
    save_single(ranked[-1], OUT_DIR / f"worst_{args.split}.png", args.thr, 0)
    print(f"best/worst dice={ranked[0]['dice']:.3f}/{ranked[-1]['dice']:.3f}")
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
