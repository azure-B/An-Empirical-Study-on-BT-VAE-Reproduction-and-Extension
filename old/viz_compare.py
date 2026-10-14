"""
원본 / GT / Pred / Overlay 비교 이미지 저장.

사용:
  .\\.venv\\Scripts\\python.exe viz_compare.py
  .\\.venv\\Scripts\\python.exe viz_compare.py --n 12 --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_load_BT import datasets
from models1 import VAE

ROOT = Path(__file__).resolve().parent
SEG_BEST = ROOT / "results" / "ms_gsf_bt_vae" / "seg_best.pt"
OUT_DIR = ROOT / "results" / "ms_gsf_bt_vae" / "compare"

BASE, LATENT_CH, DROPOUT = 16, 128, 0.05


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
    union = ps + gs - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


def make_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """RGB: green=GT only, red=Pred only, yellow=overlap."""
    g = img.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    rgb = np.stack([g, g, g], axis=-1)
    gt_m = gt >= 0.5
    pr_m = pred >= thr
    both = gt_m & pr_m
    only_gt = gt_m & ~pr_m
    only_pr = pr_m & ~gt_m
    rgb[only_gt] = [0.15, 0.85, 0.25]
    rgb[only_pr] = [0.95, 0.20, 0.20]
    rgb[both] = [0.95, 0.90, 0.15]
    return np.clip(rgb, 0, 1)


def load_model(device: torch.device) -> VAE:
    if not SEG_BEST.exists():
        raise FileNotFoundError(f"checkpoint not found: {SEG_BEST}")
    model = VAE(BASE, DROPOUT, LATENT_CH).to(device)
    try:
        state = torch.load(SEG_BEST, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(SEG_BEST, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {SEG_BEST.name} (epoch={state.get('epoch')}, best={state.get('best')})")
    return model


@torch.no_grad()
def collect_candidates(model, loader, device, thr: float, min_gt_px: int, scan_limit: int):
    """GT 마스크가 있는 샘플을 스캔해 후보 수집."""
    rows = []
    seen = 0
    for batch in loader:
        x = as_bchw(batch[0]).to(device)
        y = as_bchw(batch[1]).to(device)
        logits, *_ = model(x, sample=False, return_aux=True)
        prob = torch.sigmoid(logits)
        for i in range(x.size(0)):
            gt = to_np2d(y[i])
            gt_px = int((gt >= 0.5).sum())
            seen += 1
            if gt_px < min_gt_px:
                continue
            img = to_np2d(x[i])
            pr = to_np2d(prob[i])
            d, iou = dice_iou(pr, gt, thr)
            rows.append(
                {
                    "img": img,
                    "gt": gt,
                    "pred": pr,
                    "dice": d,
                    "iou": iou,
                    "gt_px": gt_px,
                }
            )
            if seen >= scan_limit:
                return rows
        if seen >= scan_limit:
            break
    return rows


def pick_diverse(rows: list[dict], n: int) -> list[dict]:
    """Dice 상위/중간/하위가 섞이도록 고름."""
    if len(rows) <= n:
        return sorted(rows, key=lambda r: r["dice"], reverse=True)
    ranked = sorted(rows, key=lambda r: r["dice"], reverse=True)
    # 균등 간격으로 n개
    idxs = np.linspace(0, len(ranked) - 1, n, dtype=int)
    picked, used = [], set()
    for i in idxs:
        if i not in used:
            picked.append(ranked[i])
            used.add(i)
    return picked


def save_single(row: dict, path: Path, thr: float, idx: int) -> None:
    img, gt, pred = row["img"], row["gt"], row["pred"]
    overlay = make_overlay(img, gt, pred, thr)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    titles = [
        "Image",
        "GT",
        "Pred prob",
        f"Pred ≥ {thr}",
        "Overlay\n(G=GT R=Pred Y=both)",
    ]
    panels = [img, gt, pred, pred >= thr, overlay]
    cmaps = ["gray", "gray", "magma", "gray", None]
    for ax, title, panel, cmap in zip(axes, titles, panels, cmaps):
        if cmap == "magma":
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        elif cmap:
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"sample {idx:02d}  |  Dice={row['dice']:.3f}  IoU={row['iou']:.3f}  GT_px={row.get('gt_px', '?')}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_grid(rows: list[dict], path: Path, thr: float) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(14, 2.4 * n))
    if n == 1:
        axes = np.array([axes])
    headers = ["Image", "GT", "Pred prob", f"Pred≥{thr}", "Overlay"]
    for r, row in enumerate(rows):
        panels = [
            row["img"],
            row["gt"],
            row["pred"],
            row["pred"] >= thr,
            make_overlay(row["img"], row["gt"], row["pred"], thr),
        ]
        cmaps = ["gray", "gray", "magma", "gray", None]
        for c, (panel, header, cmap) in enumerate(zip(panels, headers, cmaps)):
            ax = axes[r, c]
            if cmap == "magma":
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            elif cmap:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            if r == 0:
                ax.set_title(header, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"#{r:02d}\nD={row['dice']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("BT-VAE compare  |  green=GT  red=Pred  yellow=both", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of samples to save")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--min-gt-px", type=int, default=200, help="skip nearly-empty GT")
    ap.add_argument("--scan", type=int, default=200, help="how many test images to scan")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = datasets.covid_90p(batch_size=args.batch, num_workers=0)
    loader = data.test_loader if args.split == "test" else data.val_loader
    model = load_model(device)
    cands = collect_candidates(
        model, loader, device, args.thr, args.min_gt_px, args.scan
    )
    if not cands:
        raise RuntimeError("no samples with enough GT pixels — lower --min-gt-px")

    rows = pick_diverse(cands, args.n)
    print(f"scanned candidates with GT>={args.min_gt_px}px: {len(cands)} → show {len(rows)}")

    grid_path = OUT_DIR / f"compare_grid_{args.split}.png"
    save_grid(rows, grid_path, args.thr)
    print(f"grid -> {grid_path}")

    for i, row in enumerate(rows):
        p = OUT_DIR / f"compare_{args.split}_{i:02d}_d{row['dice']:.2f}.png"
        save_single(row, p, args.thr, i)
        print(f"  {p.name}")

    ranked = sorted(cands, key=lambda r: r["dice"], reverse=True)
    save_single(ranked[0], OUT_DIR / f"best_{args.split}.png", args.thr, 0)
    save_single(ranked[-1], OUT_DIR / f"worst_{args.split}.png", args.thr, 0)
    print(f"best dice={ranked[0]['dice']:.3f}, worst dice={ranked[-1]['dice']:.3f}")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
