"""Paper-style qualitative cases, saved whenever a ratio finishes testing.

Fig. 3 — one COVID test slice across label ratios.
  green = prediction ∩ GT, red = false positive, white = missed GT.

Fig. 5 — four BP test cases at the 30% split (bp_train_30).
  Same colors. Columns are the models we actually test, not the paper baselines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from pipeline import config as C
from pipeline.engine import OrigVAE, safe_load
from pipeline.models_presence import LesionPresenceHead
from pipeline.models_skip import VAE as SkipVAE

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = C.RESULT_DIR / "label_ratio" / "cases"
BP_CASE_ROOT = ROOT / "results" / "bp"

# Fixed test filenames. COVID matches the compact single-lesion slice in Fig. 3.
# BP cases are four different patients on the 30% test split, Fig. 5 layout.
COVID_CASE = "3_116.png"
BP_CASES = (
    ("Case1", "16_23.png"),
    ("Case2", "2_12.png"),
    ("Case3", "38_2.png"),
    ("Case4", "34_21.png"),
)


def _gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L").resize((256, 256), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L").resize((256, 256), Image.NEAREST)
        return np.asarray(img, dtype=np.uint8) > 0


def overlay_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    rgb[gt & pred] = (0, 200, 0)
    rgb[gt & ~pred] = (255, 255, 255)
    rgb[~gt & pred] = (220, 40, 40)
    return rgb


def _rgb_gray(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _gt_rgb(gt: np.ndarray) -> np.ndarray:
    rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    rgb[gt.astype(bool)] = (255, 255, 255)
    return rgb


@torch.inference_mode()
def predict_mask(model, x: torch.Tensor, kind: str, head=None, tau: float | None = None) -> np.ndarray:
    if kind == "s5":
        mu, _lv, _z, skips = model.encoder.forward_features(x)
        logits = model.decoder(mu, skips)
    else:
        mu, _lv, _z = model.encoder(x)
        logits = model.decoder(mu)
    pred = torch.sigmoid(logits) > 0.5
    if head is not None and tau is not None:
        p = torch.sigmoid(head(mu))
        if float(p.flatten()[0]) < tau:
            pred = torch.zeros_like(pred)
    return pred[0, 0].detach().cpu().numpy()


def _load_seg(kind: str, ckpt: Path, device: str):
    model = (SkipVAE if kind == "s5" else OrigVAE)(base=32).to(device)
    state = safe_load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _tensor(gray: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return t.view(1, 1, 256, 256).to(device)


def save_covid_case(kind: str, ratio: float, model, head=None, tau: float | None = None, device: str | None = None):
    """One Fig. 3 cell. kind is s5, s5_cls, or none."""
    if device is None:
        device = next(model.parameters()).device.type
    ds_root = ROOT / "datasets" / "covid_8_1_1" / "test"
    gray = _gray(ds_root / "original" / COVID_CASE)
    gt = _mask(ds_root / "label" / COVID_CASE)
    pred = predict_mask(model, _tensor(gray, device), "s5" if kind.startswith("s5") else "none", head, tau)
    out = CASE_ROOT / "covid"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb_gray(gray)).save(out / "input.png")
    Image.fromarray(_gt_rgb(gt)).save(out / "gt.png")
    tag = f"{kind}_r{int(round(ratio * 100)):02d}.png"
    Image.fromarray(overlay_rgb(pred, gt)).save(out / tag)
    _write_covid_panel(out)
    return out / tag


def _write_covid_panel(out: Path):
    ratios = (10, 30, 50, 70, 90)
    rows = ("s5", "s5_cls", "none")
    cell = 128
    pad = 4
    label_w, label_h = 72, 22
    cols = 2 + len(ratios)  # input, gt, ratios
    w = label_w + cols * (cell + pad)
    h = label_h + len(rows) * (cell + pad)
    panel = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(panel)
    headers = ["Input", "GT", "10%", "30%", "50%", "70%", "90%"]
    for i, text in enumerate(headers):
        draw.text((label_w + i * (cell + pad) + 8, 4), text, fill=(255, 220, 0))
    row_name = {"s5": "S5", "s5_cls": "S5+cls", "none": "none"}
    for r, mode in enumerate(rows):
        y = label_h + r * (cell + pad)
        draw.text((4, y + 8), row_name[mode], fill=(255, 255, 255))
        for c, name in enumerate(("input.png", "gt.png")):
            path = out / name
            if path.exists():
                im = Image.open(path).convert("RGB").resize((cell, cell))
                panel.paste(im, (label_w + c * (cell + pad), y))
        for c, pct in enumerate(ratios):
            path = out / f"{mode}_r{pct:02d}.png"
            if path.exists():
                im = Image.open(path).convert("RGB").resize((cell, cell))
                panel.paste(im, (label_w + (2 + c) * (cell + pad), y))
    panel.save(out / "panel.png")
    (out / "case.json").write_text(json.dumps({
        "figure": "paper Fig. 3 style",
        "file": COVID_CASE,
        "colors": "green=TP, red=FP, white=FN",
    }, indent=2), encoding="utf-8")


def save_bp_cases(device: str = "cpu"):
    """Fig. 5 layout on the four fixed BP test slices. none and S5 only."""
    ds = ROOT / "datasets" / "bp_data" / "train_30_split" / "test"
    none_ckpt = ROOT / "old" / "parked" / "results" / "bp_original" / "none" / "seg_best.pt"
    s5_ckpt = ROOT / "old" / "parked" / "results" / "bp_original" / "s5_concat" / "seg_best.pt"
    none = _load_seg("none", none_ckpt, device)
    s5 = _load_seg("s5", s5_ckpt, device)
    out = BP_CASE_ROOT
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for title, name in BP_CASES:
        gray = _gray(ds / "original" / name)
        gt = _mask(ds / "label" / name)
        x = _tensor(gray, device)
        preds = {
            "input": _rgb_gray(gray),
            "gt": _gt_rgb(gt),
            "none": overlay_rgb(predict_mask(none, x, "none"), gt),
            "s5": overlay_rgb(predict_mask(s5, x, "s5"), gt),
        }
        for key, rgb in preds.items():
            Image.fromarray(rgb).save(out / f"{title}_{key}.png")
        cells.append((title, preds))
    _write_bp_panel(out, cells)
    del none, s5
    return out / "panel.png"


def _write_bp_panel(out: Path, cells: list):
    cell = 160
    pad = 4
    label_w, label_h = 64, 22
    cols = ("input", "gt", "none", "s5")
    headers = ("Input", "GT", "none", "S5")
    w = label_w + len(cols) * (cell + pad)
    h = label_h + len(cells) * (cell + pad)
    panel = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(panel)
    for i, text in enumerate(headers):
        draw.text((label_w + i * (cell + pad) + 8, 4), text, fill=(255, 220, 0))
    for r, (title, preds) in enumerate(cells):
        y = label_h + r * (cell + pad)
        draw.text((4, y + 8), title, fill=(255, 220, 0))
        for c, key in enumerate(cols):
            im = Image.fromarray(preds[key]).resize((cell, cell))
            panel.paste(im, (label_w + c * (cell + pad), y))
    panel.save(out / "panel.png")
    (out / "case.json").write_text(json.dumps({
        "figure": "paper Fig. 5 style",
        "split": "bp_train_30 test",
        "cases": [{"name": a, "file": b} for a, b in BP_CASES],
        "colors": "green=TP, red=FP, white=FN",
    }, indent=2), encoding="utf-8")


def backfill_covid(device: str = "cpu"):
    """Render Fig. 3 cells for ratios that already have a checkpoint."""
    root = C.RESULT_DIR / "label_ratio"
    for pct in (10, 30, 50, 70, 90):
        ratio = pct / 100.0
        s5_dir = root / f"r{pct:02d}" / "s5"
        s5_ckpt = s5_dir / "seg_best.pt"
        if (s5_dir / "report_seg.json").exists() or (s5_dir / "report.json").exists():
            if s5_ckpt.exists():
                model = _load_seg("s5", s5_ckpt, device)
                save_covid_case("s5", ratio, model, device=device)
                del model
        cls_dir = root / f"r{pct:02d}" / "s5_cls"
        report = cls_dir / "report.json"
        head_ckpt = cls_dir / "head_best.pt"
        if report.exists() and head_ckpt.exists() and s5_ckpt.exists():
            meta = json.loads(report.read_text(encoding="utf-8"))
            model = _load_seg("s5", s5_ckpt, device)
            head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(device)
            head.load_state_dict(safe_load(head_ckpt, map_location="cpu")["head"])
            head.eval()
            save_covid_case("s5_cls", ratio, model, head, float(meta["tau"]), device=device)
            del model, head
        none_dir = root / f"r{pct:02d}" / "none"
        none_ckpt = none_dir / "seg_best.pt"
        if (none_dir / "report.json").exists() and none_ckpt.exists():
            model = _load_seg("none", none_ckpt, device)
            save_covid_case("none", ratio, model, device=device)
            del model
    return CASE_ROOT / "covid" / "panel.png"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--which", default="both", choices=("covid", "bp", "both"))
    args = p.parse_args()
    if args.which in ("covid", "both"):
        print("covid panel", backfill_covid(args.device))
    if args.which in ("bp", "both"):
        print("bp panel", save_bp_cases(args.device))
