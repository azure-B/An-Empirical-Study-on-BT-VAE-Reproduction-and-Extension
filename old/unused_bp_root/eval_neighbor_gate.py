"""Post-hoc adjacent-slice gate on frozen none. No training. original/ untouched."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_original_bp import (
    BASE,
    DEVICE,
    OrigVAE,
    RESULT_DIR,
    safe_load,
    safe_print,
    split_dice_stats,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "bp_data"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT = RESULT_DIR / "none_neighbor"
IMG_SIZE = (256, 256)
BATCH = 16
THR = 0.5


def parse_stem(stem: str) -> tuple[str, int]:
    pid, sl = stem.rsplit("_", 1)
    return pid, int(sl)


def index_originals() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in DATA.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".png" and p.parent.name == "original":
            idx[p.stem] = p
    return idx


def list_split(split: str) -> list[str]:
    d = DATA / "train_30_split" / split / "original"
    return sorted(p.stem for p in d.glob("*.png"))


def load_gray(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L").resize(IMG_SIZE, Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    return t / 255.0


def load_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L").resize(IMG_SIZE, Image.NEAREST)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    return (t > 0).float()


def label_path(orig: Path) -> Path:
    return orig.parent.parent / "label" / orig.name


@torch.no_grad()
def infer_pred_sums(model, idx: dict[str, Path], stems: list[str]) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    buf, names = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf, 0).unsqueeze(1).to(DEVICE)
        logits, _, _ = model(x)
        pred = (torch.sigmoid(logits) > THR).float()
        sums = pred.sum(dim=(1, 2, 3)).cpu().numpy()
        for n, s in zip(names, sums):
            out[n] = float(s)
        buf.clear()
        names.clear()

    for stem in stems:
        buf.append(load_gray(idx[stem]))
        names.append(stem)
        if len(buf) >= BATCH:
            flush()
    flush()
    return out


@torch.no_grad()
def infer_official(model, idx: dict[str, Path], stems: list[str]):
    model.eval()
    preds, gts, names = [], [], []
    b_img, b_gt, b_name = [], [], []

    def flush():
        if not b_img:
            return
        x = torch.stack(b_img, 0).unsqueeze(1).to(DEVICE)
        logits, _, _ = model(x)
        pred = (torch.sigmoid(logits) > THR).float().cpu()
        preds.append(pred[:, 0])
        gts.append(torch.stack(b_gt, 0))
        names.extend(b_name)
        b_img.clear()
        b_gt.clear()
        b_name.clear()

    for stem in stems:
        orig = idx[stem]
        b_img.append(load_gray(orig))
        b_gt.append(load_mask(label_path(orig)))
        b_name.append(stem)
        if len(b_img) >= BATCH:
            flush()
    flush()
    return names, torch.cat(preds, 0).numpy(), torch.cat(gts, 0).numpy()


def neighbor_stems(stem: str, idx: dict[str, Path]) -> tuple[str | None, str | None]:
    pid, sl = parse_stem(stem)
    left = f"{pid}_{sl - 1}"
    right = f"{pid}_{sl + 1}"
    return (left if left in idx else None, right if right in idx else None)


def pred_pos(pred_sums: dict[str, float], stem: str | None) -> bool:
    if stem is None:
        return False
    return pred_sums.get(stem, 0.0) > 0


def apply_gate(names, pred, gt, pred_sums, idx, rule: str, min_area: float = 0.0):
    gated = pred.copy()
    isolated = np.zeros(len(names), dtype=bool)
    for i, stem in enumerate(names):
        left, right = neighbor_stems(stem, idx)
        both = left is not None and right is not None
        lpos, rpos = pred_pos(pred_sums, left), pred_pos(pred_sums, right)
        cpos = pred[i].sum() > 0
        isolated[i] = cpos and (not lpos) and (not rpos) and both
        kill = False
        if rule == "baseline":
            kill = False
        elif rule == "isolated_kill":
            kill = isolated[i]
        elif rule == "isolated_or_single":
            known = [x for x in (lpos, rpos) if left is not None or right is not None]
            # kill if center on and every existing neighbor is off
            neigh = []
            if left is not None:
                neigh.append(lpos)
            if right is not None:
                neigh.append(rpos)
            kill = cpos and len(neigh) > 0 and (not any(neigh))
        elif rule == "majority_2of3":
            votes = int(cpos) + int(lpos) + int(rpos)
            kill = cpos and votes < 2
        elif rule == "isolated_small":
            kill = isolated[i] and pred[i].sum() < min_area
        if kill:
            gated[i] = 0
    return gated, isolated


def stats_of(pred, gt):
    pred_s = pred.reshape(pred.shape[0], -1).sum(axis=1)
    gt_s = gt.reshape(gt.shape[0], -1).sum(axis=1)
    inter = (pred * gt).reshape(pred.shape[0], -1).sum(axis=1)
    union = pred_s + gt_s
    dices = (2.0 * inter + 1e-6) / (union + 1e-6)
    return split_dice_stats(dices, gt_s, pred_s)


def diagnose(names, pred, gt, isolated):
    pred_s = pred.reshape(len(names), -1).sum(axis=1)
    gt_s = gt.reshape(len(names), -1).sum(axis=1)
    empty = gt_s == 0
    nonempty = ~empty
    empty_fp = empty & (pred_s > 0)
    ne_hit = nonempty & (pred_s > 0)
    return {
        "n_empty_fp": int(empty_fp.sum()),
        "empty_fp_isolated": float(isolated[empty_fp].mean()) if empty_fp.any() else None,
        "n_nonempty_hit": int(ne_hit.sum()),
        "nonempty_hit_isolated": float(isolated[ne_hit].mean()) if ne_hit.any() else None,
        "n_isolated": int(isolated.sum()),
    }


def eval_split(model, idx, split: str):
    official = list_split(split)
    need = set(official)
    for stem in official:
        l, r = neighbor_stems(stem, idx)
        if l:
            need.add(l)
        if r:
            need.add(r)
    safe_print(f"{split}: official={len(official)} infer={len(need)}")
    pred_sums = infer_pred_sums(model, idx, sorted(need))
    names, pred, gt = infer_official(model, idx, official)
    _, isolated = apply_gate(names, pred, gt, pred_sums, idx, "isolated_kill")
    diag = diagnose(names, pred, gt, isolated)
    rules = {}
    for rule, kwargs in [
        ("baseline", {}),
        ("isolated_kill", {}),
        ("isolated_or_single", {}),
        ("majority_2of3", {}),
        ("isolated_small_200", {"rule": "isolated_small", "min_area": 200}),
        ("isolated_small_800", {"rule": "isolated_small", "min_area": 800}),
    ]:
        use_rule = kwargs.pop("rule", rule)
        gated, _ = apply_gate(names, pred, gt, pred_sums, idx, use_rule, **kwargs)
        rules[rule] = stats_of(gated, gt)
        safe_print(
            f"  {split} {rule}: dice={rules[rule]['dice_mean']:.4f} "
            f"nonempty={rules[rule]['mean_dice_nonempty']:.4f} "
            f"empty_fp={rules[rule]['empty_fp_rate']:.3f}"
        )
    return {"diag": diag, "rules": rules, "n": len(official)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = index_originals()
    safe_print("indexed originals", len(idx), "device", DEVICE)
    vae = OrigVAE(base=BASE).to(DEVICE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    vae.eval()
    del st
    report = {
        "none_ckpt": str(NONE_BEST),
        "note": "gate uses anatomical ±1 neighbors from full bp_data dump; metrics on train_30 val/test only",
    }
    for split in ("val", "test"):
        report[split] = eval_split(vae, idx, split)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
