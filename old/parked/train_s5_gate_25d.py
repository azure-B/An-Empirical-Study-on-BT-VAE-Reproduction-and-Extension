"""Frozen S5 SEG + 2.5D presence classifier gate (paper cls-only uplift).

- Train ResNet18 on (z-1,z,z+1) FLAIR stack; τ not in loss.
- Freeze S5 decoder; pick VAL τ that dominates none (ne↑ & fp≤) if possible.
- Soft gate (mask * p) reported as side metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_presence_25d import Presence25D
from train_lesion_presence import PIXEL_THR, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OUT = RESULT_DIR / "s5_gate_25d"
BATCH = 16
EPOCHS = 60
LR = 1e-4
POS_WEIGHT = 2.0
ES = 12
IMG_SIZE = (256, 256)
NONE_NE = 0.572
NONE_FP = 0.383


def parse_stem(stem: str) -> tuple[str, int]:
    pid, sl = stem.rsplit("_", 1)
    return pid, int(sl)


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


class Stack25D(Dataset):
    """Center-slice label; input = stack(prev, center, next). Missing neighbor → replicate."""

    def __init__(self, base_ds):
        self.base = base_ds
        self.names = list(base_ds.filenames)
        self.stem_to_name = {Path(n).stem: n for n in self.names}
        self.stems = [Path(n).stem for n in self.names]

    def __len__(self):
        return len(self.names)

    def _neighbor(self, stem: str, delta: int) -> Path:
        pid, sl = parse_stem(stem)
        alt = f"{pid}_{sl + delta}"
        name = self.stem_to_name.get(alt, self.stem_to_name[stem])
        return self.base.path_original / name

    def __getitem__(self, idx):
        n = len(self.names)
        last_err = None
        for offset in range(n):
            j = (idx + offset) % n
            name = self.names[j]
            stem = Path(name).stem
            try:
                c = load_gray(self.base.path_original / name)
                p = load_gray(self._neighbor(stem, -1))
                nxt = load_gray(self._neighbor(stem, +1))
                lab = load_mask(self.base.path_label / name)
            except (OSError, ValueError, SyntaxError) as e:
                last_err = e
                continue
            x = torch.stack([p, c, nxt], dim=0)  # [3,H,W]
            return x, lab, stem
        raise OSError(f"all images failed near idx={idx}: {last_err}")


def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def freeze_seg(vae: nn.Module) -> None:
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False


@torch.inference_mode()
def collect_cls(model, loader):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        x = batch[0].to(DEVICE)
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        logit = model(x)
        ys.append(presence_y(seg).cpu().numpy())
        ss.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


@torch.inference_mode()
def collect_gated(cls_model, seg_model, loader):
    """Per-slice: gt mass, S5 pred mass, raw dice, cls score."""
    cls_model.eval()
    seg_model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        x3 = batch[0].to(DEVICE)          # [B,3,H,W]
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        center = x3[:, 1:2]               # [B,1,H,W] for S5
        score = torch.sigmoid(cls_model(x3))
        logits, _, _ = seg_model(center)
        pred = (torch.sigmoid(logits) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(score.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def eval_hard(bag, thr: float) -> dict:
    keep = bag["score"] >= thr
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    tp, fp, fn, tn = cm["TT"], cm["FT"], cm["TF"], cm["FF"]
    prec, sens, spec = cm["precision"], cm["sensitivity"], cm["specificity"]
    f1 = 2 * prec * sens / max(1e-8, prec + sens)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "thr": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auroc": binary_auroc(yt, bag["score"]),
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "seg_sensitivity": float(m["sensitivity"]),
    }


def eval_soft(bag) -> dict:
    """Soft gate: effective keep mass = mass * score; dice reweighted approx via blank if p<0.5? 
    Use continuous: keep_mass = mass * score, dice scaled only for reporting via hard soft-thr0.
    Better: blank with keep = ones, but scale pred_mass by score for empty_fp (present if mass*p>0).
    """
    score = bag["score"]
    mass = bag["mass"] * score
    # soft dice: if we blank fractionally, nonempty dice ≈ raw dice when score~1;
    # approximate gated dice as dice * score for nonempty GT, and empty-empty=1 when mass*score==0
    gt = bag["gt"]
    dice = bag["dice"].copy()
    # for empty GT: if soft mass>0 → dice 0 else 1
    empty = gt == 0
    dice[empty] = np.where(mass[empty] > 0, 0.0, 1.0)
    # for nonempty: keep dice only if soft mass>0 else 0
    ne = ~empty
    dice[ne] = np.where(mass[ne] > 0, bag["dice"][ne] * score[ne], 0.0)
    keep = mass > 0  # any soft residue counts as present
    m = metrics_bundle(gt, mass, dice, keep)
    yt = gt > 0
    cm = conf_matrix(yt, keep)
    return {
        "thr": None,
        "mode": "soft_mask_times_p",
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "sensitivity": float(cm["sensitivity"]),
        "specificity": float(cm["specificity"]),
        "auroc": binary_auroc(yt, score),
        **{k: cm[k] for k in ("TT", "FT", "TF", "FF")},
    }


def pick_paper_tau(val_bag, none_ne=NONE_NE, none_fp=NONE_FP):
    """Prefer VAL points that dominate none; else max ne - 0.5*fp."""
    dominate, fallback = None, None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_hard(val_bag, thr)
        joint = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
        m["_joint"] = joint
        if fallback is None or joint > fallback["_joint"]:
            fallback = dict(m)
        if m["nonempty_dice"] + 1e-9 >= none_ne and m["empty_fp_rate"] <= none_fp + 1e-9:
            key = (m["nonempty_dice"], -m["empty_fp_rate"])
            if dominate is None or key > dominate["_key"]:
                dominate = dict(m)
                dominate["_key"] = key
                dominate["pick"] = "dominate_none"
    if dominate is not None:
        return dominate
    fallback["pick"] = "joint_ne_minus_0.5fp"
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--pos-weight", type=float, default=POS_WEIGHT)
    ap.add_argument("--seg-ckpt", type=str, default=str(S5_BEST))
    args = ap.parse_args()

    seg_ckpt = Path(args.seg_ckpt)
    if not seg_ckpt.exists():
        raise FileNotFoundError(seg_ckpt)

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False,
    )
    tr_ds = Stack25D(data.train)
    va_ds = Stack25D(data.val)
    te_ds = Stack25D(data.test)
    tr_loader = make_loader(tr_ds, args.batch, shuffle=True)
    va_loader = make_loader(va_ds, args.batch, shuffle=False)
    te_loader = make_loader(te_ds, args.batch, shuffle=False)

    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} "
        f"pos_w={args.pos_weight} seg={seg_ckpt.name}"
    )
    safe_print(
        "design: freeze S5 | train Presence25D(z-1,z,z+1) NO-τ-in-loss | "
        "τ by VAL paper-Pareto vs none"
    )

    cls = Presence25D().to(DEVICE)
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=DEVICE)
    )
    opt = torch.optim.AdamW(cls.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    best_path = ckpt_dir / "cls25d_best_auroc.pt"
    last_path = ckpt_dir / "cls25d_last.pt"
    history = []
    best_auroc = -1.0
    es = 0

    for ep in range(1, args.epochs + 1):
        cls.train()
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in tr_loader:
            x = batch[0].to(DEVICE)
            seg = batch[1].to(DEVICE)
            if seg.ndim == 3:
                seg = seg.unsqueeze(1)
            y = presence_y(seg)
            opt.zero_grad(set_to_none=True)
            loss = bce(cls(x), y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        yv, sv = collect_cls(cls, va_loader)
        auroc = binary_auroc(yv, sv)
        pred = sv >= 0.5
        cm = conf_matrix(yv > 0.5, pred)
        rec = {
            "epoch": ep,
            "train_bce": tr / max(1, ntr),
            "val_auroc": auroc,
            "val_acc@0.5": (cm["TT"] + cm["FF"]) / max(1, len(yv)),
            "val_sens@0.5": cm["sensitivity"],
            "val_spec@0.5": cm["specificity"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[cls25d {ep}/{args.epochs}] bce={rec['train_bce']:.4f} "
            f"auroc={auroc:.4f} sens={cm['sensitivity']:.3f} "
            f"spec={cm['specificity']:.3f} ({rec['sec']:.1f}s)"
        )
        sch.step(auroc)
        payload = {
            "model": cls.state_dict(),
            "epoch": ep,
            "val_auroc": auroc,
            "design": "s5_gate_25d",
            "pos_weight": args.pos_weight,
            "seg_ckpt": str(seg_ckpt),
        }
        atomic_torch_save(payload, last_path)
        if auroc > best_auroc + 1e-4:
            best_auroc = auroc
            atomic_torch_save(payload, best_path)
            es = 0
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= ES:
                safe_print(f"early stop @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    st = safe_load(best_path, map_location="cpu")
    cls.load_state_dict(st["model"])
    cls.eval()

    seg_model = SkipVAE(base=BASE).to(DEVICE)
    sst = safe_load(seg_ckpt, map_location="cpu")
    seg_model.load_state_dict(sst["model"])
    freeze_seg(seg_model)
    safe_print(f"loaded frozen S5 from {seg_ckpt}")

    val = collect_gated(cls, seg_model, va_loader)
    test = collect_gated(cls, seg_model, te_loader)
    ones = np.ones(len(test["gt"]), dtype=bool)
    raw = metrics_bundle(test["gt"], test["mass"], test["dice"], ones)

    rows = [{
        "name": "TEST_s5_raw",
        "nonempty_dice": raw["nonempty_dice"],
        "empty_fp_rate": raw["empty_fp_rate"],
        "sensitivity": raw["sensitivity"],
        "thr": None,
    }]
    lines = [
        "# S5 × 2.5D presence gate",
        "",
        "Frozen S5 SEG · Presence25D(z-1,z,z+1) · τ not in train loss",
        f"best VAL AUROC={best_auroc:.4f} · pos_weight={args.pos_weight}",
        f"TEST S5 raw: ne={raw['nonempty_dice']:.4f} fp={raw['empty_fp_rate']:.3f}",
        f"Baseline none raw: ne={NONE_NE} fp={NONE_FP}",
        "",
        "## Operating points",
    ]

    pv = pick_paper_tau(val)
    mt = eval_hard(test, pv["thr"])
    rows.append({
        "name": f"TEST_paper_{pv.get('pick', 'pick')}",
        **{k: mt[k] for k in (
            "thr", "accuracy", "sensitivity", "specificity", "precision",
            "f1", "auroc", "tp", "tn", "fp", "fn",
            "nonempty_dice", "empty_fp_rate", "seg_sensitivity",
        )},
        "val_ne": pv["nonempty_dice"],
        "val_empty_fp": pv["empty_fp_rate"],
        "val_pick": pv.get("pick"),
    })
    safe_print(
        f"==== paper pick={pv.get('pick')} τ={pv['thr']:.2f} ===="
    )
    safe_print(
        f"  VAL ne={pv['nonempty_dice']:.4f} fp={pv['empty_fp_rate']:.3f}"
    )
    safe_print(
        f"  TEST TF TP/TN/FP/FN={mt['tp']}/{mt['tn']}/{mt['fp']}/{mt['fn']} "
        f"acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
        f"spec={mt['specificity']:.3f}"
    )
    safe_print(
        f"  TEST gated ne={mt['nonempty_dice']:.4f} empty_fp={mt['empty_fp_rate']:.3f}"
    )
    lines.append(
        f"### paper ({pv.get('pick')}) τ={pv['thr']:.2f}\n"
        f"- VAL: ne={pv['nonempty_dice']:.4f} fp={pv['empty_fp_rate']:.3f}\n"
        f"- TEST TF: acc={mt['accuracy']:.3f} sens={mt['sensitivity']:.3f} "
        f"spec={mt['specificity']:.3f} FN={mt['fn']}\n"
        f"- TEST SEG@gate: ne={mt['nonempty_dice']:.4f} empty_fp={mt['empty_fp_rate']:.3f}"
    )

    # also sens floors for transparency
    for floor in (0.95, 0.90, 0.85, 0.80):
        best = None
        for i in range(1, 100):
            thr = i / 100.0
            m = eval_hard(val, thr)
            if m["sensitivity"] + 1e-9 < floor:
                continue
            key = (-m["empty_fp_rate"], m["nonempty_dice"])
            if best is None or key > best["_key"]:
                best = dict(m)
                best["_key"] = key
        if best is None:
            continue
        mt2 = eval_hard(test, best["thr"])
        rows.append({
            "name": f"TEST_sensFloor_{floor:.2f}",
            **{k: mt2[k] for k in (
                "thr", "accuracy", "sensitivity", "specificity", "precision",
                "f1", "auroc", "tp", "tn", "fp", "fn",
                "nonempty_dice", "empty_fp_rate", "seg_sensitivity",
            )},
        })
        safe_print(
            f"==== sens≥{floor:.2f} τ={best['thr']:.2f} "
            f"TEST ne={mt2['nonempty_dice']:.4f} fp={mt2['empty_fp_rate']:.3f}"
        )
        lines.append(
            f"### sens≥{floor:.2f} τ={best['thr']:.2f} → "
            f"ne={mt2['nonempty_dice']:.4f} fp={mt2['empty_fp_rate']:.3f}"
        )

    soft_t = eval_soft(test)
    rows.append({
        "name": "TEST_soft",
        "nonempty_dice": soft_t["nonempty_dice"],
        "empty_fp_rate": soft_t["empty_fp_rate"],
        "sensitivity": soft_t["sensitivity"],
        "specificity": soft_t["specificity"],
        "auroc": soft_t["auroc"],
        "thr": None,
    })
    safe_print(
        f"==== soft mask*p TEST ne={soft_t['nonempty_dice']:.4f} "
        f"fp={soft_t['empty_fp_rate']:.3f}"
    )
    lines.extend([
        "",
        f"### soft mask×p → ne={soft_t['nonempty_dice']:.4f} "
        f"fp={soft_t['empty_fp_rate']:.3f}",
        "",
        "## Refs",
        f"- none raw: ne={NONE_NE} fp={NONE_FP}",
        f"- S5 raw: ne={raw['nonempty_dice']:.4f} fp={raw['empty_fp_rate']:.3f}",
        "- dual-role @0.22: ne=0.600 fp=0.383",
    ])

    beats_ne = mt["nonempty_dice"] > NONE_NE
    beats_fp = mt["empty_fp_rate"] <= NONE_FP + 1e-9
    lines.extend([
        "",
        "## vs none",
        f"- ne beat: {beats_ne} ({mt['nonempty_dice']:.4f} vs {NONE_NE})",
        f"- fp beat/tie: {beats_fp} ({mt['empty_fp_rate']:.3f} vs {NONE_FP})",
        f"- strict dominate: {beats_ne and beats_fp}",
    ])

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": "freeze S5 + Presence25D gate",
        "best_val_auroc": best_auroc,
        "paper_pick": {"val": {k: v for k, v in pv.items() if not str(k).startswith("_")},
                       "test": mt},
        "s5_raw_test": {
            "nonempty_dice": raw["nonempty_dice"],
            "empty_fp_rate": raw["empty_fp_rate"],
        },
        "none_ref": {"nonempty_dice": NONE_NE, "empty_fp_rate": NONE_FP},
        "soft_test": soft_t,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# S5 × 2.5D empty gate",
            "",
            "- SEG: frozen `s5_concat/seg_best.pt`",
            "- Cls: ResNet18 on FLAIR stack (z-1,z,z+1)",
            "- τ not in BCE; VAL pick dominate-none else joint",
            "",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
