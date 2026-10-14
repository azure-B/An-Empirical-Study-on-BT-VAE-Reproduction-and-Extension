"""COVID gate: Presence25D + Head(μ on frozen none) → wprod + min-mass on S5.

Requires: results/covid_pipeline/{none,s5_concat}/seg_best.pt
Writes under results/covid_pipeline/ only (never bp_original).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import covid_config as C
from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead
from models_presence_25d import Presence25D
from train_lesion_presence import PIXEL_THR, binary_auroc, conf_matrix, metrics_bundle, _prep
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    atomic_torch_save,
    safe_load,
    safe_print,
)

OUT = C.RESULT_DIR / "gate_wprod_mass"
BATCH = 8
EPOCHS_25D = 60
EPOCHS_HEAD = 40
IMG_SIZE = (256, 256)


def neighbor_name(stem_to_name: dict, stem: str, delta: int) -> str:
    pid, sl = stem.rsplit("_", 1)
    sl_i = int(sl)
    width = len(sl)
    for cand in (
        f"{pid}_{sl_i + delta}",
        f"{pid}_{sl_i + delta:0{width}d}",
        f"{pid}_{sl_i + delta:03d}",
    ):
        if cand in stem_to_name:
            return stem_to_name[cand]
    return stem_to_name[stem]


class Stack25D(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        self.names = list(base_ds.filenames)
        self.stem_to_name = {Path(n).stem: n for n in self.names}
        self.stems = [Path(n).stem for n in self.names]

    def __len__(self):
        return len(self.names)

    def _gray(self, name):
        with Image.open(self.base.path_original / name) as img:
            img = img.convert("L").resize(IMG_SIZE, Image.BILINEAR)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return t / 255.0

    def _mask(self, name):
        with Image.open(self.base.path_label / name) as img:
            img = img.convert("L").resize(IMG_SIZE, Image.NEAREST)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return (t > 0).float()

    def __getitem__(self, idx):
        n = len(self.names)
        last = None
        for off in range(n):
            j = (idx + off) % n
            name = self.names[j]
            stem = Path(name).stem
            try:
                c = self._gray(name)
                p = self._gray(neighbor_name(self.stem_to_name, stem, -1))
                nxt = self._gray(neighbor_name(self.stem_to_name, stem, +1))
                lab = self._mask(name)
            except (OSError, ValueError, SyntaxError) as e:
                last = e
                continue
            return torch.stack([p, c, nxt], 0), lab, stem
        raise OSError(last)


def make_loader(ds, batch, shuffle):
    kw = dict(
        batch_size=batch, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"), drop_last=False,
    )
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def write_csv(path, rows):
    import csv
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


def train_25d(tr_loader, va_loader):
    cls = Presence25D().to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=DEVICE))
    opt = torch.optim.AdamW(cls.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    best, es, hist = -1.0, 0, []
    ckpt = OUT / "checkpoints" / "cls25d_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, EPOCHS_25D + 1):
        cls.train()
        tr, ntr, t0 = 0.0, 0, time.time()
        for batch in tr_loader:
            x, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            if seg.ndim == 3:
                seg = seg.unsqueeze(1)
            y = (seg.flatten(1).sum(1) > 0).float()
            opt.zero_grad(set_to_none=True)
            loss = bce(cls(x), y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        cls.eval()
        ys, ss = [], []
        with torch.inference_mode():
            for batch in va_loader:
                x, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
                if seg.ndim == 3:
                    seg = seg.unsqueeze(1)
                ys.append((seg.flatten(1).sum(1) > 0).cpu().numpy())
                ss.append(torch.sigmoid(cls(x)).cpu().numpy())
        yv, sv = np.concatenate(ys), np.concatenate(ss)
        auroc = binary_auroc(yv, sv)
        safe_print(f"[COVID 25d {ep}] bce={tr/max(1,ntr):.4f} auroc={auroc:.4f} ({time.time()-t0:.1f}s)")
        sch.step(auroc)
        atomic_torch_save({"model": cls.state_dict(), "val_auroc": auroc}, OUT / "checkpoints" / "cls25d_last.pt")
        if auroc > best + 1e-4:
            best, es = auroc, 0
            atomic_torch_save({"model": cls.state_dict(), "val_auroc": auroc}, ckpt)
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= 12:
                safe_print(f"25d early stop @ {ep}")
                break
        hist.append({"ep": ep, "auroc": auroc})
    cls.load_state_dict(safe_load(ckpt, map_location="cpu")["model"])
    return cls, best


def train_mu_head(none_vae, tr1, va1):
    head = LesionPresenceHead(32 * BASE, hidden_dim=128, dropout=0.5).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=DEVICE))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    none_vae.eval()
    for p in none_vae.parameters():
        p.requires_grad = False
    best, es = -1.0, 0
    ckpt = OUT / "checkpoints" / "mu_head_best.pt"

    def collect(loader):
        ys, ss = [], []
        with torch.inference_mode():
            for batch in loader:
                flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
                mu, _, _ = none_vae.encoder(flair)
                ys.append((seg.flatten(1).sum(1) > 0).cpu().numpy())
                ss.append(torch.sigmoid(head(mu)).cpu().numpy())
        return np.concatenate(ys), np.concatenate(ss)

    for ep in range(1, EPOCHS_HEAD + 1):
        head.train()
        tr, ntr, t0 = 0.0, 0, time.time()
        for batch in tr1:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            y = (seg.flatten(1).sum(1) > 0).float()
            with torch.inference_mode():
                mu, _, _ = none_vae.encoder(flair)
            opt.zero_grad(set_to_none=True)
            loss = bce(head(mu.detach()), y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        yv, sv = collect(va1)
        auroc = binary_auroc(yv, sv)
        safe_print(f"[COVID μ-head {ep}] bce={tr/max(1,ntr):.4f} auroc={auroc:.4f} ({time.time()-t0:.1f}s)")
        sch.step(auroc)
        atomic_torch_save({"model": head.state_dict(), "val_auroc": auroc}, OUT / "checkpoints" / "mu_head_last.pt")
        if auroc > best + 1e-4:
            best, es = auroc, 0
            atomic_torch_save({"model": head.state_dict(), "val_auroc": auroc}, ckpt)
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= 10:
                safe_print(f"μ-head early stop @ {ep}")
                break
    head.load_state_dict(safe_load(ckpt, map_location="cpu")["model"])
    return head, best


@torch.inference_mode()
def collect_all(s5, none_vae, head, cls25, loader1, loader3):
    s5.eval()
    none_vae.eval()
    head.eval()
    cls25.eval()
    bags = {"gt": [], "mass": [], "dice": [], "p_dual": [], "p_25d": []}
    it3 = iter(loader3)
    for batch in loader1:
        b3 = next(it3)
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        x3 = b3[0].to(DEVICE)
        mu, _, _ = none_vae.encoder(flair)
        p_dual = torch.sigmoid(head(mu))
        p_25d = torch.sigmoid(cls25(x3))
        logits, _, _ = s5(flair)
        prob = torch.sigmoid(logits)
        pred = (prob >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        dice = torch.where((ps == 0) & (gs == 0), torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["p_dual"].append(p_dual.cpu().numpy())
        bags["p_25d"].append(p_25d.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def eval_keep(bag, keep):
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "sensitivity": cm["sensitivity"],
        "specificity": cm["specificity"],
        "fn": cm["TF"], "fp": cm["FT"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "auroc": binary_auroc(yt, bag.get("score", bag["p_dual"])),
    }


def main():
    if not C.S5_BEST.exists() or not C.NONE_BEST.exists():
        raise FileNotFoundError(
            f"Need COVID SEG first: {C.S5_BEST} and {C.NONE_BEST}. "
            f"Run: python train_covid_seg.py --phase all"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.covid(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    safe_print(f"COVID gate OUT={OUT} n={len(data.train)}/{len(data.val)}/{len(data.test)}")

    tr3 = make_loader(Stack25D(data.train), BATCH, True)
    va3 = make_loader(Stack25D(data.val), BATCH, False)
    te3 = make_loader(Stack25D(data.test), BATCH, False)
    tr1 = DataLoader(data.train, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    none_vae = OrigVAE(base=BASE).to(DEVICE)
    none_vae.load_state_dict(safe_load(C.NONE_BEST, map_location="cpu")["model"])
    none_vae.eval()
    s5 = SkipVAE(base=BASE).to(DEVICE)
    s5.load_state_dict(safe_load(C.S5_BEST, map_location="cpu")["model"])
    s5.eval()

    cls25, a25 = train_25d(tr3, va3)
    head, ah = train_mu_head(none_vae, tr1, va1)

    val = collect_all(s5, none_vae, head, cls25, va1, va3)
    test = collect_all(s5, none_vae, head, cls25, te1, te3)
    for bag in (val, test):
        s = (bag["p_dual"] ** 1.5) * (bag["p_25d"] ** 1.3)
        bag["score"] = s / (s.max() + 1e-8)

    raw_t = eval_keep(test, np.ones(len(test["gt"]), dtype=bool))
    # sweep thr × min_mass on VAL, prefer high ne then low fp
    masses = [0, 20, 40, 80, 160, 320, 640, 1280, 2560]
    best = None
    for i in range(1, 80):
        thr = i / 100.0
        for mmin in masses:
            keep = (val["score"] >= thr) & (val["mass"] >= mmin)
            m = eval_keep(val, keep)
            m.update({"thr": thr, "min_mass": mmin})
            # joint: ne - 0.5*fp, require ne not collapsed
            j = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
            if best is None or j > best["_j"]:
                best = dict(m)
                best["_j"] = j

    keep_t = (test["score"] >= best["thr"]) & (test["mass"] >= best["min_mass"])
    mt = eval_keep(test, keep_t)
    # also dual@0.22-like and s5 raw
    rows = [
        {"name": "TEST_s5_raw", **raw_t},
        {"name": "TEST_wprod_mass", "thr": best["thr"], "min_mass": best["min_mass"], **mt},
    ]
    for thr in (0.10, 0.22, 0.30):
        m = eval_keep(test, test["p_dual"] >= thr)
        rows.append({"name": f"TEST_mu_gate_{thr:.2f}", "thr": thr, **m})

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "dataset": C.DATASET_NAME,
        "isolated_from": "results/bp_original",
        "a25": a25, "a_mu": ah,
        "val_pick": {k: best[k] for k in ("thr", "min_mass", "nonempty_dice", "empty_fp_rate")},
        "test_s5_raw": raw_t,
        "test_wprod_mass": mt,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# COVID gate (wprod + mass) — isolated from BP",
        "",
        f"25d AUROC={a25:.4f} · μ-head AUROC={ah:.4f}",
        f"S5 raw TEST: ne={raw_t['nonempty_dice']:.4f} fp={raw_t['empty_fp_rate']:.3f}",
        f"wprod+mass τ={best['thr']:.2f} mass≥{best['min_mass']}: "
        f"ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f}",
        "",
        "BP numbers live in `results/bp_original/` — do not mix ckpts.",
    ]
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (C.RESULT_DIR / "COMPARE_ALL.md").write_text(
        "\n".join([
            "# COVID pipeline COMPARE (isolated)",
            "",
            "BP counterpart: `results/bp_original/COMPARE_ALL.md`",
            "",
            f"## S5 raw: ne={raw_t['nonempty_dice']:.4f} fp={raw_t['empty_fp_rate']:.3f}",
            f"## wprod+mass: ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f}",
            "",
            "Detail: `gate_wprod_mass/COMPARE.md`",
        ]) + "\n",
        encoding="utf-8",
    )
    safe_print(
        f"COVID DONE raw ne={raw_t['nonempty_dice']:.4f} fp={raw_t['empty_fp_rate']:.3f} | "
        f"gated ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f}"
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
