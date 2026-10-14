"""COVID labeled-ratio sweep for a Fig. 2 style Dice curve.

Val/test stay the full covid_8_1_1 split. Only the train set is subsampled
to {0.1, 0.3, 0.5, 0.7, 0.9}. BT encoder is the existing COVID checkpoint.

Two independent Fig. 2 curves, so a later plot can overlay or extend them:
  s5      raw SkipVAE (no gate)
  s5_cls  same S5 mask, kept iff P(lesion) ≥ τ (τ picked on VAL)
No second encoder, no 2.5D CNN, no weighted product. A new series is
another mode directory with its own report.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from pipeline import config as C
from pipeline.data import datasets
from pipeline.metrics import binary_auroc, metrics_bundle
from pipeline.models_presence import LesionPresenceHead
from pipeline.models_skip import VAE as SkipVAE
from pipeline.engine import (
    DEVICE,
    OrigVAE,
    SEG_EPOCHS,
    SEG_ES_PATIENCE,
    SEG_LR,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    load_bt_into_encoder,
    safe_load,
    safe_print,
    seg_step_loss,
    split_dice_stats,
)

RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
SEED = 0
OUT_ROOT = C.RESULT_DIR / "label_ratio"
# RTX 5060 8151 MiB. One AdamW step, 256², base 32 (measured reserved):
#   none bs8 = 6700, s5 bs8 = 7806 (too tight), both bs4 <= 4800.
# Keep reserved under ~60% so the desktop is not paged out.
VRAM_BATCH = {"none": 4, "s5": 4}
CLS_EPOCHS = 40
CLS_ES = 10
CLS_POS_WEIGHT = 2.0


class IndexSubset(Dataset):
    def __init__(self, base, indices: list[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def patient_id(name: str) -> str:
    stem = Path(name).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def subset_indices(names: list[str], ratio: float, seed: int = SEED) -> list[int]:
    """Patient-stratified slice subsample. Exact count = round(n * ratio)."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(patient_id(name), []).append(i)
    rng = np.random.RandomState(seed)
    target = int(round(len(names) * ratio))
    target = min(len(names), max(1, target))
    picked: list[int] = []
    leftover: list[int] = []
    for pid in sorted(groups):
        idx = groups[pid][:]
        rng.shuffle(idx)
        k = int(np.floor(len(idx) * ratio))
        picked.extend(idx[:k])
        leftover.extend(idx[k:])
    rng.shuffle(leftover)
    if len(picked) < target:
        picked.extend(leftover[: target - len(picked)])
    elif len(picked) > target:
        rng.shuffle(picked)
        picked = picked[:target]
    return sorted(picked)


def make_bundle(ratio: float, batch_size: int):
    full = datasets.covid(batch_size=batch_size, num_workers=0, pin_memory=False)
    idx = subset_indices(list(full.train.filenames), ratio)
    train = IndexSubset(full.train, idx)
    bs = min(batch_size, max(1, len(train)))
    train_loader = DataLoader(
        train, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        full.val, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        full.test, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False,
    )
    return full, train_loader, val_loader, test_loader, len(idx)


@torch.inference_mode()
def eval_split(model, loader, kind: str) -> dict:
    """Deterministic μ decode. Dice mean is the Fig. 2 style score."""
    model.eval()
    dices, gts, preds = [], [], []
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        if kind == "s5":
            mu, _lv, _z, skips = model.encoder.forward_features(x)
            logits = model.decoder(mu, skips)
        else:
            mu, _lv, _z = model.encoder(x)
            logits = model.decoder(mu)
        dices.append(batch_dice(logits, y))
        ps, gs = batch_pred_gt_sums(logits, y)
        preds.append(ps)
        gts.append(gs)
    st = split_dice_stats(np.concatenate(dices), np.concatenate(gts), np.concatenate(preds))
    return {
        "dice": st["dice_mean"],
        "dice_median": st["dice_median"],
        "nonempty_dice": st["mean_dice_nonempty"],
        "empty_fp_rate": st["empty_fp_rate"],
        "n": st["n"],
    }


SERIES_NOTE = (
    "Independent curves. s5 = raw segmentation, s5_cls = S5 gated by "
    "P(lesion). Add another mode's report.json to extend the plot later."
)


def _load_curve_rows() -> list[dict]:
    found: dict[tuple, dict] = {}
    if not OUT_ROOT.exists():
        return []
    # report.json is the curve point. report_seg.json is the raw S5/none
    # point written before classification, so a killed run still keeps it.
    for d in sorted(OUT_ROOT.glob("r*/**/report*.json")):
        name = d.name
        if name not in {"report.json", "report_seg.json"}:
            continue
        row = json.loads(d.read_text(encoding="utf-8"))
        key = (row["mode"], float(row["ratio"]))
        prev = found.get(key)
        if prev is None or name == "report.json":
            found[key] = row
    rows = list(found.values())
    rows.sort(key=lambda r: (r["mode"], r["ratio"]))
    return rows


def refresh_curve():
    rows = _load_curve_rows()
    series: dict[str, list] = {}
    for r in rows:
        series.setdefault(r["mode"], []).append(r)
    payload = {"note": SERIES_NOTE, "series": series}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "curve.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# COVID Dice vs labeled-data ratio",
        "",
        "Train subset only. Val/test = full covid_8_1_1. BT encoder shared.",
        "TEST dice = mean over slices, deterministic μ.",
        "Each section is its own curve. A later model is a new section.",
        "",
    ]
    for mode, group in series.items():
        lines += [
            f"## {mode}",
            "",
            "| ratio | TEST dice | nonempty | empty FP | best ep |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in group:
            t = r["test"]
            lines.append(
                f"| {r['ratio']:.1f} | {t['dice']:.4f} | "
                f"{t['nonempty_dice']:.4f} | {t['empty_fp_rate']:.3f} | {r['best_ep']} |"
            )
        lines.append("")
    (OUT_ROOT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def _prep_xy(batch):
    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    return x, y


@torch.inference_mode()
def collect_s5_cls(model, head, loader):
    """Raw S5 dice/mass plus presence score from Head(μ)."""
    model.eval()
    head.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        x, y = _prep_xy(batch)
        mu, _lv, _z, skips = model.encoder.forward_features(x)
        logits = model.decoder(mu, skips)
        score = torch.sigmoid(head(mu))
        dices = batch_dice(logits, y)
        ps, gs = batch_pred_gt_sums(logits, y)
        bags["gt"].append(gs)
        bags["mass"].append(ps)
        bags["dice"].append(dices)
        bags["score"].append(score.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def pick_tau(val):
    """VAL τ that maximizes all-slice mean Dice. Blanking empty FP raises it."""
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        keep = val["score"] >= thr
        m = metrics_bundle(val["gt"], val["mass"], val["dice"], keep)
        key = (m["overall_dice"], -m["empty_fp_rate"])
        if best is None or key > best["key"]:
            best = {"thr": thr, "key": key, "m": m}
    return best["thr"], best["m"]


def prob_group_stats(gt: np.ndarray, score: np.ndarray) -> dict:
    """p = σ(logit) is P(lesion present | slice). Split by GT."""
    lesion = gt > 0
    empty = ~lesion
    def pack(s):
        if s.size == 0:
            return {"n": 0, "mean": None, "median": None}
        return {"n": int(s.size), "mean": float(s.mean()), "median": float(np.median(s))}
    return {"p_lesion": pack(score[lesion]), "p_empty": pack(score[empty])}


def train_s5_cls(ratio: float, seg_ckpt: Path, train_loader, val_loader, test_loader, n_train: int):
    pct = int(round(ratio * 100))
    out_dir = OUT_ROOT / f"r{pct:02d}" / "s5_cls"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    if report_path.exists():
        safe_print(f"skip existing {report_path}")
        return
    model = SkipVAE(base=32).to(DEVICE)
    st = safe_load(seg_ckpt, map_location="cpu")
    model.load_state_dict(st["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLS_POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_auroc, bad, best_ep = -1.0, 0, 0
    best_path = out_dir / "head_best.pt"
    trend = []
    safe_print(
        f"[r{pct} s5+cls] freeze S5 | Head(μ) logit → p=σ(logit)=P(lesion) | "
        f"BCEWithLogits pos_weight={CLS_POS_WEIGHT}"
    )
    for ep in range(1, CLS_EPOCHS + 1):
        head.train()
        total, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            x, y = _prep_xy(batch)
            with torch.no_grad():
                mu, _lv, _z, _sk = model.encoder.forward_features(x)
            y_cls = (y.flatten(1).sum(1) > 0).float()
            opt.zero_grad(set_to_none=True)
            loss = bce(head(mu), y_cls)
            loss.backward()
            opt.step()
            total += float(loss.item()) * y_cls.size(0)
            n += y_cls.size(0)
        val = collect_s5_cls(model, head, val_loader)
        auroc = binary_auroc(val["gt"] > 0, val["score"])
        pg = prob_group_stats(val["gt"], val["score"])
        rec = {
            "ep": ep,
            "train_bce": total / max(n, 1),
            "val_auroc": auroc,
            "val_p_lesion_mean": pg["p_lesion"]["mean"],
            "val_p_lesion_median": pg["p_lesion"]["median"],
            "val_p_empty_mean": pg["p_empty"]["mean"],
            "val_p_empty_median": pg["p_empty"]["median"],
        }
        trend.append(rec)
        safe_print(
            f"[r{pct} cls {ep}/{CLS_EPOCHS}] bce={rec['train_bce']:.4f} auroc={auroc:.4f} "
            f"P(lesion)|GT+={rec['val_p_lesion_mean']:.3f} "
            f"P(lesion)|empty={rec['val_p_empty_mean']:.3f} ({time.time()-t0:.1f}s)"
        )
        if auroc > best_auroc + 1e-4:
            best_auroc, bad, best_ep = auroc, 0, ep
            atomic_torch_save({"head": head.state_dict(), "ep": ep, "auroc": auroc}, best_path)
        else:
            bad += 1
            if bad >= CLS_ES:
                safe_print(f"early stop cls r{pct} @ {ep}")
                break
    head.load_state_dict(safe_load(best_path, map_location="cpu")["head"])
    (out_dir / "prob_trend.json").write_text(json.dumps(trend, indent=2), encoding="utf-8")
    val = collect_s5_cls(model, head, val_loader)
    test = collect_s5_cls(model, head, test_loader)
    tau, _val_m = pick_tau(val)
    keep = test["score"] >= tau
    mt = metrics_bundle(test["gt"], test["mass"], test["dice"], keep)
    report = {
        "dataset": "covid",
        "ratio": ratio,
        "mode": "s5_cls",
        "n_train": n_train,
        "loss": "BCEWithLogits",
        "probability": "p = sigmoid(Head(μ)) = P(lesion present | slice)",
        "tau": tau,
        "best_ep": best_ep,
        "val_auroc": best_auroc,
        "val_prob": prob_group_stats(val["gt"], val["score"]),
        "test_prob": prob_group_stats(test["gt"], test["score"]),
        "prob_trend": "prob_trend.json",
        "test": {
            "dice": float(mt["overall_dice"]),
            "nonempty_dice": float(mt["nonempty_dice"]),
            "empty_fp_rate": float(mt["empty_fp_rate"]),
        },
        "seed": SEED,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        from pipeline.cases import save_covid_case
        save_covid_case("s5_cls", ratio, model, head, tau)
    except Exception as exc:
        safe_print(f"case panel skip s5_cls r{pct}: {exc}")
    safe_print(
        f"TEST r{pct} s5+cls τ={tau:.2f} dice={mt['overall_dice']:.4f} "
        f"ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f}"
    )
    del model, head
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    refresh_curve()


def run_one(mode: str, ratio: float, max_ep: int):
    assert mode in {"none", "s5"}
    pct = int(round(ratio * 100))
    out_dir = OUT_ROOT / f"r{pct:02d}" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report_seg.json"
    # none writes report.json when SEG finishes. r10 already has that point.
    if report_path.exists() or (mode == "none" and (out_dir / "report.json").exists()):
        safe_print(f"skip existing {out_dir}")
        if mode == "s5":
            seg_bs = VRAM_BATCH["s5"]
            full, train_loader, val_loader, test_loader, n_train = make_bundle(ratio, seg_bs)
            train_s5_cls(ratio, out_dir / "seg_best.pt", train_loader, val_loader, test_loader, n_train)
        return
    seg_bs = VRAM_BATCH[mode]
    full, train_loader, val_loader, test_loader, n_train = make_bundle(ratio, seg_bs)
    safe_print(
        f"[ratio {ratio:.1f} {mode}] train={n_train}/{len(full.train)} "
        f"val={len(full.val)} test={len(full.test)} batch={seg_bs} "
        f"(5060 8GB, reserved budget ~4.8GB)"
    )
    model = (OrigVAE if mode == "none" else SkipVAE)(base=32).to(DEVICE)
    bt = C.BT_BEST if C.BT_BEST.exists() else C.BT_LAST
    opt = torch.optim.AdamW(model.parameters(), lr=SEG_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.95, patience=3, min_lr=1e-8)
    best_loss, best_ep, bad = float("inf"), 0, 0
    best_path, last_path = out_dir / "seg_best.pt", out_dir / "seg_last.pt"
    start_ep = 1
    if last_path.exists():
        st = safe_load(last_path, map_location="cpu")
        model.load_state_dict(st["model"])
        start_ep = int(st.get("ep", 0)) + 1
        best_loss = float(st.get("best_loss", best_loss))
        best_ep = int(st.get("best_ep", 0))
        bad = max(0, start_ep - 1 - best_ep)
        safe_print(f"resume {mode} r{pct} from ep {start_ep} best={best_loss:.4f} @ {best_ep}")
        del st
    else:
        load_bt_into_encoder(model.encoder, bt)
    limit = max_ep
    for ep in range(start_ep, limit + 1):
        model.train()
        tr_loss, ntr = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, _dice, _kl = seg_step_loss(model, x, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            ntr += 1
        model.eval()
        va_loss, nva = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss, _, _ = seg_step_loss(model, x, y)
                va_loss += float(loss.item())
                nva += 1
        va_loss /= max(1, nva)
        sched.step(va_loss)
        safe_print(
            f"[r{pct} {mode} {ep}/{limit}] train={tr_loss/max(1,ntr):.4f} "
            f"val={va_loss:.4f} ({time.time()-t0:.1f}s)"
        )
        payload = {
            "model": model.state_dict(),
            "ep": ep,
            "best_loss": best_loss,
            "best_ep": best_ep,
            "ratio": ratio,
            "mode": mode,
        }
        atomic_torch_save(payload, last_path)
        if va_loss < best_loss - 1e-5:
            best_loss, best_ep, bad = va_loss, ep, 0
            payload["best_loss"] = best_loss
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, best_path)
        else:
            bad += 1
            if bad >= SEG_ES_PATIENCE:
                safe_print(f"early stop r{pct} {mode} @ {ep}")
                break

    st = safe_load(best_path if best_path.exists() else last_path, map_location="cpu")
    model.load_state_dict(st["model"])
    test = eval_split(model, test_loader, mode)
    report = {
        "dataset": "covid",
        "ratio": ratio,
        "mode": mode,
        "n_train": n_train,
        "n_train_full": len(full.train),
        "best_ep": int(st.get("best_ep", best_ep)),
        "best_val_loss": float(st.get("best_loss", best_loss)),
        "test": test,
        "seed": SEED,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Same numbers under report.json so the raw S5 curve is a first-class
    # series, separate from rXX/s5_cls/report.json.
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        from pipeline.cases import save_covid_case
        save_covid_case(mode, ratio, model)
    except Exception as exc:
        safe_print(f"case panel skip {mode} r{pct}: {exc}")
    safe_print(
        f"TEST r{pct} {mode} raw dice={test['dice']:.4f} ne={test['nonempty_dice']:.4f} "
        f"fp={test['empty_fp_rate']:.3f}"
    )
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if mode == "s5":
        train_s5_cls(ratio, best_path if best_path.exists() else last_path, train_loader, val_loader, test_loader, n_train)
    refresh_curve()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--modes", default="none,s5")
    p.add_argument("--max-ep", type=int, default=SEG_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # sanity: full covid sizes
    data = datasets.covid(batch_size=4, num_workers=0, pin_memory=False)
    safe_print(f"covid full train/val/test {len(data.train)}/{len(data.val)}/{len(data.test)}")
    for r in ratios:
        n = len(subset_indices(list(data.train.filenames), r))
        safe_print(f"  ratio {r:.1f} -> {n} train slices")
    for mode in modes:
        for ratio in ratios:
            run_one(mode, ratio, args.max_ep)
    refresh_curve()
    safe_print("curve", OUT_ROOT / "curve.json")


if __name__ == "__main__":
    main()
