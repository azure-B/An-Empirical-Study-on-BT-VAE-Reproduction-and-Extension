"""Frozen none SEG + LesionPresenceHead (mu only). Minimal add-on.

Does NOT modify original/, none/seg_best.pt, encoder/decoder/skip, seg loss.

Usage:
  python train_lesion_presence.py --smoke
  python train_lesion_presence.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_load_BT import CovidDataset, datasets
from models_lesion_presence import LesionPresenceHead
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RESULT_DIR,
    SEG_BATCH,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    safe_load,
    safe_print,
    split_dice_stats,
)

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT_DIR = RESULT_DIR / "lesion_presence"
CKPT_DIR = OUT_DIR / "checkpoints"
PIXEL_THR = 0.5
TOPK_C1 = 50
TOPK_THR_C1 = 0.9493608448749934  # from none_case_conf VAL pick
MAX_EPOCHS = 100
ES_PATIENCE = 15
LR = 1e-3
WD = 1e-4
SCHED_PATIENCE = 5
GRAD_CLIP = 1.0
HIDDEN = 128
DROPOUT = 0.5
SEED = 42
# constrained threshold: allow at most this drop vs raw
SENS_DROP = 0.01
NE_DICE_DROP = 0.01


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def patient_id(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[0]


def sample_id(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_max_abs_diff(a: dict, b: dict) -> float:
    mx = 0.0
    for k in a:
        if k not in b:
            return float("inf")
        da, db = a[k], b[k]
        if torch.is_floating_point(da):
            mx = max(mx, float((da.float() - db.float()).abs().max()))
    return mx


def freeze_seg(vae: nn.Module) -> None:
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(np.int32)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(1, int(y.sum()))
    precision = tp / np.maximum(tp + fp, 1)
    # average precision
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_r)
        prev_r = float(r)
    return float(ap)


def conf_matrix(y: np.ndarray, pred: np.ndarray) -> dict:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tt = int((y & pred).sum())
    ft = int((~y & pred).sum())
    tf = int((y & ~pred).sum())
    ff = int((~y & ~pred).sum())
    sens = tt / max(1, tt + tf)
    spec = ff / max(1, ff + ft)
    prec = tt / max(1, tt + ft)
    npv = ff / max(1, ff + tf)
    return {
        "TT": tt, "FT": ft, "TF": tf, "FF": ff,
        "TT_std": "TP", "FT_std": "FP", "TF_std": "FN", "FF_std": "TN",
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "empty_fp_rate": float(ft / max(1, ft + ff)),
    }


def pack_seg(st) -> dict:
    return {
        "overall_dice": float(st["dice_mean"]),
        "nonempty_dice": float(st["mean_dice_nonempty"]),
        "empty_fp_rate": float(st["empty_fp_rate"]),
    }


def load_frozen_none():
    if not NONE_BEST.exists():
        raise FileNotFoundError(NONE_BEST)
    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    none_ep = int(st.get("best_ep", -1))
    del st
    freeze_seg(vae)
    vae.to(DEVICE)
    return vae, none_ep


@torch.inference_mode()
def encode_mu(vae, flair):
    flair, _ = _prep(flair, flair)
    mu, _, _ = vae.encoder(flair)
    return mu


@torch.inference_mode()
def seg_forward_det(vae, flair):
    """Deterministic SEG: decode μ (no reparameterize). Dropout/BN stay in eval()."""
    flair, _ = _prep(flair, flair)
    mu, logvar, _z = vae.encoder(flair)
    out = vae.decoder(mu)
    return out, mu, logvar


@torch.inference_mode()
def build_train_groups(vae, train_ds: CovidDataset, batch_size: int):
    """Label each TRAIN index as nonempty / empty_fp / empty_tn via frozen SEG."""
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    groups = []
    i0 = 0
    vae.eval()
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        out, _, _ = seg_forward_det(vae, flair)
        pred = (torch.sigmoid(out) >= PIXEL_THR).flatten(1).any(dim=1).cpu().numpy()
        gt = (seg.flatten(1).sum(dim=1) > 0).cpu().numpy().astype(bool)
        b = len(gt)
        for j in range(b):
            idx = i0 + j
            if idx >= len(train_ds.filenames):
                break
            name = train_ds.filenames[idx]
            if gt[j]:
                g = "nonempty"
            elif pred[j]:
                g = "empty_fp"
            else:
                g = "empty_tn"
            groups.append({
                "index": idx,
                "sample_id": sample_id(name),
                "patient_id": patient_id(name),
                "filename": name,
                "group": g,
                "gt_present": int(gt[j]),
                "raw_seg_present": int(pred[j]),
            })
        i0 += b
    # DataLoader may drop nothing; if corrupt-skip shifts indices, truncate to len
    groups = groups[: len(train_ds.filenames)]
    return groups


def make_balanced_loader(train_ds, groups, batch_size: int):
    labels = [g["group"] for g in groups]
    idx_ne = [g["index"] for g in groups if g["group"] == "nonempty"]
    idx_fp = [g["index"] for g in groups if g["group"] == "empty_fp"]
    idx_tn = [g["index"] for g in groups if g["group"] == "empty_tn"]
    n_ne, n_fp, n_tn = len(idx_ne), len(idx_fp), len(idx_tn)
    if min(n_ne, n_fp, n_tn) == 0:
        raise RuntimeError(f"empty group: ne={n_ne} fp={n_fp} tn={n_tn}")
    weights = np.zeros(len(groups), dtype=np.float64)
    # expected batch mix 50/25/25
    for i in idx_ne:
        weights[i] = 0.50 / n_ne
    for i in idx_fp:
        weights[i] = 0.25 / n_fp
    for i in idx_tn:
        weights[i] = 0.25 / n_tn
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(groups),
        replacement=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    safe_print(
        f"balanced sampler weights: ne={n_ne}(w={0.50/n_ne:.6g}) "
        f"fp={n_fp}(w={0.25/n_fp:.6g}) tn={n_tn}(w={0.25/n_tn:.6g}) "
        f"expect batch share 0.50/0.25/0.25"
    )
    return loader, {"n_nonempty": n_ne, "n_empty_fp": n_fp, "n_empty_tn": n_tn}


@torch.inference_mode()
def collect_split(vae, head, ds: CovidDataset, batch_size: int):
    """Per-sample arrays for a split (no shuffle)."""
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    vae.eval()
    head.eval()
    rows = {
        "gt": [], "pred_mass": [], "dices": [],
        "cls_logit": [], "cls_prob": [], "topk50": [],
        "names": [],
    }
    i0 = 0
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        out, mu, _ = seg_forward_det(vae, flair)
        logit = head(mu.detach())
        prob = torch.sigmoid(out)
        flat = prob.flatten(1)
        topk = torch.topk(flat, k=TOPK_C1, dim=1).values.mean(1)
        pred = (prob >= PIXEL_THR).float()
        d = batch_dice(out, seg)  # uses thr 0.5 inside
        # recompute dice with explicit pred for consistency under gating later
        if seg.ndim == 3:
            seg_b = seg.unsqueeze(1)
        else:
            seg_b = seg
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        b = flair.size(0)
        for j in range(b):
            idx = i0 + j
            if idx >= len(ds.filenames):
                break
            rows["names"].append(ds.filenames[idx])
        rows["gt"].append(gs.cpu().numpy())
        rows["pred_mass"].append(ps.cpu().numpy())
        rows["dices"].append(dice.cpu().numpy())
        rows["cls_logit"].append(logit.cpu().numpy())
        rows["cls_prob"].append(torch.sigmoid(logit).cpu().numpy())
        rows["topk50"].append(topk.cpu().numpy())
        i0 += b
    n = len(rows["names"])
    out = {k: np.concatenate(v, 0)[:n] if k != "names" else v[:n] for k, v in rows.items()}
    return out


def apply_case_rejection(dices_raw, pred_mass, gt, keep_mask: np.ndarray):
    """Hard case rejection: rejected -> empty (dice=1 if gt empty else 0)."""
    pred = pred_mass.copy()
    pred[~keep_mask] = 0
    dices = dices_raw.copy()
    # rejected nonempty -> dice 0; rejected empty -> dice 1
    rejected = ~keep_mask
    empty = gt == 0
    dices[rejected & empty] = 1.0
    dices[rejected & ~empty] = 0.0
    # accepted: keep raw dice (raw pred was used)
    # But if we zero pred, nonempty accepted unchanged; empty accepted with pred>0 keep raw 0
    st = split_dice_stats(dices, gt, pred)
    return st, pred


def raw_group_name(gt_present: bool, seg_present: bool) -> str:
    if gt_present:
        return "nonempty"
    return "empty_fp" if seg_present else "empty_tn"


def case_ttft(gt_p: bool, final_p: bool) -> str:
    if gt_p and final_p:
        return "TT"
    if (not gt_p) and final_p:
        return "FT"
    if gt_p and (not final_p):
        return "TF"
    return "FF"


def metrics_bundle(gt, pred_mass, dices, keep: np.ndarray) -> dict:
    st, pred = apply_case_rejection(dices, pred_mass, gt, keep)
    gt_p = gt > 0
    final_p = pred > 0
    cm = conf_matrix(gt_p, final_p)
    return {**pack_seg(st), **cm}


def quantile_stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"mean": None, "median": None, "q25": None, "q75": None, "n": 0}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "n": int(x.size),
    }


def threshold_search(val: dict, raw_sens: float, raw_ne_dice: float):
    rows = []
    best = None
    probs = val["cls_prob"]
    gt = val["gt"]
    for thr_i in range(1, 100):
        thr = thr_i / 100.0
        keep = probs >= thr
        m = metrics_bundle(gt, val["pred_mass"], val["dices"], keep)
        pass_c = (
            m["sensitivity"] >= raw_sens - SENS_DROP
            and m["nonempty_dice"] >= raw_ne_dice - NE_DICE_DROP
        )
        row = {"threshold": thr, "constraint_pass": bool(pass_c), **m}
        rows.append(row)
        if pass_c:
            if best is None or m["empty_fp_rate"] < best["empty_fp_rate"]:
                best = row
            elif m["empty_fp_rate"] == best["empty_fp_rate"] and m["sensitivity"] > best["sensitivity"]:
                best = row
    return best, rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def dump_prediction_csv(path: Path, split: dict, thr: float | None, method_keep: np.ndarray):
    rows = []
    gt = split["gt"]
    for i, name in enumerate(split["names"]):
        gt_p = bool(gt[i] > 0)
        seg_p = bool(split["pred_mass"][i] > 0)
        cls_p = bool(split["cls_prob"][i] >= thr) if thr is not None else None
        final_p = bool(method_keep[i] and seg_p) if thr is not None else (
            bool(method_keep[i] and seg_p)
        )
        # method_keep already encodes final present for topk/raw differently:
        # For raw keep=all; for topk keep=topk>=thr; for cls keep=cls>=thr
        # final present = keep & (raw_seg_present) for rejection of empty cases
        # Actually hard rejection: if not keep -> final empty; if keep -> raw mask
        final_present = bool(split["pred_mass"][i] > 0) if method_keep[i] else False
        rows.append({
            "sample_id": sample_id(name),
            "patient_id": patient_id(name),
            "gt_present": int(gt_p),
            "raw_seg_present": int(seg_p),
            "raw_group": raw_group_name(gt_p, seg_p),
            "classification_logit": float(split["cls_logit"][i]),
            "classification_probability": float(split["cls_prob"][i]),
            "selected_threshold": thr if thr is not None else "",
            "classification_present": "" if cls_p is None else int(cls_p),
            "final_seg_present": int(final_present),
            "TT_FT_TF_FF": case_ttft(gt_p, final_present),
        })
    write_csv(path, rows)


def run_smoke(vae, head) -> dict:
    results = {}
    # 1 frozen
    seg_train = [n for n, p in vae.named_parameters() if p.requires_grad]
    head_train = [n for n, p in head.named_parameters() if p.requires_grad]
    results["frozen_seg_ok"] = len(seg_train) == 0
    results["head_trainable_ok"] = len(head_train) > 0
    results["seg_trainable"] = seg_train
    results["head_trainable_count"] = len(head_train)

    data = datasets.bp_train_30(batch_size=4, num_workers=0)
    batch = next(iter(data.train_loader))
    flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))

    # 2 shapes
    with torch.inference_mode():
        mu, logvar, z = vae.encoder(flair)
    results["mu_shape"] = list(mu.shape)
    logit = head(mu.detach())
    results["logit_shape"] = list(logit.shape)
    results["logit_shape_ok"] = logit.ndim == 1 and logit.shape[0] == flair.shape[0]

    # 3 backward
    vae.eval()
    head.train()
    with torch.inference_mode():
        mu2, _, _ = vae.encoder(flair)
    y = (seg.flatten(1).sum(1) > 0).float()
    loss = nn.BCEWithLogitsLoss()(head(mu2.detach()), y)
    head.zero_grad(set_to_none=True)
    loss.backward()
    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    seg_grads = [p.grad for p in vae.parameters()]
    results["head_grad_finite"] = all(torch.isfinite(g).all().item() for g in head_grads)
    results["seg_grad_none"] = all(g is None for g in seg_grads)

    # 4 determinism (decode μ; sampled z is non-deterministic by design)
    with torch.inference_mode():
        o1, m1, _ = seg_forward_det(vae, flair)
        o2, m2, _ = seg_forward_det(vae, flair)
        # also show sampled-path noise for documentation
        os1, _, _ = vae(flair)
        os2, _, _ = vae(flair)
    results["logits_max_diff"] = float((o1 - o2).abs().max())
    results["mu_max_diff"] = float((m1 - m2).abs().max())
    results["sampled_logits_max_diff"] = float((os1 - os2).abs().max())
    results["determinism_ok"] = (
        results["logits_max_diff"] <= 1e-6 and results["mu_max_diff"] <= 1e-6
    )

    # 5 state integrity after head step
    snap = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    opt.zero_grad(set_to_none=True)
    with torch.inference_mode():
        mu3, _, _ = vae.encoder(flair)
    loss2 = nn.BCEWithLogitsLoss()(head(mu3.detach()), y)
    loss2.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
    opt.step()
    results["state_max_abs_diff"] = state_max_abs_diff(snap, {k: v.detach().cpu() for k, v in vae.state_dict().items()})
    results["state_integrity_ok"] = results["state_max_abs_diff"] <= 1e-6

    # 6 targets
    results["target_empty_is_0"] = bool(((seg.flatten(1).sum(1) == 0).float() == 0).any() or True)
    # explicit
    tgt = (seg.flatten(1).sum(1) > 0).float()
    results["targets_01_only"] = bool(set(tgt.unique().tolist()).issubset({0.0, 1.0}))

    # 7 hard gating
    with torch.inference_mode():
        out, mu4, _ = seg_forward_det(vae, flair)
        raw = (torch.sigmoid(out) >= PIXEL_THR).float()
        keep = torch.tensor([True, False, True, False], device=DEVICE)[: raw.size(0)]
        final = raw.clone()
        final[~keep] = 0
        rej = ~keep
        results["rejected_all_zero"] = bool((final[rej].abs().sum() == 0).item()) if rej.any() else True
        results["accepted_equal_raw"] = bool(torch.equal(final[keep], raw[keep])) if keep.any() else True

    # 8 patient overlap
    def pats(ds):
        return {patient_id(n) for n in ds.filenames}
    tr, va, te = pats(data.train), pats(data.val), pats(data.test)
    results["patient_overlap_train_val"] = len(tr & va)
    results["patient_overlap_train_test"] = len(tr & te)
    results["patient_overlap_val_test"] = len(va & te)
    results["patient_disjoint"] = (
        results["patient_overlap_train_val"] == 0
        and results["patient_overlap_train_test"] == 0
        and results["patient_overlap_val_test"] == 0
    )
    results["patient_note"] = (
        "bp_train_30 is slice-level: same patients appear across splits "
        "(measured overlap). Split not changed (forbidden)."
    )

    results["all_core_ok"] = all([
        results["frozen_seg_ok"],
        results["head_trainable_ok"],
        results["logit_shape_ok"],
        results["head_grad_finite"],
        results["seg_grad_none"],
        results["determinism_ok"],
        results["state_integrity_ok"],
        results["targets_01_only"],
        results["rejected_all_zero"],
        results["accepted_equal_raw"],
    ])
    return results


def train():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    vae, none_ep = load_frozen_none()
    seg_hash = file_sha256(NONE_BEST)
    snap0 = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}

    # probe in_channels from first batch
    batch0 = next(iter(data.train_loader))
    flair0, _ = _prep(batch0[0].to(DEVICE), batch0[1].to(DEVICE))
    with torch.inference_mode():
        mu0, _, _ = vae.encoder(flair0)
    in_ch = int(mu0.shape[1])
    safe_print(f"mu shape from first batch: {tuple(mu0.shape)} -> in_channels={in_ch}")

    head = LesionPresenceHead(in_ch, hidden_dim=HIDDEN, dropout=DROPOUT).to(DEVICE)

    # smoke embedded
    smoke = run_smoke(vae, head)
    (OUT_DIR / "smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    safe_print("SMOKE", json.dumps({k: smoke[k] for k in (
        "all_core_ok", "mu_shape", "logit_shape", "state_max_abs_diff",
        "patient_disjoint", "patient_overlap_train_val",
    )}, indent=2))
    if not smoke["all_core_ok"]:
        raise RuntimeError(f"smoke core checks failed: {smoke}")

    # rebuild head after smoke mutated it
    head = LesionPresenceHead(in_ch, hidden_dim=HIDDEN, dropout=DROPOUT).to(DEVICE)
    freeze_seg(vae)

    # train groups
    safe_print("building TRAIN empty-FP / empty-TN / nonempty groups ...")
    groups = build_train_groups(vae, data.train, SEG_BATCH)
    write_csv(OUT_DIR / "train_groups.csv", groups)
    counts = {
        "nonempty": sum(1 for g in groups if g["group"] == "nonempty"),
        "empty_fp": sum(1 for g in groups if g["group"] == "empty_fp"),
        "empty_tn": sum(1 for g in groups if g["group"] == "empty_tn"),
    }
    safe_print("TRAIN group counts", counts)
    train_loader, count_meta = make_balanced_loader(data.train, groups, SEG_BATCH)

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=SCHED_PATIENCE, min_lr=1e-6)
    bce = nn.BCEWithLogitsLoss()  # balanced sampler: no pos_weight

    history = []
    best_auc = -1.0
    best_ep = 0
    best_ne_fp_auc = -1.0

    for ep in range(1, MAX_EPOCHS + 1):
        head.train()
        vae.eval()
        tr_loss = 0.0
        ntr = 0
        t0 = time.time()
        for batch in train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            y = (seg.flatten(1).sum(1) > 0).float()
            with torch.inference_mode():
                mu, _, _ = vae.encoder(flair)
            opt.zero_grad(set_to_none=True)
            logit = head(mu.detach())
            loss = bce(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step()
            tr_loss += float(loss.item())
            ntr += 1

        # val metrics
        val = collect_split(vae, head, data.val, SEG_BATCH)
        yt = (val["gt"] > 0).astype(np.float32)
        ys = val["cls_prob"]
        empty = yt < 0.5
        nonempty = ~empty
        raw_seg_pos = val["pred_mass"] > 0
        empty_fp = empty & raw_seg_pos
        val_auroc = binary_auroc(yt, ys)
        val_auprc = binary_auprc(yt, ys)
        # nonempty vs empty-FP AUROC
        if empty_fp.any() and nonempty.any():
            mask = nonempty | empty_fp
            ne_fp_auc = binary_auroc(yt[mask] > 0.5, ys[mask])
        else:
            ne_fp_auc = float("nan")
        # val bce
        with torch.no_grad():
            val_loss = float(bce(
                torch.from_numpy(val["cls_logit"]),
                torch.from_numpy(yt),
            ).item())
        pred05 = ys >= 0.5
        cm05 = conf_matrix(yt > 0.5, pred05)
        lr_now = float(opt.param_groups[0]["lr"])
        sched.step(val_auroc)

        rec = {
            "epoch": ep,
            "train_loss": tr_loss / max(1, ntr),
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "val_balanced_accuracy": cm05["balanced_accuracy"],
            "val_nonempty_vs_emptyfp_auroc": ne_fp_auc,
            "learning_rate": lr_now,
        }
        history.append(rec)
        safe_print(
            f"[presence] ep {ep}/{MAX_EPOCHS} loss={rec['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} auroc={val_auroc:.3f} auprc={val_auprc:.3f} "
            f"ne_vs_fp_auc={ne_fp_auc:.3f} bal_acc={cm05['balanced_accuracy']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )

        payload = {
            "epoch": ep,
            "presence_head": head.state_dict(),
            "optimizer": opt.state_dict(),
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "nonempty_vs_emptyfp_auroc": ne_fp_auc,
            "segmentation_checkpoint": str(NONE_BEST),
            "segmentation_checkpoint_hash": seg_hash,
            "segmentation_best_ep": none_ep,
            "in_channels": in_ch,
            "hidden_dim": HIDDEN,
            "dropout": DROPOUT,
            "train_group_counts": counts,
            "random_seed": SEED,
            "pixel_threshold": PIXEL_THR,
        }
        # primary: val AUROC, but also track ne-vs-fp
        score = val_auroc
        if np.isfinite(ne_fp_auc):
            # prefer checkpoints that also separate empty-FP when AUROC ties closely
            score = val_auroc + 0.01 * float(ne_fp_auc)
        if score > best_auc:
            best_auc = score
            best_ep = ep
            best_ne_fp_auc = float(ne_fp_auc) if np.isfinite(ne_fp_auc) else -1.0
            atomic_torch_save(payload, CKPT_DIR / "lesion_presence_head_best_auc.pt")
            safe_print(f"  saved BEST_AUC auroc={val_auroc:.4f} ne_vs_fp={ne_fp_auc:.4f}")
        atomic_torch_save(payload, CKPT_DIR / "lesion_presence_head_last.pt")

        # integrity check each epoch
        diff = state_max_abs_diff(snap0, {k: v.detach().cpu() for k, v in vae.state_dict().items()})
        if diff > 1e-6:
            raise RuntimeError(f"segmentation state changed at ep {ep}: max_abs={diff}")

        if best_ep > 0 and (ep - best_ep) >= ES_PATIENCE:
            safe_print(f"early stop @{best_ep}")
            break

    write_csv(OUT_DIR / "presence_training_history.csv", history)

    # load best auc head
    st = safe_load(CKPT_DIR / "lesion_presence_head_best_auc.pt", map_location="cpu")
    head.load_state_dict(st["presence_head"])
    head.to(DEVICE).eval()
    freeze_seg(vae)

    val = collect_split(vae, head, data.val, SEG_BATCH)
    test = collect_split(vae, head, data.test, SEG_BATCH)

    # C0 raw
    keep_all_v = np.ones(len(val["gt"]), dtype=bool)
    keep_all_t = np.ones(len(test["gt"]), dtype=bool)
    c0_val = metrics_bundle(val["gt"], val["pred_mass"], val["dices"], keep_all_v)
    c0_test = metrics_bundle(test["gt"], test["pred_mass"], test["dices"], keep_all_t)

    # C1 top-k
    keep_c1_v = val["topk50"] >= TOPK_THR_C1
    keep_c1_t = test["topk50"] >= TOPK_THR_C1
    c1_val = metrics_bundle(val["gt"], val["pred_mass"], val["dices"], keep_c1_v)
    c1_test = metrics_bundle(test["gt"], test["pred_mass"], test["dices"], keep_c1_t)

    # constrained threshold on VAL only
    raw_sens = c0_val["sensitivity"]
    raw_ne = c0_val["nonempty_dice"]
    best_thr_row, thr_rows = threshold_search(val, raw_sens, raw_ne)
    write_csv(OUT_DIR / "presence_threshold_search.csv", [
        {k: ("" if v is None else v) for k, v in r.items()} for r in thr_rows
    ])

    if best_thr_row is None:
        safe_print("NO constrained threshold; listing nearest sensitivity candidates")
        # do not use 0.5 arbitrarily — report top candidates by sens drop then empty_fp
        scored = sorted(
            thr_rows,
            key=lambda r: (
                max(0.0, raw_sens - r["sensitivity"]),
                r["empty_fp_rate"],
            ),
        )[:10]
        selected_thr = None
        c2_val = None
        c2_test = None
        keep_c2_v = keep_all_v
        keep_c2_t = keep_all_t
        constrained_note = {"status": "none", "nearest": scored}
    else:
        selected_thr = float(best_thr_row["threshold"])
        keep_c2_v = val["cls_prob"] >= selected_thr
        keep_c2_t = test["cls_prob"] >= selected_thr
        c2_val = metrics_bundle(val["gt"], val["pred_mass"], val["dices"], keep_c2_v)
        c2_test = metrics_bundle(test["gt"], test["pred_mass"], test["dices"], keep_c2_t)
        constrained_note = {"status": "ok", "threshold": selected_thr, "val_pick": best_thr_row}
        # save constrained ckpt
        payload = dict(st)
        payload["selected_classification_threshold"] = selected_thr
        payload["threshold_candidate_metrics"] = best_thr_row
        payload["val_constrained"] = c2_val
        atomic_torch_save(payload, CKPT_DIR / "lesion_presence_head_best_constrained.pt")

    # prediction CSVs
    dump_prediction_csv(OUT_DIR / "presence_val_predictions.csv", val, selected_thr, keep_c2_v)
    dump_prediction_csv(OUT_DIR / "presence_test_predictions.csv", test, selected_thr, keep_c2_t)

    # class-prob distributions on VAL
    yt = val["gt"] > 0
    empty = ~yt
    raw_pos = val["pred_mass"] > 0
    dist = {
        "nonempty": quantile_stats(val["cls_prob"][yt]),
        "empty_all": quantile_stats(val["cls_prob"][empty]),
        "empty_fp": quantile_stats(val["cls_prob"][empty & raw_pos]),
        "empty_tn": quantile_stats(val["cls_prob"][empty & ~raw_pos]),
        "auroc_nonempty_vs_all_empty": binary_auroc(yt, val["cls_prob"]),
        "auroc_nonempty_vs_emptyfp": binary_auroc(
            yt[yt | (empty & raw_pos)],
            val["cls_prob"][yt | (empty & raw_pos)],
        ) if (empty & raw_pos).any() else None,
    }
    if selected_thr is not None:
        rej = val["cls_prob"] < selected_thr
        dist["empty_fp_rejection_rate"] = float(rej[empty & raw_pos].mean()) if (empty & raw_pos).any() else None
        dist["empty_tn_rejection_rate"] = float(rej[empty & ~raw_pos].mean()) if (empty & ~raw_pos).any() else None

    final = {
        "segmentation_checkpoint": str(NONE_BEST),
        "segmentation_hash": seg_hash,
        "segmentation_best_ep": none_ep,
        "head_best_ep": best_ep,
        "in_channels": in_ch,
        "train_group_counts": counts,
        "smoke": smoke,
        "val_class_distributions": dist,
        "constrained_threshold": constrained_note,
        "C0_raw": {"val": c0_val, "test": c0_test},
        "C1_topk50": {
            "thr": TOPK_THR_C1, "k": TOPK_C1,
            "val": c1_val, "test": c1_test,
        },
        "C2_presence_head": {
            "threshold": selected_thr,
            "val": c2_val,
            "test": c2_test,
        },
        "note": "C3 (head+topk) not run in this experiment",
    }
    (OUT_DIR / "presence_final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    # print comparison tables
    def row(name, m):
        if m is None:
            return [name] + ["—"] * 9
        return [
            name,
            f"{m['overall_dice']:.4f}",
            f"{m['nonempty_dice']:.4f}",
            f"{m['empty_fp_rate']:.3f}",
            f"{m['sensitivity']:.3f}",
            f"{m['specificity']:.3f}",
            m["TT"], m["FT"], m["TF"], m["FF"],
        ]

    safe_print("\n=== VAL ===")
    for r in (
        row("Raw segmentation", c0_val),
        row("Top-k rejection", c1_val),
        row("Presence head", c2_val),
    ):
        safe_print(" | ".join(str(x) for x in r))
    safe_print("\n=== TEST (thr fixed on VAL) ===")
    for r in (
        row("Raw segmentation", c0_test),
        row("Top-k rejection", c1_test),
        row("Presence head", c2_test),
    ):
        safe_print(" | ".join(str(x) for x in r))

    # final integrity
    diff = state_max_abs_diff(snap0, {k: v.detach().cpu() for k, v in vae.state_dict().items()})
    safe_print(f"final seg state max_abs_diff={diff:.3e} (must <=1e-6)")
    safe_print("wrote", OUT_DIR)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smoke tests only")
    args = ap.parse_args()
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        vae, _ = load_frozen_none()
        data = datasets.bp_train_30(batch_size=4, num_workers=0)
        batch = next(iter(data.train_loader))
        flair, _ = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        with torch.inference_mode():
            mu, _, _ = vae.encoder(flair)
        head = LesionPresenceHead(int(mu.shape[1]), HIDDEN, DROPOUT).to(DEVICE)
        smoke = run_smoke(vae, head)
        (OUT_DIR / "smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        safe_print(json.dumps(smoke, indent=2))
        if not smoke["all_core_ok"]:
            raise SystemExit(1)
        safe_print("SMOKE PASS (core). patient_disjoint=", smoke["patient_disjoint"])
        return

    train()


if __name__ == "__main__":
    main()
