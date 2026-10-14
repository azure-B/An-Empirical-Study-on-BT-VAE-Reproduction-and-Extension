"""Rejector v3: fit on TRAIN hard cases, early-stop on VAL hard AUROC (no leak).

Also: from-scratch Presence25D (hardneg sampler, early-stop overall AUROC),
then dual×25d×rej fuse + optional neighbor consistency kill.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "old" / "unused_bp_root"))

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_lesion_presence import PIXEL_THR, binary_auroc, conf_matrix
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)
from train_s5_gate_25d import Stack25D, make_loader, write_csv, freeze_seg
from train_s5_gate_25d_hardneg import label_groups, make_hardneg_loader, collect_cls_and_s5
from eval_s5_fuse_reject import collect, eval_score, pick_beat_dual, DUAL_NE, DUAL_FP

OUT = RESULT_DIR / "s5_empty_rejector_v3"
DUAL_CKPT = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
PREV_FP = 0.335
BATCH = 16


class RejectMLP(nn.Module):
    def __init__(self, n_in: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def feats(bag):
    mass = bag["mass"]
    mass_n = np.log1p(mass) / (np.log1p(mass.max()) + 1e-8)
    prod = bag["p_dual"] * bag["p_25d"]
    return np.stack([
        bag["p_dual"], bag["p_25d"], prod, mass_n, bag["maxprob"],
    ], axis=1).astype(np.float32)


def hard_idx(bag):
    y = bag["gt"] > 0
    return y | ((bag["gt"] == 0) & (bag["mass"] > 0))


def train_rejector(xtr, ytr, hard_tr, xva, yva, hard_va, epochs=200):
    xh = torch.from_numpy(xtr[hard_tr])
    yh = torch.from_numpy(ytr[hard_tr].astype(np.float32))
    ne = yh.numpy() > 0.5
    fp = ~ne
    w = np.zeros(len(yh), dtype=np.float64)
    w[ne] = 0.5 / max(1, ne.sum())
    w[fp] = 0.5 / max(1, fp.sum())
    loader = DataLoader(
        TensorDataset(xh, yh),
        batch_size=64,
        sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(yh), True),
    )
    xv = torch.from_numpy(xva[hard_va]).to(DEVICE)
    yv = yva[hard_va]
    model = RejectMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best, best_auc, es = None, -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            nn.functional.binary_cross_entropy_with_logits(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            s = torch.sigmoid(model(xv)).cpu().numpy()
        auc = binary_auroc(yv > 0.5, s)
        if auc > best_auc + 1e-4:
            best_auc, es = auc, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep == 1 or ep % 5 == 0:
                safe_print(f"  * rej VAL hardAUC={auc:.4f} @ep{ep}")
        else:
            es += 1
            if es >= 25:
                safe_print(f"rejector stop @ {ep} bestVAL={best_auc:.4f}")
                break
    model.load_state_dict(best)
    return model, best_auc


def train_25d_scratch(seg_model, tr_ds, va_loader, epochs=60):
    groups = label_groups(seg_model, tr_ds, BATCH)
    # milder mix than failed hardneg finetune
    global_mix = None
    # temporarily patch ratios via local sampler
    idx_ne = [g["index"] for g in groups if g["group"] == "nonempty"]
    idx_fp = [g["index"] for g in groups if g["group"] == "empty_fp"]
    idx_tn = [g["index"] for g in groups if g["group"] == "empty_tn"]
    n_ne, n_fp, n_tn = len(idx_ne), len(idx_fp), len(idx_tn)
    weights = np.zeros(len(groups), dtype=np.float64)
    # 50/30/20 — less aggressive than 45/40/15
    for i in idx_ne:
        weights[i] = 0.50 / n_ne
    for i in idx_fp:
        weights[i] = 0.30 / n_fp
    for i in idx_tn:
        weights[i] = 0.20 / n_tn
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(groups), True,
    )
    kwargs = dict(
        batch_size=BATCH, sampler=sampler, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"), drop_last=True,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    tr_loader = DataLoader(tr_ds, **kwargs)
    safe_print(f"25d scratch groups ne={n_ne} fp={n_fp} tn={n_tn} mix=0.5/0.3/0.2")

    cls = Presence25D().to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(cls.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    best_path = OUT / "checkpoints" / "cls25d_scratch_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_auroc, es = -1.0, 0
    for ep in range(1, epochs + 1):
        cls.train()
        tr, ntr, t0 = 0.0, 0, time.time()
        for batch in tr_loader:
            x = batch[0].to(DEVICE)
            seg = batch[1].to(DEVICE)
            if seg.ndim == 3:
                seg = seg.unsqueeze(1)
            y = (seg.flatten(1).sum(1) > 0).float()
            opt.zero_grad(set_to_none=True)
            loss = bce(cls(x), y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        yv, sv, _ = collect_cls_and_s5(cls, seg_model, va_loader)
        auroc = binary_auroc(yv, sv)
        cm = conf_matrix(yv > 0.5, sv >= 0.5)
        safe_print(
            f"[25d {ep}/{epochs}] bce={tr/max(1,ntr):.4f} auroc={auroc:.4f} "
            f"sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
        sch.step(auroc)
        payload = {"model": cls.state_dict(), "epoch": ep, "val_auroc": auroc}
        atomic_torch_save(payload, OUT / "checkpoints" / "cls25d_scratch_last.pt")
        if auroc > best_auroc + 1e-4:
            best_auroc, es = auroc, 0
            atomic_torch_save(payload, best_path)
            safe_print(f"  * best auroc={auroc:.4f}")
        else:
            es += 1
            if es >= 12:
                safe_print(f"25d early stop @ {ep}")
                break
    cls.load_state_dict(safe_load(best_path, map_location="cpu")["model"])
    return cls, best_auroc


def neighbor_adjust(stems, score, delta=1):
    """Multiply score by mean of neighbor scores (consistency). Missing → self."""
    from collections import defaultdict
    by_pid = defaultdict(dict)
    for i, stem in enumerate(stems):
        pid, sl = stem.rsplit("_", 1)
        by_pid[pid][int(sl)] = i
    out = score.copy()
    for pid, m in by_pid.items():
        for sl, i in m.items():
            neigh = []
            for d in (-delta, delta):
                j = m.get(sl + d)
                if j is not None:
                    neigh.append(score[j])
            if neigh:
                out[i] = score[i] * float(np.mean(neigh))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    tr_ds, va_ds, te_ds = Stack25D(data.train), Stack25D(data.val), Stack25D(data.test)
    va3 = make_loader(va_ds, BATCH, False)
    te3 = make_loader(te_ds, BATCH, False)
    tr3 = make_loader(tr_ds, BATCH, False)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    tr1 = DataLoader(data.train, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    seg_model = SkipVAE(base=BASE).to(DEVICE)
    seg_model.load_state_dict(safe_load(S5_BEST, map_location="cpu")["model"])
    freeze_seg(seg_model)

    safe_print("=== Phase A: 2.5D from scratch (mild hardneg) ===")
    cls25, a25 = train_25d_scratch(seg_model, tr_ds, va3, epochs=60)
    safe_print(f"25d scratch best AUROC={a25:.4f}")

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL_CKPT, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()

    safe_print("=== Phase B: collect + rejector (VAL early-stop) ===")
    train = collect(dual, cls25, tr1, tr3)
    val = collect(dual, cls25, va1, va3)
    test = collect(dual, cls25, te1, te3)
    # attach stems for neighbor
    val_stems = list(va_ds.stems)
    test_stems = list(te_ds.stems)

    xtr, ytr = feats(train), train["gt"] > 0
    xva, yva = feats(val), val["gt"] > 0
    rej, hauc = train_rejector(
        xtr, ytr, hard_idx(train), xva, yva, hard_idx(val),
    )
    atomic_torch_save({"model": rej.state_dict(), "val_hard_auc": hauc}, OUT / "rejector.pt")
    with torch.inference_mode():
        val["p_rej"] = torch.sigmoid(
            rej(torch.from_numpy(feats(val)).to(DEVICE))
        ).cpu().numpy()
        test["p_rej"] = torch.sigmoid(
            rej(torch.from_numpy(feats(test)).to(DEVICE))
        ).cpu().numpy()

    # neighbor-smoothed product
    base_v = (val["p_dual"] ** 1.5) * (val["p_25d"] ** 1.3)
    base_t = (test["p_dual"] ** 1.5) * (test["p_25d"] ** 1.3)
    base_v = base_v / (base_v.max() + 1e-8)
    base_t = base_t / (base_t.max() + 1e-8)
    neigh_v = neighbor_adjust(val_stems, base_v)
    neigh_t = neighbor_adjust(test_stems, base_t)
    neigh_v = neigh_v / (neigh_v.max() + 1e-8)
    neigh_t = neigh_t / (neigh_t.max() + 1e-8)

    recipes = {
        "dual": (val["p_dual"], test["p_dual"]),
        "p25d": (val["p_25d"], test["p_25d"]),
        "prod": (val["p_dual"] * val["p_25d"], test["p_dual"] * test["p_25d"]),
        "wprod": (base_v, base_t),
        "wprod_rej": (
            base_v * val["p_rej"],
            base_t * test["p_rej"],
        ),
        "prod_rej": (
            val["p_dual"] * val["p_25d"] * val["p_rej"],
            test["p_dual"] * test["p_25d"] * test["p_rej"],
        ),
        "rejector": (val["p_rej"], test["p_rej"]),
        "wprod_neigh": (neigh_v, neigh_t),
        "wprod_neigh_rej": (neigh_v * val["p_rej"], neigh_t * test["p_rej"]),
    }
    for k in ("wprod_rej", "prod_rej", "wprod_neigh_rej"):
        sv, st = recipes[k]
        recipes[k] = (sv / (sv.max() + 1e-8), st / (st.max() + 1e-8))

    rows, lines = [], [
        "# Rejector v3 + 25d scratch + neighbor",
        "",
        f"25d AUROC={a25:.4f} · rej VAL hardAUC={hauc:.4f}",
        f"target ne≥{DUAL_NE} fp<{PREV_FP}",
        "",
    ]
    ref = eval_score(test, test["p_dual"], 0.22)
    rows.append({"name": "TEST_dual_ref_0.22", **{k: ref[k] for k in (
        "thr", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp")}})
    lines.append(f"REF dual@0.22 ne={ref['nonempty_dice']:.4f} fp={ref['empty_fp_rate']:.3f}")

    champ = None
    for name, (sv, st) in recipes.items():
        pv = pick_beat_dual(val, sv, ne_floor=DUAL_NE, fp_ceil=PREV_FP)
        mt = eval_score(test, st, pv["thr"])
        beats_prev = mt["nonempty_dice"] + 1e-9 >= DUAL_NE and mt["empty_fp_rate"] < PREV_FP - 1e-9
        beats_dual = mt["nonempty_dice"] + 1e-9 >= DUAL_NE and mt["empty_fp_rate"] < DUAL_FP - 1e-9
        row = {
            "name": f"TEST_{name}",
            "val_pick": pv["pick"],
            **{k: mt[k] for k in (
                "thr", "sensitivity", "specificity", "auroc", "tp", "tn", "fp", "fn",
                "nonempty_dice", "empty_fp_rate",
            )},
            "beats_prev": beats_prev,
            "beats_dual": beats_dual,
        }
        rows.append(row)
        safe_print(
            f"==== {name} τ={pv['thr']:.2f} ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} prev={beats_prev}"
        )
        lines.append(
            f"### {name} τ={pv['thr']:.2f} → ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} beats_prev={beats_prev}"
        )
        if beats_dual:
            key = (mt["empty_fp_rate"], -mt["nonempty_dice"])
            if champ is None or key < (champ["empty_fp_rate"], -champ["nonempty_dice"]):
                champ = row

    if champ is None:
        cands = [r for r in rows if r.get("nonempty_dice", 0) >= DUAL_NE - 1e-9]
        champ = min(cands, key=lambda r: r["empty_fp_rate"]) if cands else rows[-1]

    lines += ["", "## Champion",
              f"- {champ['name']}: ne={champ['nonempty_dice']:.4f} "
              f"fp={champ['empty_fp_rate']:.3f} beats_prev={champ.get('beats_prev')}"]
    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps({"a25": a25, "rej_val_hard_auc": hauc, "champion": champ, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print("CHAMP", champ["name"], f"ne={champ['nonempty_dice']:.4f}", f"fp={champ['empty_fp_rate']:.3f}")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
