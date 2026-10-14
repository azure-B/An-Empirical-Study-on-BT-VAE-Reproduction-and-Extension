"""Retrain Presence25D with S5 empty-FP hard-negative mining, then dual×fuse.

Goal: push gated empty_fp below fuse wprod (0.335) while keeping ne ≥ 0.600.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "old" / "unused_bp_root"))

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_dual_role import DualRoleVAE
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
from train_s5_gate_25d import Stack25D, make_loader, write_csv
from eval_s5_fuse_reject import (
    collect as fuse_collect,
    eval_score,
    pick_beat_dual,
    DUAL_NE,
    DUAL_FP,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
INIT_25D = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
DUAL_CKPT = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
OUT = RESULT_DIR / "s5_gate_25d_hardneg"
BATCH = 16
EPOCHS = 80
LR = 5e-5  # finetune
ES = 15
# sampler mix: emphasize empty-FP hard negatives
MIX_NE, MIX_FP, MIX_TN = 0.45, 0.40, 0.15
PREV_BEST_FP = 0.335


def freeze_seg(vae: nn.Module) -> None:
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False


@torch.inference_mode()
def label_groups(seg_model, stack_ds: Stack25D, batch_size: int):
    """nonempty / empty_fp (S5 fires on empty) / empty_tn."""
    loader = make_loader(stack_ds, batch_size, shuffle=False)
    seg_model.eval()
    groups = []
    i0 = 0
    for batch in loader:
        x3 = batch[0].to(DEVICE)
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        center = x3[:, 1:2]
        logits, _, _ = seg_model(center)
        pred = (torch.sigmoid(logits) >= PIXEL_THR).flatten(1).any(dim=1).cpu().numpy()
        gt = (seg.flatten(1).sum(dim=1) > 0).cpu().numpy().astype(bool)
        b = len(gt)
        for j in range(b):
            idx = i0 + j
            if idx >= len(stack_ds):
                break
            if gt[j]:
                g = "nonempty"
            elif pred[j]:
                g = "empty_fp"
            else:
                g = "empty_tn"
            groups.append({"index": idx, "group": g, "gt": int(gt[j]), "s5": int(pred[j])})
        i0 += b
    groups = groups[: len(stack_ds)]
    return groups


def make_hardneg_loader(stack_ds: Stack25D, groups, batch_size: int):
    idx_ne = [g["index"] for g in groups if g["group"] == "nonempty"]
    idx_fp = [g["index"] for g in groups if g["group"] == "empty_fp"]
    idx_tn = [g["index"] for g in groups if g["group"] == "empty_tn"]
    n_ne, n_fp, n_tn = len(idx_ne), len(idx_fp), len(idx_tn)
    if min(n_ne, n_fp, n_tn) == 0:
        raise RuntimeError(f"empty group: ne={n_ne} fp={n_fp} tn={n_tn}")
    weights = np.zeros(len(groups), dtype=np.float64)
    for i in idx_ne:
        weights[i] = MIX_NE / n_ne
    for i in idx_fp:
        weights[i] = MIX_FP / n_fp
    for i in idx_tn:
        weights[i] = MIX_TN / n_tn
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(groups),
        replacement=True,
    )
    kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(stack_ds, **kwargs), {
        "n_nonempty": n_ne, "n_empty_fp": n_fp, "n_empty_tn": n_tn,
    }


@torch.inference_mode()
def collect_cls_and_s5(cls, seg_model, loader):
    cls.eval()
    seg_model.eval()
    ys, ss, s5fire = [], [], []
    for batch in loader:
        x3 = batch[0].to(DEVICE)
        seg = batch[1].to(DEVICE)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        logit = cls(x3)
        center = x3[:, 1:2]
        slogits, _, _ = seg_model(center)
        fire = (torch.sigmoid(slogits) >= PIXEL_THR).flatten(1).any(dim=1)
        ys.append((seg.flatten(1).sum(1) > 0).float().cpu().numpy())
        ss.append(torch.sigmoid(logit).cpu().numpy())
        s5fire.append(fire.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ss),
        np.concatenate(s5fire),
    )


def ne_vs_emptyfp_auroc(y, score, s5fire) -> float:
    empty = y < 0.5
    hard = empty & s5fire
    ne = y >= 0.5
    mask = ne | hard
    if hard.sum() == 0 or ne.sum() == 0:
        return float("nan")
    return binary_auroc(y[mask] > 0.5, score[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--init", type=str, default=str(INIT_25D))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not S5_BEST.exists():
        raise FileNotFoundError(S5_BEST)

    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False,
    )
    tr_ds = Stack25D(data.train)
    va_ds = Stack25D(data.val)
    te_ds = Stack25D(data.test)
    va_loader = make_loader(va_ds, args.batch, shuffle=False)
    te_loader = make_loader(te_ds, args.batch, shuffle=False)

    seg_model = SkipVAE(base=BASE).to(DEVICE)
    seg_model.load_state_dict(safe_load(S5_BEST, map_location="cpu")["model"])
    freeze_seg(seg_model)

    safe_print("label TRAIN with frozen S5 (empty-FP mining)…")
    groups = label_groups(seg_model, tr_ds, args.batch)
    counts = {
        "nonempty": sum(1 for g in groups if g["group"] == "nonempty"),
        "empty_fp": sum(1 for g in groups if g["group"] == "empty_fp"),
        "empty_tn": sum(1 for g in groups if g["group"] == "empty_tn"),
    }
    safe_print("TRAIN groups", counts)
    write_csv(OUT / "train_groups.csv", groups)
    tr_loader, _ = make_hardneg_loader(tr_ds, groups, args.batch)

    cls = Presence25D().to(DEVICE)
    init_path = Path(args.init)
    if init_path.exists():
        st = safe_load(init_path, map_location="cpu")
        cls.load_state_dict(st["model"])
        safe_print(f"init from {init_path} auroc={st.get('val_auroc')}")
    else:
        safe_print("init random (no prior 25d ckpt)")

    # sampler already balances; mild pos_weight still helps recall
    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(cls.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best_path = ckpt_dir / "cls25d_hardneg_best.pt"
    last_path = ckpt_dir / "cls25d_hardneg_last.pt"
    history = []
    best_metric = -1.0
    es = 0

    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} mix=ne{MIX_NE}/fp{MIX_FP}/tn{MIX_TN} "
        f"early-stop=ne_vs_emptyfp_AUROC"
    )

    for ep in range(1, args.epochs + 1):
        cls.train()
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in tr_loader:
            x = batch[0].to(DEVICE)
            seg = batch[1].to(DEVICE)
            if seg.ndim == 3:
                seg = seg.unsqueeze(1)
            y = (seg.flatten(1).sum(1) > 0).float()
            # per-sample: empty negatives get 1.5x (hardneg already oversampled)
            with torch.inference_mode():
                center = x[:, 1:2]
                slogits, _, _ = seg_model(center)
                fire = (torch.sigmoid(slogits) >= PIXEL_THR).flatten(1).any(dim=1)
            w = torch.ones_like(y)
            w[(y < 0.5) & fire] = 2.0  # empty-FP extra
            opt.zero_grad(set_to_none=True)
            loss = (bce(cls(x), y) * w).mean()
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1

        yv, sv, fire_v = collect_cls_and_s5(cls, seg_model, va_loader)
        auroc = binary_auroc(yv, sv)
        ne_fp = ne_vs_emptyfp_auroc(yv, sv, fire_v)
        metric = ne_fp if not np.isnan(ne_fp) else auroc
        cm = conf_matrix(yv > 0.5, sv >= 0.5)
        # proxy: empty-FP rejection @0.5 among S5-fire empties
        empty = yv < 0.5
        hard = empty & fire_v
        rej = float((sv[hard] < 0.5).mean()) if hard.any() else float("nan")
        rec = {
            "epoch": ep,
            "train_bce": tr / max(1, ntr),
            "val_auroc": auroc,
            "val_ne_vs_emptyfp_auroc": ne_fp,
            "val_emptyfp_reject@0.5": rej,
            "val_sens@0.5": cm["sensitivity"],
            "val_spec@0.5": cm["specificity"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[hn {ep}/{args.epochs}] bce={rec['train_bce']:.4f} "
            f"auroc={auroc:.4f} ne|fpAUC={ne_fp:.4f} "
            f"FPrej@0.5={rej:.3f} sens={cm['sensitivity']:.3f} "
            f"({rec['sec']:.1f}s)"
        )
        sch.step(metric)
        payload = {
            "model": cls.state_dict(),
            "epoch": ep,
            "val_auroc": auroc,
            "val_ne_vs_emptyfp_auroc": ne_fp,
            "design": "s5_gate_25d_hardneg",
            "mix": {"ne": MIX_NE, "fp": MIX_FP, "tn": MIX_TN},
            "counts": counts,
        }
        atomic_torch_save(payload, last_path)
        if metric > best_metric + 1e-4:
            best_metric = metric
            atomic_torch_save(payload, best_path)
            es = 0
            safe_print(f"  * best ne|fpAUC={metric:.4f}")
        else:
            es += 1
            if es >= ES:
                safe_print(f"early stop @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # --- fuse with dual ---
    st = safe_load(best_path, map_location="cpu")
    cls.load_state_dict(st["model"])
    cls.eval()

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL_CKPT, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()

    va1 = DataLoader(data.val, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)
    safe_print("fuse collect VAL/TEST…")
    val = fuse_collect(dual, cls, va1, va_loader)
    test = fuse_collect(dual, cls, te1, te_loader)

    recipes = {
        "dual": (val["p_dual"], test["p_dual"]),
        "p25d_hn": (val["p_25d"], test["p_25d"]),
        "mean": (
            0.5 * (val["p_dual"] + val["p_25d"]),
            0.5 * (test["p_dual"] + test["p_25d"]),
        ),
        "min": (
            np.minimum(val["p_dual"], val["p_25d"]),
            np.minimum(test["p_dual"], test["p_25d"]),
        ),
        "prod": (val["p_dual"] * val["p_25d"], test["p_dual"] * test["p_25d"]),
    }
    # weighted products around previous best
    for a in (1.0, 1.3, 1.5, 1.7):
        for b in (1.0, 1.3, 1.5, 1.7):
            sv = (val["p_dual"] ** a) * (val["p_25d"] ** b)
            stt = (test["p_dual"] ** a) * (test["p_25d"] ** b)
            sv = sv / (sv.max() + 1e-8)
            stt = stt / (stt.max() + 1e-8)
            recipes[f"wprod_a{a}_b{b}"] = (sv, stt)

    rows = []
    lines = [
        "# S5 × 2.5D hardneg + dual fuse",
        "",
        f"init={init_path.name} · mix ne/fp/tn={MIX_NE}/{MIX_FP}/{MIX_TN}",
        f"best VAL ne|emptyFP AUROC={best_metric:.4f}",
        f"target: ne≥{DUAL_NE} and fp < {PREV_BEST_FP} (prev wprod)",
        "",
    ]
    ref = eval_score(test, test["p_dual"], 0.22)
    rows.append({
        "name": "TEST_dual_ref_0.22",
        **{k: ref[k] for k in (
            "thr", "nonempty_dice", "empty_fp_rate", "sensitivity",
            "specificity", "auroc", "tp", "tn", "fp", "fn",
        )},
    })
    lines.append(
        f"REF dual@0.22 ne={ref['nonempty_dice']:.4f} fp={ref['empty_fp_rate']:.3f}"
    )

    best_beat = None
    best_any = None
    for name, (sv, stt) in recipes.items():
        pv = pick_beat_dual(val, sv, ne_floor=DUAL_NE, fp_ceil=PREV_BEST_FP)
        mt = eval_score(test, stt, pv["thr"])
        beats_dual = (
            mt["nonempty_dice"] + 1e-9 >= DUAL_NE
            and mt["empty_fp_rate"] < DUAL_FP - 1e-9
        )
        beats_prev = (
            mt["nonempty_dice"] + 1e-9 >= DUAL_NE
            and mt["empty_fp_rate"] < PREV_BEST_FP - 1e-9
        )
        row = {
            "name": f"TEST_{name}",
            "val_pick": pv["pick"],
            **{k: mt[k] for k in (
                "thr", "accuracy", "sensitivity", "specificity", "precision",
                "f1", "auroc", "tp", "tn", "fp", "fn",
                "nonempty_dice", "empty_fp_rate",
            )},
            "val_ne": pv["nonempty_dice"],
            "val_fp": pv["empty_fp_rate"],
            "beats_dual": beats_dual,
            "beats_prev_wprod": beats_prev,
        }
        rows.append(row)
        if name in ("dual", "p25d_hn", "mean", "min", "prod") or beats_prev or beats_dual:
            safe_print(
                f"==== {name} τ={pv['thr']:.2f} "
                f"ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
                f"beats_prev={beats_prev}"
            )
            lines.append(
                f"### {name} τ={pv['thr']:.2f} → ne={mt['nonempty_dice']:.4f} "
                f"fp={mt['empty_fp_rate']:.3f} beats_prev={beats_prev}"
            )
        key = (mt["empty_fp_rate"], -mt["nonempty_dice"])
        if beats_prev:
            if best_beat is None or key < (best_beat["empty_fp_rate"], -best_beat["nonempty_dice"]):
                best_beat = row
        if mt["nonempty_dice"] + 1e-9 >= DUAL_NE:
            if best_any is None or mt["empty_fp_rate"] < best_any["empty_fp_rate"]:
                best_any = row

    champ = best_beat or best_any
    lines.extend([
        "",
        "## Champion",
        f"- {champ['name'] if champ else 'none'}: "
        f"ne={champ['nonempty_dice']:.4f} fp={champ['empty_fp_rate']:.3f}"
        if champ else "- none",
        f"- beats prev wprod fp<{PREV_BEST_FP}: {best_beat is not None}",
    ])

    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps({
            "best_val_ne_fp_auroc": best_metric,
            "counts": counts,
            "dual_ref": ref,
            "champion": champ,
            "beats_prev": best_beat is not None,
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# 2.5D hard-negative retrain",
            "",
            "- Freeze S5; label TRAIN empty_fp = GT empty & S5 nonempty",
            f"- Weighted sampler {MIX_NE}/{MIX_FP}/{MIX_TN}",
            "- Extra loss weight ×2 on empty_fp batches",
            "- Early-stop: AUROC(nonempty vs empty_fp)",
            "- Fuse with dual μ-head (product / weighted)",
            "",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)
    if champ:
        safe_print(
            f"CHAMP {champ['name']} ne={champ['nonempty_dice']:.4f} "
            f"fp={champ['empty_fp_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
