"""Lower empty-FP beyond dual-role@0.22 by fusing dual μ-head + 2.5D + S5 cues.

No SEG retrain. Recipes on VAL → TEST:
  - dual / 25d alone
  - mean / min / product fuse
  - logistic rejector on [p_dual, p_25d, mass_n, maxprob] (empty-FP aware)
Primary target: TEST ne ≥ 0.600 and empty_fp < 0.383.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    safe_load,
    safe_print,
)
from train_s5_gate_25d import Stack25D, make_loader

OUT = RESULT_DIR / "s5_fuse_reject"
DUAL_CKPT = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
CLS25_CKPT = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
DUAL_NE, DUAL_FP = 0.600, 0.383
BATCH = 16


class LogitFuse(nn.Module):
    """Tiny linear fuse: logit = w·x + b → score = sigmoid."""

    def __init__(self, n_in: int = 4):
        super().__init__()
        self.fc = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)


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


@torch.inference_mode()
def collect(dual, cls25, loader_1ch, loader_3ch):
    """Align 1ch dual/S5 loader with 3ch 25d loader (same order, no shuffle)."""
    dual.eval()
    cls25.eval()
    bags = {
        "gt": [], "mass": [], "dice": [], "maxprob": [],
        "p_dual": [], "p_25d": [],
    }
    it3 = iter(loader_3ch)
    for batch in loader_1ch:
        b3 = next(it3)
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        x3 = b3[0].to(DEVICE)
        # presence from dual head + S5 mask from skip line
        skip_logits = dual.skip_logits_det(flair)
        _plain, plog, _mu, _lv = dual.cls_forward_det(flair)
        p_dual = torch.sigmoid(plog)
        p_25d = torch.sigmoid(cls25(x3))
        prob = torch.sigmoid(skip_logits)
        maxprob = prob.flatten(1).amax(dim=1)
        pred = (prob >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["maxprob"].append(maxprob.cpu().numpy())
        bags["p_dual"].append(p_dual.cpu().numpy())
        bags["p_25d"].append(p_25d.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def features(bag) -> np.ndarray:
    mass = bag["mass"]
    mass_n = np.log1p(mass) / (np.log1p(mass).max() + 1e-8)
    return np.stack([
        bag["p_dual"],
        bag["p_25d"],
        mass_n,
        bag["maxprob"],
    ], axis=1).astype(np.float32)


def eval_score(bag, score, thr) -> dict:
    keep = score >= thr
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
        "auroc": binary_auroc(yt, score),
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
    }


def pick_beat_dual(val, score, ne_floor=DUAL_NE, fp_ceil=DUAL_FP):
    """Prefer VAL: ne≥floor & fp≤ceil with min fp; else max ne-0.5*fp among ne≥floor; else joint."""
    beat, ne_ok, joint = None, None, None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_score(val, score, thr)
        j = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
        m["_j"] = j
        if joint is None or j > joint["_j"]:
            joint = dict(m)
        if m["nonempty_dice"] + 1e-9 >= ne_floor:
            if ne_ok is None or m["empty_fp_rate"] < ne_ok["empty_fp_rate"] - 1e-12:
                ne_ok = dict(m)
            elif abs(m["empty_fp_rate"] - ne_ok["empty_fp_rate"]) < 1e-12 and m["nonempty_dice"] > ne_ok["nonempty_dice"]:
                ne_ok = dict(m)
            if m["empty_fp_rate"] <= fp_ceil + 1e-9:
                key = (-m["empty_fp_rate"], m["nonempty_dice"])
                if beat is None or key > beat["_key"]:
                    beat = dict(m)
                    beat["_key"] = key
    if beat is not None:
        beat["pick"] = "beat_dual"
        return beat
    if ne_ok is not None:
        ne_ok["pick"] = "ne_floor_min_fp"
        return ne_ok
    joint["pick"] = "joint"
    return joint


def train_rejector(val, epochs=80, lr=5e-3):
    """Empty-aware logistic: overweight empty GT that S5 fires (empty FP hard cases)."""
    x = torch.from_numpy(features(val)).to(DEVICE)
    y = torch.from_numpy((val["gt"] > 0).astype(np.float32)).to(DEVICE)
    empty = val["gt"] == 0
    s5_fire = val["mass"] > 0
    hard = empty & s5_fire
    w = np.ones(len(y), dtype=np.float32)
    w[hard] = 3.0          # punish keeping empty-FP
    w[val["gt"] > 0] = 2.0  # keep lesion recall
    w = torch.from_numpy(w).to(DEVICE)

    model = LogitFuse(4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logit = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y, weight=w)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        score = torch.sigmoid(model(x)).cpu().numpy()
    return model, score


@torch.inference_mode()
def rejector_score(model, bag):
    x = torch.from_numpy(features(bag)).to(DEVICE)
    return torch.sigmoid(model(x)).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (DUAL_CKPT, CLS25_CKPT):
        if not p.exists():
            raise FileNotFoundError(p)

    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    va3 = make_loader(Stack25D(data.val), BATCH, shuffle=False)
    te3 = make_loader(Stack25D(data.test), BATCH, shuffle=False)

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    st = safe_load(DUAL_CKPT, map_location="cpu")
    dual.load_state_dict(st["model"])
    dual.freeze_skip()
    dual.eval()

    cls25 = Presence25D().to(DEVICE)
    st25 = safe_load(CLS25_CKPT, map_location="cpu")
    cls25.load_state_dict(st25["model"])
    cls25.eval()

    safe_print("collect VAL/TEST scores…")
    val = collect(dual, cls25, va1, va3)
    test = collect(dual, cls25, te1, te3)

    recipes = {
        "dual": val["p_dual"],
        "p25d": val["p_25d"],
        "mean": 0.5 * (val["p_dual"] + val["p_25d"]),
        "min": np.minimum(val["p_dual"], val["p_25d"]),
        "prod": val["p_dual"] * val["p_25d"],
        "max": np.maximum(val["p_dual"], val["p_25d"]),
    }
    recipes_t = {
        "dual": test["p_dual"],
        "p25d": test["p_25d"],
        "mean": 0.5 * (test["p_dual"] + test["p_25d"]),
        "min": np.minimum(test["p_dual"], test["p_25d"]),
        "prod": test["p_dual"] * test["p_25d"],
        "max": np.maximum(test["p_dual"], test["p_25d"]),
    }

    safe_print("train empty-aware logistic rejector on VAL…")
    rej, val_rej = train_rejector(val)
    recipes["rejector"] = val_rej
    recipes_t["rejector"] = rejector_score(rej, test)
    torch.save({"model": rej.state_dict(), "design": "logit_fuse_4"}, OUT / "rejector.pt")

    rows, lines = [], [
        "# S5 fuse + empty rejector",
        "",
        f"Target: beat dual@0.22 (ne≥{DUAL_NE}, fp<{DUAL_FP})",
        f"dual VAL AUROC={binary_auroc(val['gt']>0, val['p_dual']):.4f} | "
        f"25d={binary_auroc(val['gt']>0, val['p_25d']):.4f} | "
        f"rejector={binary_auroc(val['gt']>0, recipes['rejector']):.4f}",
        "",
    ]

    best_test = None
    for name, score_v in recipes.items():
        score_t = recipes_t[name]
        pv = pick_beat_dual(val, score_v)
        mt = eval_score(test, score_t, pv["thr"])
        beats = (
            mt["nonempty_dice"] + 1e-9 >= DUAL_NE
            and mt["empty_fp_rate"] < DUAL_FP - 1e-9
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
            "beats_dual": beats,
        }
        rows.append(row)
        safe_print(
            f"==== {name} pick={pv['pick']} τ={pv['thr']:.2f} "
            f"TEST ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
            f"sens={mt['sensitivity']:.3f} beats={beats}"
        )
        lines.append(
            f"### {name} ({pv['pick']}) τ={pv['thr']:.2f}\n"
            f"- VAL ne={pv['nonempty_dice']:.4f} fp={pv['empty_fp_rate']:.3f}\n"
            f"- TEST ne={mt['nonempty_dice']:.4f} fp={mt['empty_fp_rate']:.3f} "
            f"sens={mt['sensitivity']:.3f} FN={mt['fn']} beats_dual={beats}"
        )
        if beats and (best_test is None or mt["empty_fp_rate"] < best_test["empty_fp_rate"]):
            best_test = dict(row)

    # also: dual@0.22 reference recompute
    ref = eval_score(test, test["p_dual"], 0.22)
    rows.insert(0, {
        "name": "TEST_dual_ref_0.22",
        **{k: ref[k] for k in (
            "thr", "accuracy", "sensitivity", "specificity", "precision",
            "f1", "auroc", "tp", "tn", "fp", "fn",
            "nonempty_dice", "empty_fp_rate",
        )},
        "beats_dual": False,
    })
    lines.insert(4, (
        f"REF dual@0.22 TEST ne={ref['nonempty_dice']:.4f} "
        f"fp={ref['empty_fp_rate']:.3f}"
    ))

    if best_test is None:
        # report closest: highest ne among fp < dual, or lowest fp among ne≥dual
        cand = [r for r in rows if r["name"].startswith("TEST_") and r["name"] != "TEST_dual_ref_0.22"]
        by_fp = [r for r in cand if r["nonempty_dice"] + 1e-9 >= DUAL_NE]
        if by_fp:
            best_test = min(by_fp, key=lambda r: r["empty_fp_rate"])
            note = "ne≥dual, min fp (fp may not beat)"
        else:
            best_test = max(cand, key=lambda r: r["nonempty_dice"] - 0.5 * r["empty_fp_rate"])
            note = "no ne≥dual; best joint"
        lines.extend(["", f"## Best effort ({note})", f"- {best_test['name']}"])
    else:
        lines.extend(["", "## Beat dual", f"- {best_test['name']}"])

    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps({"rows": rows, "best": best_test, "dual_ref": ref}, indent=2),
        encoding="utf-8",
    )
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
