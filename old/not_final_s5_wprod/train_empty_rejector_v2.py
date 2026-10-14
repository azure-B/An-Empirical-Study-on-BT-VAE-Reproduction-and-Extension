"""Stage-2 empty rejector: train only on nonempty vs S5-empty-FP hard cases.

Uses frozen dual μ-head + original Presence25D + S5 cues.
keep = sigmoid(MLP(features)); pick τ for ne≥0.60 & min empty_fp.
Also report AND with dual and product with dual.
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
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_lesion_presence import binary_auroc
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
from eval_s5_fuse_reject import collect, eval_score, pick_beat_dual, DUAL_NE, DUAL_FP

OUT = RESULT_DIR / "s5_empty_rejector_v2"
DUAL_CKPT = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
CLS25_CKPT = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
PREV_FP = 0.335
BATCH = 16
EPOCHS = 120
ES = 20


class RejectMLP(nn.Module):
    def __init__(self, n_in: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def feats(bag) -> np.ndarray:
    mass = bag["mass"]
    mass_n = np.log1p(mass) / (np.log1p(mass.max()) + 1e-8)
    # interaction helps AND-like behavior
    prod = bag["p_dual"] * bag["p_25d"]
    return np.stack([
        bag["p_dual"], bag["p_25d"], prod, mass_n, bag["maxprob"],
    ], axis=1).astype(np.float32)


def hard_mask(bag) -> np.ndarray:
    """Train on nonempty ∪ (empty & S5-fire)."""
    y = bag["gt"] > 0
    empty_fp = (bag["gt"] == 0) & (bag["mass"] > 0)
    return y | empty_fp


def train_mlp(x, y, hard, epochs=EPOCHS):
    # only hard subset
    xh, yh = x[hard], y[hard]
    ne = yh > 0.5
    fp = ~ne
    w = np.zeros(len(yh), dtype=np.float64)
    w[ne] = 0.5 / max(1, ne.sum())
    w[fp] = 0.5 / max(1, fp.sum())
    sampler = WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double),
        num_samples=len(yh),
        replacement=True,
    )
    ds = TensorDataset(torch.from_numpy(xh), torch.from_numpy(yh.astype(np.float32)))
    loader = DataLoader(ds, batch_size=64, sampler=sampler)
    model = RejectMLP(n_in=x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_auc, es = None, -1.0, 0
    # fixed val = all hard samples (small); use epoch shuffle train only
    xv = torch.from_numpy(xh).to(DEVICE)
    yv = yh
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            s = torch.sigmoid(model(xv)).cpu().numpy()
        auc = binary_auroc(yv > 0.5, s)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es = 0
        else:
            es += 1
            if es >= ES:
                safe_print(f"rejector early stop @ {ep} best_auc={best_auc:.4f}")
                break
        if ep % 10 == 0 or ep == 1:
            safe_print(f"[rej {ep}] hardAUC={auc:.4f} best={best_auc:.4f}")
    model.load_state_dict(best_state)
    return model, best_auc


@torch.inference_mode()
def score_all(model, bag):
    x = torch.from_numpy(feats(bag)).to(DEVICE)
    return torch.sigmoid(model(x)).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    # train features too for rejector fit
    tr1 = DataLoader(data.train, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    va3 = make_loader(Stack25D(data.val), BATCH, False)
    te3 = make_loader(Stack25D(data.test), BATCH, False)
    tr3 = make_loader(Stack25D(data.train), BATCH, False)

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL_CKPT, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()
    cls25 = Presence25D().to(DEVICE)
    cls25.load_state_dict(safe_load(CLS25_CKPT, map_location="cpu")["model"])
    cls25.eval()

    safe_print("collect TRAIN/VAL/TEST…")
    train = collect(dual, cls25, tr1, tr3)
    val = collect(dual, cls25, va1, va3)
    test = collect(dual, cls25, te1, te3)

    xtr, ytr = feats(train), train["gt"] > 0
    hard_tr = hard_mask(train)
    safe_print(
        f"TRAIN hard n={hard_tr.sum()} ne={(ytr & hard_tr).sum()} "
        f"empty_fp={(~ytr & hard_tr).sum()}"
    )

    t0 = time.time()
    model, hauc = train_mlp(xtr, ytr.astype(np.float32), hard_tr)
    safe_print(f"rejector trained in {time.time()-t0:.1f}s hardAUC={hauc:.4f}")
    atomic_torch_save({"model": model.state_dict(), "hard_auc": hauc}, OUT / "rejector_mlp.pt")

    val["p_rej"] = score_all(model, val)
    test["p_rej"] = score_all(model, test)

    recipes = {
        "rejector": (val["p_rej"], test["p_rej"]),
        "dual": (val["p_dual"], test["p_dual"]),
        "prod_dual_25": (val["p_dual"] * val["p_25d"], test["p_dual"] * test["p_25d"]),
        "prod_dual_rej": (val["p_dual"] * val["p_rej"], test["p_dual"] * test["p_rej"]),
        "prod_25_rej": (val["p_25d"] * val["p_rej"], test["p_25d"] * test["p_rej"]),
        "prod3": (
            val["p_dual"] * val["p_25d"] * val["p_rej"],
            test["p_dual"] * test["p_25d"] * test["p_rej"],
        ),
        "min_dual_rej": (
            np.minimum(val["p_dual"], val["p_rej"]),
            np.minimum(test["p_dual"], test["p_rej"]),
        ),
        "wprod_prev": (
            ((val["p_dual"] ** 1.5) * (val["p_25d"] ** 1.3)),
            ((test["p_dual"] ** 1.5) * (test["p_25d"] ** 1.3)),
        ),
        "wprod_rej": (
            ((val["p_dual"] ** 1.5) * (val["p_25d"] ** 1.3) * val["p_rej"]),
            ((test["p_dual"] ** 1.5) * (test["p_25d"] ** 1.3) * test["p_rej"]),
        ),
    }
    # normalize weighted
    for k in ("wprod_prev", "wprod_rej"):
        sv, st = recipes[k]
        recipes[k] = (sv / (sv.max() + 1e-8), st / (st.max() + 1e-8))

    rows, lines = [], [
        "# Empty rejector v2 (hardneg MLP)",
        "",
        f"train on nonempty ∪ S5-empty-FP · hardAUC={hauc:.4f}",
        f"target: ne≥{DUAL_NE} fp<{PREV_FP}",
        "",
    ]
    ref = eval_score(test, test["p_dual"], 0.22)
    rows.append({"name": "TEST_dual_ref_0.22", **{k: ref[k] for k in (
        "thr", "nonempty_dice", "empty_fp_rate", "sensitivity", "fn", "fp", "auroc")}})
    lines.append(f"REF dual@0.22 ne={ref['nonempty_dice']:.4f} fp={ref['empty_fp_rate']:.3f}")

    champ = None
    for name, (sv, st) in recipes.items():
        pv = pick_beat_dual(val, sv, ne_floor=DUAL_NE, fp_ceil=PREV_FP)
        mt = eval_score(test, st, pv["thr"])
        beats_prev = (
            mt["nonempty_dice"] + 1e-9 >= DUAL_NE
            and mt["empty_fp_rate"] < PREV_FP - 1e-9
        )
        beats_dual = (
            mt["nonempty_dice"] + 1e-9 >= DUAL_NE
            and mt["empty_fp_rate"] < DUAL_FP - 1e-9
        )
        row = {
            "name": f"TEST_{name}",
            "val_pick": pv["pick"],
            **{k: mt[k] for k in (
                "thr", "accuracy", "sensitivity", "specificity", "f1", "auroc",
                "tp", "tn", "fp", "fn", "nonempty_dice", "empty_fp_rate",
            )},
            "beats_prev": beats_prev,
            "beats_dual": beats_dual,
        }
        rows.append(row)
        safe_print(
            f"==== {name} τ={pv['thr']:.2f} ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} beats_prev={beats_prev}"
        )
        lines.append(
            f"### {name} τ={pv['thr']:.2f} → ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} beats_prev={beats_prev}"
        )
        if beats_prev or beats_dual:
            key = (mt["empty_fp_rate"], -mt["nonempty_dice"])
            if champ is None or key < (champ["empty_fp_rate"], -champ["nonempty_dice"]):
                champ = row

    if champ is None:
        # fallback: best among ne≥0.60
        cands = [r for r in rows if r.get("nonempty_dice", 0) + 1e-9 >= DUAL_NE]
        if cands:
            champ = min(cands, key=lambda r: r["empty_fp_rate"])
        else:
            champ = min(
                [r for r in rows if r["name"] != "TEST_dual_ref_0.22"],
                key=lambda r: r["empty_fp_rate"] - 0.5 * r["nonempty_dice"],
            )

    lines.extend([
        "",
        "## Champion",
        f"- {champ['name']}: ne={champ['nonempty_dice']:.4f} "
        f"fp={champ['empty_fp_rate']:.3f} beats_prev={champ.get('beats_prev')}",
    ])
    write_csv(OUT / "final_metrics.csv", rows)
    (OUT / "report.json").write_text(
        json.dumps({"hard_auc": hauc, "champion": champ, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print("CHAMP", champ["name"], champ["nonempty_dice"], champ["empty_fp_rate"])
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
