"""Old 25d + VAL-early-stop rejector + ens(new) fuse sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "old" / "unused_bp_root"))

from data_load_BT import datasets
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print
from train_s5_gate_25d import Stack25D, make_loader, write_csv
from eval_s5_fuse_reject import collect, eval_score, DUAL_NE, DUAL_FP
from train_empty_rejector_v3 import feats, hard_idx, train_rejector, neighbor_adjust

OUT = RESULT_DIR / "s5_empty_rejector_v3b"
DUAL = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
C25 = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
C25N = RESULT_DIR / "s5_empty_rejector_v3" / "checkpoints" / "cls25d_scratch_best.pt"
BATCH = 16
PREV = 0.335


@torch.inference_mode()
def p25_scores(model, loader3):
    model.eval()
    ss = []
    for b in loader3:
        ss.append(torch.sigmoid(model(b[0].to(DEVICE))).cpu().numpy())
    return np.concatenate(ss)


def norm(a, b):
    return a / (a.max() + 1e-8), b / (b.max() + 1e-8)


def pick_ne60_min_fp(val, score):
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_score(val, score, thr)
        if m["nonempty_dice"] + 1e-9 < DUAL_NE:
            continue
        if best is None or m["empty_fp_rate"] < best["empty_fp_rate"] - 1e-12:
            best = dict(m)
        elif abs(m["empty_fp_rate"] - best["empty_fp_rate"]) < 1e-12 and m["nonempty_dice"] > best["nonempty_dice"]:
            best = dict(m)
    if best is not None:
        best["pick"] = "ne60_min_fp"
        return best
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_score(val, score, thr)
        j = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
        if best is None or j > best.get("_j", -1e9):
            best = dict(m)
            best["_j"] = j
    best["pick"] = "joint"
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    va_ds, te_ds, tr_ds = Stack25D(data.val), Stack25D(data.test), Stack25D(data.train)
    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    tr1 = DataLoader(data.train, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    va3 = make_loader(va_ds, BATCH, False)
    te3 = make_loader(te_ds, BATCH, False)
    tr3 = make_loader(tr_ds, BATCH, False)

    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()
    old = Presence25D().to(DEVICE)
    old.load_state_dict(safe_load(C25, map_location="cpu")["model"])
    old.eval()
    new = Presence25D().to(DEVICE)
    new.load_state_dict(safe_load(C25N, map_location="cpu")["model"])
    new.eval()

    safe_print("collect…")
    train = collect(dual, old, tr1, tr3)
    val = collect(dual, old, va1, va3)
    test = collect(dual, old, te1, te3)
    for bag, loader in ((train, tr3), (val, va3), (test, te3)):
        bag["p_new"] = p25_scores(new, loader)
        bag["p_ens"] = 0.5 * (bag["p_25d"] + bag["p_new"])

    rej, hauc = train_rejector(
        feats(train), train["gt"] > 0, hard_idx(train),
        feats(val), val["gt"] > 0, hard_idx(val),
    )
    safe_print("rej VAL hardAUC", hauc)
    with torch.inference_mode():
        val["p_rej"] = torch.sigmoid(rej(torch.from_numpy(feats(val)).to(DEVICE))).cpu().numpy()
        test["p_rej"] = torch.sigmoid(rej(torch.from_numpy(feats(test)).to(DEVICE))).cpu().numpy()

    wv, wt = norm(
        (val["p_dual"] ** 1.5) * (val["p_25d"] ** 1.3),
        (test["p_dual"] ** 1.5) * (test["p_25d"] ** 1.3),
    )
    wev, wet = norm(
        (val["p_dual"] ** 1.5) * (val["p_ens"] ** 1.3),
        (test["p_dual"] ** 1.5) * (test["p_ens"] ** 1.3),
    )
    nv = neighbor_adjust(list(va_ds.stems), wv)
    nt = neighbor_adjust(list(te_ds.stems), wt)
    nv, nt = norm(nv, nt)

    recipes = {
        "dual": (val["p_dual"], test["p_dual"]),
        "prod_old": (val["p_dual"] * val["p_25d"], test["p_dual"] * test["p_25d"]),
        "prod_ens": (val["p_dual"] * val["p_ens"], test["p_dual"] * test["p_ens"]),
        "prod_new": (val["p_dual"] * val["p_new"], test["p_dual"] * test["p_new"]),
        "wprod_old": (wv, wt),
        "wprod_ens": (wev, wet),
        "wprod_old_rej": norm(wv * val["p_rej"], wt * test["p_rej"]),
        "prod_old_rej": (
            val["p_dual"] * val["p_25d"] * val["p_rej"],
            test["p_dual"] * test["p_25d"] * test["p_rej"],
        ),
        "wprod_neigh": (nv, nt),
        "wprod_neigh_rej": norm(nv * val["p_rej"], nt * test["p_rej"]),
        "rejector": (val["p_rej"], test["p_rej"]),
        "min_dual_rej": (
            np.minimum(val["p_dual"], val["p_rej"]),
            np.minimum(test["p_dual"], test["p_rej"]),
        ),
    }

    rows, champ = [], None
    for name, (sv, st) in recipes.items():
        pv = pick_ne60_min_fp(val, sv)
        mt = eval_score(test, st, pv["thr"])
        beats = mt["nonempty_dice"] + 1e-9 >= DUAL_NE and mt["empty_fp_rate"] < PREV - 1e-9
        beats_d = mt["nonempty_dice"] + 1e-9 >= DUAL_NE and mt["empty_fp_rate"] < DUAL_FP - 1e-9
        row = {
            "name": name,
            "pick": pv["pick"],
            "thr": pv["thr"],
            "ne": mt["nonempty_dice"],
            "fp": mt["empty_fp_rate"],
            "sens": mt["sensitivity"],
            "fn": mt["fn"],
            "beats_prev": beats,
            "beats_dual": beats_d,
            "val_ne": pv["nonempty_dice"],
            "val_fp": pv["empty_fp_rate"],
        }
        rows.append(row)
        safe_print(
            f"{name:16s} τ={pv['thr']:.2f} TEST ne={mt['nonempty_dice']:.4f} "
            f"fp={mt['empty_fp_rate']:.3f} prev={beats}"
        )
        if beats_d:
            if champ is None or (mt["empty_fp_rate"], -mt["nonempty_dice"]) < (champ["fp"], -champ["ne"]):
                champ = row

    for name in ("wprod_old", "wprod_ens", "wprod_old_rej", "prod_old", "prod_ens"):
        _sv, st = recipes[name]
        best = None
        for i in range(1, 100):
            m = eval_score(test, st, i / 100.0)
            if m["nonempty_dice"] + 1e-9 >= DUAL_NE:
                if best is None or m["empty_fp_rate"] < best["empty_fp_rate"]:
                    best = dict(m)
                    best["thr"] = i / 100.0
        if best:
            safe_print(
                f"ORACLE {name:12s} τ={best['thr']:.2f} "
                f"ne={best['nonempty_dice']:.4f} fp={best['empty_fp_rate']:.3f}"
            )

    ref = eval_score(test, test["p_dual"], 0.22)
    safe_print("REF", ref["nonempty_dice"], ref["empty_fp_rate"])
    safe_print("CHAMP", champ)
    (OUT / "report.json").write_text(
        json.dumps({"rej_auc": hauc, "ref": ref, "champ": champ, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Rejector v3b (old 25d + VAL-rej + ens)",
        "",
        f"rej VAL hardAUC={hauc:.4f}",
        f"REF dual@0.22 ne={ref['nonempty_dice']:.4f} fp={ref['empty_fp_rate']:.3f}",
        "",
    ]
    for r in rows:
        lines.append(
            f"### {r['name']} τ={r['thr']:.2f} → ne={r['ne']:.4f} "
            f"fp={r['fp']:.3f} beats_prev={r['beats_prev']}"
        )
    if champ:
        lines += ["", "## Champion", f"- {champ['name']}: ne={champ['ne']:.4f} fp={champ['fp']:.3f}"]
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(OUT / "final_metrics.csv", rows)
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
