"""Paper-style Dice on BP TEST, plus per-slice curves vs none.

Candidates: none, S5 (1 enc · 1 dec), wprod+mass, 1enc2dec+head.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from models_bt_cascade_cls import BTCascade1Enc2DecHead, load_from_cascade_focal
from models_dual_role import DualRoleVAE
from models_presence_25d import Presence25D
from train_lesion_presence import PIXEL_THR, metrics_bundle
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RESULT_DIR,
    batch_dice,
    batch_pred_gt_sums,
    safe_load,
    safe_print,
)
from train_s5_gate_25d import Stack25D, make_loader
from eval_s5_fuse_reject import collect

NONE = RESULT_DIR / "none" / "seg_best.pt"
S5 = RESULT_DIR / "s5_concat" / "seg_best.pt"
DUAL = RESULT_DIR / "dual_role_train" / "checkpoints" / "dual_role_best_auroc.pt"
C25 = RESULT_DIR / "s5_gate_25d" / "checkpoints" / "cls25d_best_auroc.pt"
FOCAL = RESULT_DIR / "bt_cascade_focal" / "checkpoints" / "bt_cascade_best_seg.pt"
HEAD = RESULT_DIR / "cascade_1enc2dec_head_tffp" / "checkpoints" / "cls_best_tffp.pt"
BATCH = 8


def report(name, gt, mass, dice, keep):
    m = metrics_bundle(gt, mass, dice, keep)
    # gated dice vector for median
    d = dice.copy()
    rejected = ~keep
    empty = gt == 0
    d[rejected & empty] = 1.0
    d[rejected & ~empty] = 0.0
    pred = mass.copy()
    pred[rejected] = 0
    ne = d[gt > 0]
    safe_print(
        f"{name}\n"
        f"  mean={d.mean():.4f} median={np.median(d):.4f} "
        f"ne_mean={ne.mean():.4f} ne_median={np.median(ne):.4f} "
        f"empty_fp={m['empty_fp_rate']:.3f} n={len(d)}"
    )
    return {
        "name": name,
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "ne_mean": float(ne.mean()),
        "ne_median": float(np.median(ne)),
        "fp": float(m["empty_fp_rate"]),
        "dice": d,
        "gt": gt,
    }


@torch.inference_mode()
def collect_vae(model, loader, kind: str):
    model.eval()
    gts, masses, dices = [], [], []
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
        gts.append(batch_pred_gt_sums(logits, y)[1])
        masses.append(batch_pred_gt_sums(logits, y)[0])
        dices.append(batch_dice(logits, y))
    return (
        np.concatenate(gts),
        np.concatenate(masses),
        np.concatenate(dices),
    )


def main():
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    te = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    rows = []

    none = OrigVAE(BASE).to(DEVICE)
    none.load_state_dict(safe_load(NONE, map_location="cpu")["model"])
    gt, mass, dice = collect_vae(none, te, "none")
    rows.append(report("none (1enc-1dec raw)", gt, mass, dice, np.ones(len(gt), dtype=bool)))
    del none
    torch.cuda.empty_cache()

    s5 = SkipVAE(BASE).to(DEVICE)
    s5.load_state_dict(safe_load(S5, map_location="cpu")["model"])
    gt, mass, dice = collect_vae(s5, te, "s5")
    rows.append(report("S5 (1enc-1dec raw)", gt, mass, dice, np.ones(len(gt), dtype=bool)))
    del s5
    torch.cuda.empty_cache()

    va1 = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te1 = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    va3 = make_loader(Stack25D(data.val), BATCH, False)
    te3 = make_loader(Stack25D(data.test), BATCH, False)
    dual = DualRoleVAE(base=BASE).to(DEVICE)
    dual.load_state_dict(safe_load(DUAL, map_location="cpu")["model"])
    dual.freeze_skip()
    dual.eval()
    cls = Presence25D().to(DEVICE)
    cls.load_state_dict(safe_load(C25, map_location="cpu")["model"])
    cls.eval()
    test = collect(dual, cls, te1, te3)
    s = (test["p_dual"] ** 1.5) * (test["p_25d"] ** 1.3)
    test["score"] = s / (s.max() + 1e-8)
    keep = (test["score"] >= 0.01) & (test["mass"] >= 320)
    rows.append(report("wprod+mass (S5 dec + dual enc + 25d)", test["gt"], test["mass"], test["dice"], keep))
    del dual, cls
    torch.cuda.empty_cache()

    head = BTCascade1Enc2DecHead(base=BASE).to(DEVICE)
    load_from_cascade_focal(head, safe_load(FOCAL, map_location="cpu"))
    st = safe_load(HEAD, map_location="cpu")
    head.load_state_dict(st["model"], strict=False)
    head.freeze_encoder()
    head.freeze_seg()
    head.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in te:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        seg_l, presence, *_ = head.forward_parts(x)
        score = torch.sigmoid(presence)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = y.sum(dim=(1, 2, 3))
        inter = (pred * y).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(score.detach().cpu().numpy())
    bag = {k: np.concatenate(v) for k, v in bags.items()}
    for thr, mmin, tag in ((0.01, 1280, "ne60"), (0.09, 640, "ne57")):
        keep = (bag["score"] >= thr) & (bag["mass"] >= mmin)
        rows.append(report(f"1enc2dec+head {tag} τ={thr} m≥{mmin}", bag["gt"], bag["mass"], bag["dice"], keep))

    safe_print("--- rank by mean dice (paper-style, all slices) ---")
    for r in sorted(rows, key=lambda z: -z["mean"])[:2]:
        safe_print(f"TOP mean {r['name']} {r['mean']:.4f}")
    safe_print("--- rank by nonempty mean ---")
    for r in sorted(rows, key=lambda z: -z["ne_mean"])[:2]:
        safe_print(f"TOP ne {r['name']} {r['ne_mean']:.4f}")

    none_row = rows[0]
    base = none_row["dice"]
    gt = none_row["gt"]
    pcts = list(range(0, 101, 5))
    curves = {}
    vs = {}
    for r in rows:
        d = r["dice"]
        curves[r["name"]] = {
            "all": [round(float(np.percentile(np.sort(d), p)), 4) for p in pcts],
            "nonempty": [round(float(np.percentile(np.sort(d[gt > 0]), p)), 4) for p in pcts],
        }
        if r is none_row:
            continue
        delta = d - base
        ne = gt > 0
        vs[r["name"]] = {
            "n": int(len(d)),
            "win": int((delta > 1e-4).sum()),
            "tie": int((np.abs(delta) <= 1e-4).sum()),
            "lose": int((delta < -1e-4).sum()),
            "ne_n": int(ne.sum()),
            "ne_win": int((delta[ne] > 1e-4).sum()),
            "ne_tie": int((np.abs(delta[ne]) <= 1e-4).sum()),
            "ne_lose": int((delta[ne] < -1e-4).sum()),
        }
        safe_print(
            f"vs none {r['name']}: all win/tie/lose "
            f"{vs[r['name']]['win']}/{vs[r['name']]['tie']}/{vs[r['name']]['lose']} | "
            f"nonempty {vs[r['name']]['ne_win']}/{vs[r['name']]['ne_tie']}/{vs[r['name']]['ne_lose']}"
        )
    out = RESULT_DIR / "paper_dice_vs_none.json"
    out.write_text(
        json.dumps({"percentiles": pcts, "curves": curves, "vs_none": vs}, indent=2),
        encoding="utf-8",
    )
    safe_print("wrote", out)


if __name__ == "__main__":
    main()
