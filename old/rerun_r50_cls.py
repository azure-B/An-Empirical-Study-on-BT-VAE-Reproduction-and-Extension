"""Redo COVID ratio 0.5 S5+cls.

The segmenter stays. The old head used tau=0.81 and blanked too many
lesion slices (test nonempty Dice 0.768 -> 0.629). The new tau still
maximizes validation mean Dice, but only among thresholds that keep
validation nonempty Dice within 0.05 of the ungated S5.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models_lesion_presence import LesionPresenceHead
from train_covid_ratio import (
    CLS_EPOCHS,
    CLS_ES,
    CLS_POS_WEIGHT,
    collect_s5_cls,
    make_bundle,
    prob_group_stats,
)
from train_lesion_presence import binary_auroc, metrics_bundle
from train_original_bp import (
    DEVICE,
    atomic_torch_save,
    safe_load,
    safe_print,
)
from models_bp_skip import VAE as SkipVAE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "covid_pipeline" / "label_ratio" / "r50" / "s5_cls"
SEG = ROOT / "results" / "covid_pipeline" / "label_ratio" / "r50" / "s5" / "seg_best.pt"
NE_DROP = 0.05


def pick_tau(val):
    raw = metrics_bundle(
        val["gt"], val["mass"], val["dice"], np.ones(len(val["gt"]), dtype=bool),
    )
    floor = float(raw["nonempty_dice"]) - NE_DROP
    best = None
    for i in range(1, 100):
        thr = i / 100.0
        keep = val["score"] >= thr
        m = metrics_bundle(val["gt"], val["mass"], val["dice"], keep)
        if m["nonempty_dice"] < floor:
            continue
        key = (m["overall_dice"], -m["empty_fp_rate"])
        if best is None or key > best["key"]:
            best = {"thr": thr, "key": key, "m": m}
    if best is None:
        raise RuntimeError("no tau kept nonempty Dice")
    return best["thr"], best["m"], raw


def main():
    if (OUT / "report.json").exists() and not (OUT / "report_tau081.json").exists():
        shutil.copy(OUT / "report.json", OUT / "report_tau081.json")
        if (OUT / "head_best.pt").exists():
            shutil.copy(OUT / "head_best.pt", OUT / "head_tau081.pt")
    _full, train_loader, val_loader, test_loader, n_train = make_bundle(0.5, 4)
    model = SkipVAE(base=32).to(DEVICE)
    model.load_state_dict(safe_load(SEG, map_location="cpu")["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLS_POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_auroc, bad, best_ep = -1.0, 0, 0
    best_path = OUT / "head_best.pt"
    for ep in range(1, CLS_EPOCHS + 1):
        head.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if y.ndim == 3:
                y = y.unsqueeze(1)
            with torch.no_grad():
                mu, _lv, _z, _sk = model.encoder.forward_features(x)
            y_cls = (y.flatten(1).sum(1) > 0).float()
            opt.zero_grad(set_to_none=True)
            loss = bce(head(mu), y_cls)
            loss.backward()
            opt.step()
            total += float(loss.item()) * y_cls.numel()
            n += y_cls.numel()
        val = collect_s5_cls(model, head, val_loader)
        auroc = binary_auroc(val["gt"] > 0, val["score"])
        safe_print(f"[r50 cls {ep}/{CLS_EPOCHS}] bce={total/max(n,1):.4f} auroc={auroc:.4f}")
        if auroc > best_auroc + 1e-4:
            best_auroc, bad, best_ep = auroc, 0, ep
            atomic_torch_save({"head": head.state_dict(), "ep": ep, "auroc": auroc}, best_path)
        else:
            bad += 1
            if bad >= CLS_ES:
                safe_print(f"early stop @ {ep}")
                break
    head.load_state_dict(safe_load(best_path, map_location="cpu")["head"])
    head.eval()
    val = collect_s5_cls(model, head, val_loader)
    test = collect_s5_cls(model, head, test_loader)
    tau, val_m, raw = pick_tau(val)
    keep = test["score"] >= tau
    mt = metrics_bundle(test["gt"], test["mass"], test["dice"], keep)
    report = {
        "dataset": "covid",
        "ratio": 0.5,
        "mode": "s5_cls",
        "n_train": n_train,
        "loss": "BCEWithLogits",
        "probability": "p = sigmoid(Head(mu)) = P(lesion present | slice)",
        "tau": tau,
        "tau_rule": "max val mean Dice among tau with nonempty drop <= 0.05",
        "best_ep": best_ep,
        "val_auroc": best_auroc,
        "val_prob": prob_group_stats(val["gt"], val["score"]),
        "test_prob": prob_group_stats(test["gt"], test["score"]),
        "test": {
            "dice": float(mt["overall_dice"]),
            "nonempty_dice": float(mt["nonempty_dice"]),
            "empty_fp_rate": float(mt["empty_fp_rate"]),
        },
        "seed": 0,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print(
        f"tau={tau:.2f} val raw ne={raw['nonempty_dice']:.3f} "
        f"val ne={val_m['nonempty_dice']:.3f} "
        f"TEST dice={mt['overall_dice']:.4f} ne={mt['nonempty_dice']:.4f} "
        f"fp={mt['empty_fp_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
