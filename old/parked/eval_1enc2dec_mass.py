"""Clean 1-encoder 2-decoder cascade + min-mass gate. Report vs messy wprod+mass."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_load_BT import datasets
from models_bt_cascade import BTCascadeVAE
from models_bt_cascade_cls import BTCascadeClsUpgrade, load_from_cascade_focal
from train_lesion_presence import PIXEL_THR, _prep, binary_auroc, conf_matrix, metrics_bundle
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_load, safe_print

OUT = RESULT_DIR / "cascade_1enc2dec_mass"
FOCAL_TAU = RESULT_DIR / "bt_cascade_focal" / "checkpoints" / "bt_cascade_best_tau.pt"
FOCAL_SEG = RESULT_DIR / "bt_cascade_focal" / "checkpoints" / "bt_cascade_best_seg.pt"
CLS_RECALL = RESULT_DIR / "bt_cascade_cls_recall" / "checkpoints" / "cls_best_auroc.pt"
BATCH = 8
NONE_NE, NONE_FP = 0.572, 0.383
WPROD_NE, WPROD_FP = 0.601, 0.305


@torch.inference_mode()
def collect_focal(model, loader):
    model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, _cmap, plog, _dec, *_ = model.forward_parts(flair)
        # score = sigmoid(presence); τ separate later (ignore learned tau for sweep)
        score = torch.sigmoid(plog)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        dice = torch.where((ps == 0) & (gs == 0), torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(score.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


@torch.inference_mode()
def collect_recall(model, loader):
    model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, presence, *_ = model.forward_parts(flair)
        score = torch.sigmoid(presence)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg.sum(dim=(1, 2, 3))
        inter = (pred * seg).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        dice = torch.where((ps == 0) & (gs == 0), torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(score.cpu().numpy())
    return {k: np.concatenate(v) for k, v in bags.items()}


def eval_keep(bag, keep):
    yt = bag["gt"] > 0
    cm = conf_matrix(yt, keep)
    m = metrics_bundle(bag["gt"], bag["mass"], bag["dice"], keep)
    return {
        "sensitivity": float(cm["sensitivity"]),
        "specificity": float(cm["specificity"]),
        "tp": cm["TT"], "tn": cm["FF"], "fp": cm["FT"], "fn": cm["TF"],
        "nonempty_dice": float(m["nonempty_dice"]),
        "empty_fp_rate": float(m["empty_fp_rate"]),
        "auroc": binary_auroc(yt, bag["score"]),
    }


def pick_val(val, ne_floor=0.57):
    """Prefer ne≥floor & min fp; else max ne-0.5*fp."""
    masses = [0, 80, 160, 320, 640, 1280, 2560]
    beat, joint = None, None
    for i in range(1, 100):
        thr = i / 100.0
        for mmin in masses:
            keep = (val["score"] >= thr) & (val["mass"] >= mmin)
            m = eval_keep(val, keep)
            m.update({"thr": thr, "min_mass": mmin})
            j = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
            if joint is None or j > joint["_j"]:
                joint = dict(m)
                joint["_j"] = j
            if m["nonempty_dice"] + 1e-9 >= ne_floor:
                key = (-m["empty_fp_rate"], m["nonempty_dice"])
                if beat is None or key > beat.get("_key", (-1e9, 0)):
                    beat = dict(m)
                    beat["_key"] = key
                    beat["pick"] = "ne_floor_min_fp"
    if beat is not None:
        return beat
    joint["pick"] = "joint"
    return joint


def run_recipe(name, val, test):
    raw = eval_keep(test, np.ones(len(test["gt"]), dtype=bool))
    pv = pick_val(val)
    keep = (test["score"] >= pv["thr"]) & (test["mass"] >= pv["min_mass"])
    mt = eval_keep(test, keep)
    # also learned-style: τ only no mass (best joint on score alone)
    best_t = None
    for i in range(1, 100):
        thr = i / 100.0
        m = eval_keep(val, val["score"] >= thr)
        j = m["nonempty_dice"] - 0.5 * m["empty_fp_rate"]
        if best_t is None or j > best_t["_j"]:
            best_t = dict(m)
            best_t["_j"] = j
            best_t["thr"] = thr
    mt0 = eval_keep(test, test["score"] >= best_t["thr"])
    row = {
        "name": name,
        "raw_ne": raw["nonempty_dice"],
        "raw_fp": raw["empty_fp_rate"],
        "gate_only_thr": best_t["thr"],
        "gate_only_ne": mt0["nonempty_dice"],
        "gate_only_fp": mt0["empty_fp_rate"],
        "gate_only_sens": mt0["sensitivity"],
        "mass_thr": pv["thr"],
        "min_mass": pv["min_mass"],
        "mass_ne": mt["nonempty_dice"],
        "mass_fp": mt["empty_fp_rate"],
        "mass_sens": mt["sensitivity"],
        "mass_fn": mt["fn"],
        "pick": pv["pick"],
        "auroc": mt["auroc"],
    }
    safe_print(
        f"==== {name} ====\n"
        f"  raw ne={row['raw_ne']:.4f} fp={row['raw_fp']:.3f}\n"
        f"  score-only τ={row['gate_only_thr']:.2f} → ne={row['gate_only_ne']:.4f} fp={row['gate_only_fp']:.3f}\n"
        f"  +mass τ={row['mass_thr']:.2f} m≥{row['min_mass']} → "
        f"ne={row['mass_ne']:.4f} fp={row['mass_fp']:.3f} sens={row['mass_sens']:.3f}"
    )
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    va = DataLoader(data.val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te = DataLoader(data.test, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    rows = []

    # A) strict 1enc-2dec: Decoder_cls max → presence
    m_f = BTCascadeVAE(base=BASE).to(DEVICE)
    st = safe_load(FOCAL_TAU, map_location="cpu")
    m_f.load_state_dict(st["model"], strict=False)
    m_f.freeze_encoder()
    m_f.eval()
    safe_print("collect cascade_focal (Dec_cls amax)…")
    rows.append(run_recipe("1enc2dec_focal_DecCls", collect_focal(m_f, va), collect_focal(m_f, te)))

    # B) 1enc + Dec_seg + Head(μ)+Head(skips)  [still one encoder]
    m_r = BTCascadeClsUpgrade(base=BASE).to(DEVICE)
    load_from_cascade_focal(m_r, safe_load(FOCAL_SEG, map_location="cpu"))
    m_r.load_state_dict(safe_load(CLS_RECALL, map_location="cpu")["model"], strict=False)
    m_r.freeze_encoder()
    m_r.freeze_seg()
    m_r.eval()
    safe_print("collect cascade_cls_recall (Head μ+skips)…")
    rows.append(run_recipe("1enc2dec_cls_recall_heads", collect_recall(m_r, va), collect_recall(m_r, te)))

    report = {
        "design": "1 shared encoder → Decoder_seg (S5) + cls branch",
        "refs": {
            "none": {"ne": NONE_NE, "fp": NONE_FP},
            "wprod_mass_messy": {"ne": WPROD_NE, "fp": WPROD_FP},
        },
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Clean 1-encoder · 2-decoder (cascade) + min-mass",
        "",
        "```text",
        "x → Encoder (shared, frozen BT)",
        "      ├─ μ + skips → Decoder_seg (S5) → mask, mass",
        "      └─ μ → Decoder_cls  OR  Head(μ)+Head(skips) → p",
        "keep if p≥τ and mass≥m_min",
        "```",
        "",
        "| recipe | raw ne/fp | score-gate ne/fp | +mass ne/fp |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['raw_ne']:.3f}/{r['raw_fp']:.3f} | "
            f"{r['gate_only_ne']:.3f}/{r['gate_only_fp']:.3f} @τ={r['gate_only_thr']:.2f} | "
            f"**{r['mass_ne']:.3f}/{r['mass_fp']:.3f}** "
            f"(τ={r['mass_thr']:.2f}, m≥{r['min_mass']}) |"
        )
    lines += [
        "",
        "## Refs (messy multi-net final)",
        f"- none raw: {NONE_NE}/{NONE_FP}",
        f"- wprod+mass (S5+dual+25d): {WPROD_NE}/{WPROD_FP}",
        "",
        "## Takeaway",
        "- Architecture is reviewable: **one encoder, seg decoder + cls branch**.",
        "- Raw SEG ne is strong (~0.72); gate still trades FP vs ne.",
        "- Compare mass column to wprod+mass for paper choice.",
    ]
    (OUT / "COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# 1 Encoder · 2 Decoder (paper-clean)",
            "",
            "## Structure",
            "- **1× Encoder** (BT, shared μ + skips)",
            "- **Decoder_seg**: S5 concat skips → lesion mask",
            "- **Cls branch**: either `Decoder_cls(μ)` (amax→p) or `Head(μ)+Head(skips)`",
            "- **Gate**: τ on p + optional min pred-mass",
            "",
            "No second VAE encoder, no external ResNet — unlike wprod+mass.",
            "",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
