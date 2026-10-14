"""none + exactly two trainable methods.

Does not modify original/ or none/seg_best.pt.

  python train_none_plus2.py --combo cp_gate
  python train_none_plus2.py --combo cp_tversky
  python train_none_plus2.py --combo gate_fp
  python train_none_plus2.py --all

Combos
  cp_gate      copy-paste + presence gate (VAE frozen)
  cp_tversky   copy-paste + Tversky-FP (light FT VAE)
  gate_fp      presence gate + empty-FP on the gate (VAE frozen)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_presence import PresenceVAE
from train_emptyfp import empty_fp_penalty, eval_split
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    DEVICE,
    NUM_WORKERS,
    OrigVAE,
    RECON_WEIGHT,
    RESULT_DIR,
    SEG_BATCH,
    atomic_torch_save,
    combined_binary_loss_sigmoid,
    evaluate_test,
    safe_load,
    safe_print,
)
from train_presence import eval_gated

NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
GATE_EPOCHS = 50
FT_EPOCHS = 30
GATE_LR = 1e-3
FT_LR = 3e-5
ES_GATE = 12
ES_FT = 10
POS_WEIGHT = 8.0
PASTE_P = 0.7
TVERSKY_A = 0.3  # FN
TVERSKY_B = 0.7  # FP
GATE_FP_W = 1.5
TAU_GRID = tuple(round(x, 2) for x in np.linspace(0.05, 0.90, 18))
COMBOS = ("cp_gate", "cp_tversky", "gate_fp")


def _prep(flair, seg):
    if flair.ndim == 3:
        flair = flair.unsqueeze(1)
    if seg.ndim == 3:
        seg = seg.unsqueeze(1)
    return flair, seg


def copypaste(img: torch.Tensor, mask: torch.Tensor, p: float = PASTE_P):
    """Paste a real lesion onto some empty slices. Same edges, lesion on/off."""
    empty = mask.flatten(1).sum(dim=1) == 0
    nonempty = ~empty
    ni = nonempty.nonzero(as_tuple=False).view(-1)
    ei = empty.nonzero(as_tuple=False).view(-1)
    if ni.numel() == 0 or ei.numel() == 0:
        return img, mask
    img = img.clone()
    mask = mask.clone()
    H, W = img.shape[-2:]
    for i in ei.tolist():
        if torch.rand((), device=img.device) > p:
            continue
        src = int(ni[torch.randint(0, ni.numel(), (1,), device=img.device)].item())
        m = mask[src, 0]
        if float(m.sum()) < 20:
            continue
        ys, xs = torch.where(m > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        ph, pw = y1 - y0 + 1, x1 - x0 + 1
        if ph >= H or pw >= W:
            continue
        ny = int(torch.randint(0, H - ph + 1, (1,), device=img.device))
        nx = int(torch.randint(0, W - pw + 1, (1,), device=img.device))
        pm = mask[src, :, y0 : y1 + 1, x0 : x1 + 1]
        pi = img[src, :, y0 : y1 + 1, x0 : x1 + 1]
        sel = pm > 0
        dest_i = img[i, :, ny : ny + ph, nx : nx + pw]
        dest_m = mask[i, :, ny : ny + ph, nx : nx + pw]
        img[i, :, ny : ny + ph, nx : nx + pw] = torch.where(sel, pi, dest_i)
        mask[i, :, ny : ny + ph, nx : nx + pw] = torch.where(sel, pm, dest_m)
    return img, mask


def tversky_loss(logits, target, alpha=TVERSKY_A, beta=TVERSKY_B, smooth=1.0):
    p = torch.sigmoid(logits).reshape(logits.size(0), -1)
    t = target.reshape(target.size(0), -1)
    tp = (p * t).sum(1)
    fn = ((1.0 - p) * t).sum(1)
    fp = (p * (1.0 - t)).sum(1)
    tv = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1.0 - tv).mean()


def load_frozen_none():
    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    ep = int(st.get("best_ep", -1))
    del st
    for p in vae.parameters():
        p.requires_grad = False
    vae.to(DEVICE).eval()
    return vae, ep


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def pick_tau_min_fp(model, val_loader, floor: float):
    rows, best = [], None
    for tau in TAU_GRID:
        s = eval_gated(model, val_loader, tau)
        ok = s["mean_dice_nonempty"] is not None and s["mean_dice_nonempty"] >= floor
        s["ok_nonempty"] = bool(ok)
        rows.append(s)
        safe_print(
            f"  tau={tau:.2f} dice={s['dice_mean']:.4f} nonempty={s['mean_dice_nonempty']:.4f} "
            f"empty_fp={s['empty_fp_rate']:.3f} rec={s['presence_recall']:.3f} "
            f"{'OK' if ok else 'drop'}"
        )
        if ok and (best is None or s["empty_fp_rate"] < best["empty_fp_rate"]):
            best = s
    if best is None:
        best = max(rows, key=lambda r: r["mean_dice_nonempty"] or -1.0)
        safe_print("no tau kept floor; fallback max nonempty")
    return best, rows


def train_gate(out_dir: Path, use_paste: bool, use_gate_fp: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    vae, none_ep = load_frozen_none()
    model = PresenceVAE(vae, base=BASE).to(DEVICE)
    opt = torch.optim.AdamW(model.head.parameters(), lr=GATE_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    ungated = evaluate_test(vae, data.val_loader)
    floor = float(ungated["mean_dice_nonempty"]) - 0.02
    safe_print(
        "ungated VAL",
        f"nonempty={ungated['mean_dice_nonempty']:.4f} empty_fp={ungated['empty_fp_rate']:.3f} "
        f"floor={floor:.4f} paste={use_paste} gate_fp={use_gate_fp}",
    )
    best_score, best_ep, history = -1.0, 0, []
    for ep in range(1, GATE_EPOCHS + 1):
        model.head.train()
        model.vae.eval()
        tr = ntr = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            if use_paste:
                flair, seg = copypaste(flair, seg)
            with torch.no_grad():
                _, mu, _ = model.vae(flair)
            y = (seg.flatten(1).sum(dim=1) > 0).float().unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logit = model.presence_logit(mu)
            loss = bce(logit, y)
            if use_gate_fp:
                empty = (y < 0.5).float()
                loss = loss + GATE_FP_W * (empty * torch.sigmoid(logit)).mean()
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        val_s = eval_gated(model, data.val_loader, 0.5)
        sched.step(val_s["presence_auroc"])
        safe_print(
            f"[gate] ep {ep}/{GATE_EPOCHS} loss={tr/max(1,ntr):.4f} "
            f"auroc={val_s['presence_auroc']:.3f} nonempty={val_s['mean_dice_nonempty']:.4f} "
            f"empty_fp={val_s['empty_fp_rate']:.3f} rec={val_s['presence_recall']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "val": val_s})
        payload = {
            "ep": ep, "head": model.head.state_dict(), "best_score": best_score,
            "best_ep": best_ep, "none_ep": none_ep,
        }
        if val_s["presence_auroc"] > best_score:
            best_score = float(val_s["presence_auroc"])
            best_ep = ep
            payload["best_score"] = best_score
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, out_dir / "head_best.pt")
            safe_print(f"  saved HEAD_BEST auroc={best_score:.4f}")
        atomic_torch_save(payload, out_dir / "head_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_GATE:
            safe_print(f"early stop @{best_ep}")
            break
    st = safe_load(out_dir / "head_best.pt", map_location="cpu")
    model.head.load_state_dict(st["head"])
    model.to(DEVICE).eval()
    best_tau, tau_rows = pick_tau_min_fp(model, data.val_loader, floor)
    tau = float(best_tau["tau"])
    val_g = eval_gated(model, data.val_loader, tau)
    test_g = eval_gated(model, data.test_loader, tau)
    test_u = evaluate_test(vae, data.test_loader)
    report = {
        "none_ep": none_ep, "head_best_ep": int(st.get("best_ep", best_ep)),
        "tau": tau, "pos_weight": POS_WEIGHT, "paste": use_paste, "gate_fp": use_gate_fp,
        "ungated_val": ungated, "ungated_test": test_u, "gated_val": val_g,
        "gated_test": test_g, "tau_sweep": tau_rows, "nonempty_floor": floor,
    }
    (out_dir / "report.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(jsonable(history), indent=2), encoding="utf-8")
    safe_print("TEST ungated", test_u["mean_dice_nonempty"], test_u["empty_fp_rate"])
    safe_print("TEST gated  ", test_g["mean_dice_nonempty"], test_g["empty_fp_rate"], "tau", tau)
    return report


def train_tversky(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=SEG_BATCH, num_workers=NUM_WORKERS)
    vae = OrigVAE(base=BASE)
    st = safe_load(NONE_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    none_ep = int(st.get("best_ep", -1))
    del st
    vae.to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=FT_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-7)
    init = evaluate_test(vae, data.val_loader)
    floor = float(init["mean_dice_nonempty"]) - 0.02
    safe_print(
        "init VAL",
        f"nonempty={init['mean_dice_nonempty']:.4f} empty_fp={init['empty_fp_rate']:.3f} floor={floor:.4f}",
    )
    best_fp, best_ep, best_ne = 2.0, 0, -1.0
    history = []
    for ep in range(1, FT_EPOCHS + 1):
        vae.train()
        tr = ntr = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
            flair, seg = copypaste(flair, seg)
            opt.zero_grad(set_to_none=True)
            out, mu, logvar = vae(flair)
            dice_bce, _, _ = combined_binary_loss_sigmoid(out, seg)
            tv = tversky_loss(out, seg)
            kl = 1 + logvar - torch.square(mu) - torch.exp(logvar)
            kl = (-0.5 * kl).sum(dim=-1).mean()
            extra = empty_fp_penalty(out, seg)
            # Tversky is the empty-FP method; keep λ small so TT does not collapse
            loss = RECON_WEIGHT * (0.5 * dice_bce + 0.5 * tv) + BETA_LOCAL * kl + 0.5 * extra
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        val = eval_split(vae, data.val_loader)
        sched.step(val["empty_fp_rate"])
        safe_print(
            f"[tv] ep {ep}/{FT_EPOCHS} loss={tr/max(1,ntr):.4f} "
            f"nonempty={val['mean_dice_nonempty']:.4f} empty_fp={val['empty_fp_rate']:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "val": val})
        payload = {
            "ep": ep, "model": vae.state_dict(), "best_fp": best_fp,
            "best_ep": best_ep, "none_ep": none_ep,
        }
        ok = val["mean_dice_nonempty"] is not None and val["mean_dice_nonempty"] >= floor
        if ok and val["empty_fp_rate"] < best_fp:
            best_fp = float(val["empty_fp_rate"])
            best_ep = ep
            best_ne = float(val["mean_dice_nonempty"])
            payload.update({"best_fp": best_fp, "best_ep": best_ep})
            atomic_torch_save(payload, out_dir / "seg_best.pt")
            safe_print(f"  saved BEST fp={best_fp:.3f} nonempty={best_ne:.4f}")
        atomic_torch_save(payload, out_dir / "seg_last.pt")
        if best_ep > 0 and (ep - best_ep) >= ES_FT:
            safe_print(f"early stop @{best_ep}")
            break
    if (out_dir / "seg_best.pt").exists():
        st = safe_load(out_dir / "seg_best.pt", map_location="cpu")
        vae.load_state_dict(st["model"])
        vae.to(DEVICE)
    val_b = eval_split(vae, data.val_loader)
    test_b = eval_split(vae, data.test_loader)
    report = {
        "none_ep": none_ep, "best_ep": best_ep, "init_val": init, "floor": floor,
        "val": val_b, "test": test_b, "paste": True, "tversky": [TVERSKY_A, TVERSKY_B],
    }
    (out_dir / "report.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(jsonable(history), indent=2), encoding="utf-8")
    safe_print("TEST", test_b["mean_dice_nonempty"], test_b["empty_fp_rate"])
    return report


def run_combo(name: str):
    safe_print("======== COMBO", name)
    if name == "cp_gate":
        return train_gate(RESULT_DIR / "none_cp_gate", use_paste=True, use_gate_fp=False)
    if name == "gate_fp":
        return train_gate(RESULT_DIR / "none_gate_fp", use_paste=False, use_gate_fp=True)
    if name == "cp_tversky":
        return train_tversky(RESULT_DIR / "none_cp_tversky")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", choices=COMBOS, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not NONE_BEST.exists():
        raise FileNotFoundError(NONE_BEST)
    names = COMBOS if args.all or args.combo is None else (args.combo,)
    if args.combo is None and not args.all:
        names = COMBOS
        safe_print("no --combo; running all three")
    summary = {}
    for name in names:
        summary[name] = run_combo(name)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    (RESULT_DIR / "none_plus2_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2), encoding="utf-8"
    )
    safe_print("wrote", RESULT_DIR / "none_plus2_summary.json")


if __name__ == "__main__":
    main()
