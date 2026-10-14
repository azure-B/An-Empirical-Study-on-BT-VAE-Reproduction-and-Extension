"""Cascade: Cls decoder + learnable τ + SEG (nonempty-only).

Phase1: Decoder_cls + τ_logit — BCE(presence − τ, y∈{0,1}) on all slices
Phase2: freeze cls/τ; Decoder_seg on nonempty only
Phase3: freeze all but τ — recalibrate τ on full mix
Infer: keep SEG iff presence_logit ≥ τ_logit (learned; no hand-picked thr)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_bt_cascade import BTCascadeVAE, load_bt_cascade
from train_lesion_presence import PIXEL_THR, _prep, metrics_bundle
from train_original_bp import (
    BASE,
    BT_BEST,
    BT_LAST,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    combined_binary_loss_sigmoid,
    safe_load,
    safe_print,
    split_dice_stats,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
OUT = RESULT_DIR / "bt_cascade"
BATCH = 8
CLS_EPOCHS = 40
SEG_EPOCHS = 60
TAU_EPOCHS = 20
LR_CLS = 1e-4
LR_SEG = 1e-4
LR_TAU = 1e-2
POS_WEIGHT = 1.5
ES_CLS = 12
ES_SEG = 15
ES_TAU = 8


def binary_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def presence_y(seg: torch.Tensor) -> torch.Tensor:
    return (seg.flatten(1).sum(dim=1) > 0).float()


def nonempty_mask(seg: torch.Tensor) -> torch.Tensor:
    return seg.flatten(1).sum(dim=1) > 0


def tau_stats(model) -> dict:
    with torch.no_grad():
        tl = float(model.tau_logit.detach().cpu())
        tp = float(torch.sigmoid(model.tau_logit).detach().cpu())
    return {"tau_logit": tl, "tau_prob": tp}


def cls_step(model, flair, seg, bce):
    """BCE on decision = presence_logit − τ_logit."""
    flair, seg = _prep(flair, seg)
    _seg, _cmap, _plog, decision, *_ = model.forward_parts(flair)
    y = presence_y(seg)
    loss = bce(decision, y)
    return loss, float(loss.item()), y, decision


def seg_step_nonempty(model, flair, seg):
    flair, seg = _prep(flair, seg)
    keep = nonempty_mask(seg)
    if int(keep.sum()) == 0:
        return None
    flair = flair[keep]
    seg = seg[keep]
    seg_l, *_ = model.forward_parts(flair)
    dice_s, _, _ = combined_binary_loss_sigmoid(seg_l, seg)
    return dice_s, seg_l, seg


@torch.inference_mode()
def val_gate_metrics(model, loader):
    """AUROC on presence scores + gated empty-FP / nonempty dice at learned τ."""
    model.eval()
    bags = {"gt": [], "mass": [], "dice": [], "score": [], "keep": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, _c, plog, decision, *_ = model.forward_parts(flair)
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass"].append(ps.cpu().numpy())
        bags["dice"].append(dice.cpu().numpy())
        bags["score"].append(torch.sigmoid(plog).cpu().numpy())
        bags["keep"].append((decision >= 0).cpu().numpy())
    gt = np.concatenate(bags["gt"])
    mass = np.concatenate(bags["mass"])
    dice = np.concatenate(bags["dice"])
    score = np.concatenate(bags["score"])
    keep = np.concatenate(bags["keep"]).astype(bool)
    m = metrics_bundle(gt, mass, dice, keep)
    yt = gt > 0
    m["presence_auroc"] = binary_auroc(yt, score)
    m["presence_recall"] = float((keep & yt).sum() / max(1, int(yt.sum())))
    return m


@torch.inference_mode()
def eval_seg_full(model, loader):
    model.eval()
    dices, gt, pred = [], [], []
    loss_sum, n_ne = 0.0, 0
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, *_ = model.forward_parts(flair)
        keep = nonempty_mask(seg)
        if int(keep.sum()) > 0:
            dice_s, _, _ = combined_binary_loss_sigmoid(seg_l[keep], seg[keep])
            loss_sum += float(dice_s.item())
            n_ne += 1
        dices.append(batch_dice(seg_l, seg))
        ps, gs = batch_pred_gt_sums(seg_l, seg)
        pred.append(ps)
        gt.append(gs)
    st = split_dice_stats(np.concatenate(dices), np.concatenate(gt), np.concatenate(pred))
    st["val_loss_ne"] = loss_sum / max(1, n_ne)
    return st


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    bags = {"gt": [], "mass_seg": [], "dice_seg": [], "cls_prob": [], "keep": []}
    for batch in loader:
        flair, seg = _prep(batch[0].to(DEVICE), batch[1].to(DEVICE))
        seg_l, _c, plog, decision, *_ = model.forward_parts(flair)
        seg_b = seg if seg.ndim == 4 else seg.unsqueeze(1)
        pred = (torch.sigmoid(seg_l) >= PIXEL_THR).float()
        ps = pred.sum(dim=(1, 2, 3))
        gs = seg_b.sum(dim=(1, 2, 3))
        inter = (pred * seg_b).sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (ps + gs + 1e-6)
        both = (ps == 0) & (gs == 0)
        dice = torch.where(both, torch.ones_like(dice), dice)
        bags["gt"].append(gs.cpu().numpy())
        bags["mass_seg"].append(ps.cpu().numpy())
        bags["dice_seg"].append(dice.cpu().numpy())
        bags["cls_prob"].append(torch.sigmoid(plog).cpu().numpy())
        bags["keep"].append((decision >= 0).cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in bags.items()}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # union keys across phases (cls/seg/tau rows differ)
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-epochs", type=int, default=CLS_EPOCHS)
    ap.add_argument("--seg-epochs", type=int, default=SEG_EPOCHS)
    ap.add_argument("--tau-epochs", type=int, default=TAU_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--skip-cls", action="store_true", help="skip phase1; use best_cls")
    ap.add_argument(
        "--resume-seg", type=str, default="",
        help="load this seg ckpt and continue phase2+3 (implies --skip-cls)",
    )
    args = ap.parse_args()
    if args.resume_seg:
        args.skip_cls = True

    bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
    if not bt_ckpt.exists():
        raise FileNotFoundError(bt_ckpt)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # pin_memory=False avoids Windows semaphore flake after abrupt kills; workers OK
    data = datasets.bp_train_30(
        batch_size=args.batch, num_workers=NUM_WORKERS, pin_memory=False,
    )
    safe_print(
        f"OUT={OUT} DEVICE={DEVICE} batch={args.batch} workers={NUM_WORKERS} "
        f"BT={bt_ckpt.name} n={len(data.train)}/{len(data.val)}/{len(data.test)}"
    )
    safe_print("design: shared BT μ → ClsDecoder+τ | SEG(nonempty); decision≥0")

    model = BTCascadeVAE(base=BASE).to(DEVICE)
    load_bt_cascade(
        model,
        bt_ckpt,
        safe_load(S5_BEST, map_location="cpu") if S5_BEST.exists() else None,
        safe_load(NONE_BEST, map_location="cpu") if NONE_BEST.exists() else None,
    )
    model.freeze_encoder()
    safe_print("encoder←BT frozen (shared μ); decoder_seg←s5; decoder_cls←none; τ learnable")

    history = []
    best_cls_path = ckpt_dir / "bt_cascade_best_cls.pt"
    best_seg_path = ckpt_dir / "bt_cascade_best_seg.pt"
    best_tau_path = ckpt_dir / "bt_cascade_best_tau.pt"
    last_path = ckpt_dir / "bt_cascade_last.pt"
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    best_auroc = -1.0

    # ----- Phase 1: cls decoder + τ -----
    if args.skip_cls:
        if not best_cls_path.exists():
            raise FileNotFoundError(best_cls_path)
        st = safe_load(best_cls_path, map_location="cpu")
        model.load_state_dict(st["model"])
        best_auroc = float(st.get("val_auroc", -1.0))
        safe_print(f"skip-cls: loaded {best_cls_path.name} auroc={best_auroc:.4f}")
    else:
        model.set_trainable(seg=False, cls=True, tau=True)
        opt = torch.optim.AdamW(
            [
                {"params": model.decoder_cls.parameters(), "lr": LR_CLS},
                {"params": [model.tau_logit], "lr": LR_TAU, "weight_decay": 0.0},
            ],
            weight_decay=1e-4,
        )
        sch = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
        es = 0
        safe_print(f"phase1 CLS+τ ep≤{args.cls_epochs}")
        for ep in range(1, args.cls_epochs + 1):
            model.train()
            model.freeze_encoder()
            model.decoder_seg.eval()
            tr, ntr = 0.0, 0
            t0 = time.time()
            for batch in data.train_loader:
                flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss, lb, _, _ = cls_step(model, flair, seg, bce)
                loss.backward()
                opt.step()
                tr += lb
                ntr += 1
            vm = val_gate_metrics(model, data.val_loader)
            auroc = float(vm["presence_auroc"])
            ts = tau_stats(model)
            rec = {
                "phase": "cls", "epoch": ep,
                "train_bce": tr / max(1, ntr), "val_auroc": auroc,
                "val_fp_gated": float(vm["empty_fp_rate"]),
                "val_ne_gated": float(vm["nonempty_dice"]),
                **ts, "sec": time.time() - t0,
            }
            history.append(rec)
            safe_print(
                f"[cls {ep}/{args.cls_epochs}] bce={rec['train_bce']:.4f} "
                f"auroc={auroc:.4f} τ={ts['tau_prob']:.3f}(L={ts['tau_logit']:.3f}) "
                f"gate ne={rec['val_ne_gated']:.3f} fp={rec['val_fp_gated']:.3f} "
                f"({rec['sec']:.1f}s)"
            )
            sch.step(auroc)
            payload = {
                "model": model.state_dict(), "epoch": ep, "phase": "cls",
                "val_auroc": auroc, **ts, "design": "bt_cascade_learnable_tau",
                "bt_ckpt": str(bt_ckpt),
            }
            atomic_torch_save(payload, last_path)
            if auroc > best_auroc + 1e-4:
                best_auroc = auroc
                atomic_torch_save(payload, best_cls_path)
                es = 0
                safe_print(f"  * best auroc={auroc:.4f} τ={ts['tau_prob']:.3f}")
            else:
                es += 1
                if es >= ES_CLS:
                    safe_print(f"early stop CLS @ ep {ep}")
                    break

    # ----- Phase 2: SEG nonempty -----
    resume_seg_path = Path(args.resume_seg) if args.resume_seg else None
    if resume_seg_path and resume_seg_path.exists():
        st = safe_load(resume_seg_path, map_location="cpu")
        model.load_state_dict(st["model"])
        best_val = float(st.get("val_loss_ne", float("inf")))
        best_ne = float(st.get("val_ne", -1.0))
        if best_val == float("inf") or best_ne < 0:
            stv0 = eval_seg_full(model, data.val_loader)
            best_val = float(stv0["val_loss_ne"])
            best_ne = float(stv0["mean_dice_nonempty"] or 0.0)
            atomic_torch_save(
                {
                    "model": model.state_dict(), "epoch": int(st.get("epoch", 0)),
                    "phase": "seg", "val_loss_ne": best_val, "val_ne": best_ne,
                    "val_auroc": best_auroc, "design": "bt_cascade_learnable_tau",
                    "bt_ckpt": str(bt_ckpt),
                },
                best_seg_path,
            )
        safe_print(
            f"resume-seg: {resume_seg_path.name} baseline "
            f"val_ne_loss={best_val:.4f} ne={best_ne:.4f}"
        )
    else:
        st = safe_load(best_cls_path, map_location="cpu")
        model.load_state_dict(st["model"])
        best_val = float("inf")
        best_ne = -1.0

    model.freeze_encoder()
    model.set_trainable(seg=True, cls=False, tau=False)
    model.decoder_cls.eval()

    opt = torch.optim.AdamW(
        [p for p in model.decoder_seg.parameters() if p.requires_grad],
        lr=LR_SEG, weight_decay=1e-4,
    )
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    es = 0
    safe_print(f"phase2 SEG nonempty-only ep≤{args.seg_epochs}")
    for ep in range(1, args.seg_epochs + 1):
        model.train()
        model.freeze_encoder()
        model.decoder_cls.eval()
        tr, ntr, n_skip = 0.0, 0, 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = seg_step_nonempty(model, flair, seg)
            if out is None:
                n_skip += 1
                continue
            loss, _, _ = out
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            ntr += 1
        stv = eval_seg_full(model, data.val_loader)
        ne = float(stv["mean_dice_nonempty"] or 0.0)
        vl = float(stv["val_loss_ne"])
        ts = tau_stats(model)
        rec = {
            "phase": "seg", "epoch": ep,
            "train_loss": tr / max(1, ntr),
            "train_batches": ntr, "skip_empty_batches": n_skip,
            "val_loss_ne": vl, "val_ne": ne,
            "val_fp": float(stv["empty_fp_rate"]),
            **ts, "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[seg {ep}/{args.seg_epochs}] tr={rec['train_loss']:.4f} "
            f"batches={ntr}/{ntr + n_skip} val_ne_loss={vl:.4f} "
            f"ne={ne:.4f} fp={rec['val_fp']:.3f} ({rec['sec']:.1f}s)"
        )
        sch.step(vl)
        payload = {
            "model": model.state_dict(), "epoch": ep, "phase": "seg",
            "val_loss_ne": vl, "val_ne": ne, "val_auroc": best_auroc,
            **ts, "design": "bt_cascade_learnable_tau", "bt_ckpt": str(bt_ckpt),
        }
        atomic_torch_save(payload, last_path)
        if vl < best_val - 1e-4:
            best_val = vl
            best_ne = ne
            atomic_torch_save(payload, best_seg_path)
            es = 0
            safe_print(f"  * best seg val_ne_loss={vl:.4f} ne={ne:.4f}")
        else:
            es += 1
            if es >= ES_SEG:
                safe_print(f"early stop SEG @ ep {ep}")
                break

    # ----- Phase 3: τ-only recalibration -----
    st = safe_load(best_seg_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_encoder()
    model.set_trainable(seg=False, cls=False, tau=True)
    model.decoder_cls.eval()
    model.decoder_seg.eval()

    opt = torch.optim.AdamW([model.tau_logit], lr=LR_TAU, weight_decay=0.0)
    # score = gated ne - 0.5*fp  (higher better) for early stop
    best_score = -1e9
    es = 0
    safe_print(f"phase3 τ-only ep≤{args.tau_epochs}")
    for ep in range(1, args.tau_epochs + 1):
        model.train()
        model.freeze_encoder()
        model.decoder_cls.eval()
        model.decoder_seg.eval()
        model.tau_logit.requires_grad = True
        tr, ntr = 0.0, 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, lb, _, _ = cls_step(model, flair, seg, bce)
            loss.backward()
            opt.step()
            tr += lb
            ntr += 1
        vm = val_gate_metrics(model, data.val_loader)
        ts = tau_stats(model)
        score = float(vm["nonempty_dice"]) - 0.5 * float(vm["empty_fp_rate"])
        rec = {
            "phase": "tau", "epoch": ep,
            "train_bce": tr / max(1, ntr),
            "val_auroc": float(vm["presence_auroc"]),
            "val_ne_gated": float(vm["nonempty_dice"]),
            "val_fp_gated": float(vm["empty_fp_rate"]),
            "val_score": score, **ts, "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[tau {ep}/{args.tau_epochs}] bce={rec['train_bce']:.4f} "
            f"τ={ts['tau_prob']:.3f}(L={ts['tau_logit']:.3f}) "
            f"ne={rec['val_ne_gated']:.3f} fp={rec['val_fp_gated']:.3f} "
            f"score={score:.3f} ({rec['sec']:.1f}s)"
        )
        payload = {
            "model": model.state_dict(), "epoch": ep, "phase": "tau",
            "val_score": score, **ts, "design": "bt_cascade_learnable_tau",
        }
        atomic_torch_save(payload, last_path)
        if score > best_score + 1e-4:
            best_score = score
            atomic_torch_save(payload, best_tau_path)
            es = 0
            safe_print(f"  * best τ score={score:.4f} τ={ts['tau_prob']:.3f}")
        else:
            es += 1
            if es >= ES_TAU:
                safe_print(f"early stop τ @ ep {ep}")
                break

    write_csv(OUT / "history.csv", history)

    # ----- Final: learned τ only (no hand sweep as primary) -----
    final_path = best_tau_path if best_tau_path.exists() else best_seg_path
    st = safe_load(final_path, map_location="cpu")
    model.load_state_dict(st["model"])
    model.freeze_encoder()
    model.eval()
    ts = tau_stats(model)
    val = collect(model, data.val_loader)
    test = collect(model, data.test_loader)

    def tf_from_keep(gt, keep):
        yt = gt > 0
        yp = keep.astype(bool)
        tp = int((yp & yt).sum())
        tn = int((~yp & ~yt).sum())
        fp = int((yp & ~yt).sum())
        fn = int((~yp & yt).sum())
        sens = tp / max(1, tp + fn)  # recall / TPR
        spec = tn / max(1, tn + fp)  # TNR
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * sens / max(1e-8, prec + sens)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": float(sens), "specificity": float(spec),
            "precision": float(prec), "f1": float(f1), "accuracy": float(acc),
            "n_pos": int(yt.sum()), "n_neg": int((~yt).sum()),
        }

    def pack(name, bag):
        keep = bag["keep"].astype(bool)
        m = metrics_bundle(bag["gt"], bag["mass_seg"], bag["dice_seg"], keep)
        m["name"] = name
        m["thr"] = ts["tau_prob"]
        m["tau_logit"] = ts["tau_logit"]
        m["presence_auroc"] = binary_auroc(bag["gt"] > 0, bag["cls_prob"])
        m.update(tf_from_keep(bag["gt"], keep))
        return m

    ones = np.ones(len(test["gt"]), dtype=bool)
    raw_t = metrics_bundle(test["gt"], test["mass_seg"], test["dice_seg"], ones)
    rows = [{
        "name": "TEST_seg_raw",
        "overall_dice": raw_t["overall_dice"],
        "nonempty_dice": raw_t["nonempty_dice"],
        "empty_fp_rate": raw_t["empty_fp_rate"],
        "sensitivity": raw_t["sensitivity"],
        "thr": None,
    }]
    tf_rows = []
    for split_name, bag in [("VAL", val), ("TEST", test)]:
        m = pack(f"{split_name}_learned_tau", bag)
        rows.append({
            "name": m["name"],
            "overall_dice": m["overall_dice"],
            "nonempty_dice": m["nonempty_dice"],
            "empty_fp_rate": m["empty_fp_rate"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
            "thr": m["thr"],
            "tau_logit": m["tau_logit"],
            "presence_auroc": m["presence_auroc"],
        })
        tf_rows.append(m)
        safe_print(
            f"==== {split_name} TF (presence gate) τ={ts['tau_prob']:.3f} ===="
        )
        safe_print(
            f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} "
            f"(pos={m['n_pos']} neg={m['n_neg']})"
        )
        safe_print(
            f"  sens(recall)={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"prec={m['precision']:.3f} F1={m['f1']:.3f} acc={m['accuracy']:.3f} "
            f"AUROC={m['presence_auroc']:.3f}"
        )
        safe_print(
            f"  SEG@gate ne={m['nonempty_dice']:.4f} empty_fp_rate={m['empty_fp_rate']:.3f}"
        )

    write_csv(OUT / "final_metrics.csv", rows)
    report = {
        "design": {
            "encoder": "BT frozen → one shared μ (+ skips)",
            "cls": "Decoder_cls(μ) → presence_logit",
            "tau": "learnable tau_logit; decision = presence − tau; keep iff decision≥0",
            "seg": "Decoder_seg(μ, skips) nonempty-only",
            "infer": "cls then SEG gated by learned τ (no hand-picked thr)",
        },
        "learned_tau": ts,
        "best_cls_auroc": best_auroc,
        "best_seg_val_ne": best_ne,
        "best_tau_score": best_score,
        "tf_primary": [
            {k: m[k] for k in (
                "name", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
                "precision", "f1", "accuracy", "presence_auroc",
                "nonempty_dice", "empty_fp_rate", "thr",
            )}
            for m in tf_rows
        ],
        "rows": rows,
        "ref_presence_tau020": {"nonempty_dice": 0.533, "empty_fp_rate": 0.269},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# BT cascade + learnable τ",
        "",
        "Primary view: presence T/F (medical screening). SEG dice secondary.",
        f"learned τ_prob≈{ts['tau_prob']:.4f} (logit={ts['tau_logit']:.4f})",
        f"CLS AUROC={best_auroc:.4f} | SEG VAL ne≈{best_ne:.4f}",
        "",
        "## Presence T/F (gate)",
    ]
    for m in tf_rows:
        lines += [
            f"### {m['name']}",
            f"- TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}",
            f"- sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"prec={m['precision']:.3f} F1={m['f1']:.3f} AUROC={m['presence_auroc']:.3f}",
            f"- SEG@gate: ne={m['nonempty_dice']:.4f} empty_fp={m['empty_fp_rate']:.3f}",
            "",
        ]
    lines.append("## SEG / other")
    for r in rows:
        lines.append(
            f"- {r['name']}: ne={r['nonempty_dice']:.4f} fp={r['empty_fp_rate']:.3f}"
            + (f" τ={r['thr']:.3f}" if r.get("thr") is not None else "")
        )
    (OUT / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "DESIGN.md").write_text(
        "\n".join([
            "# BT cascade + learnable τ",
            "",
            "```",
            "x → BT Encoder (frozen) → μ (+ skips)   # shared latent",
            "      ├─ μ → Decoder_cls → presence − τ",
            "      └─ μ + skips → Decoder_seg → mask",
            "",
            "infer: if decision ≥ 0 then SEG else blank",
            "```",
            "",
            "Same μ on purpose: encoder frozen, only decoders specialize.",
            f"BT ckpt: `{bt_ckpt}`",
        ]),
        encoding="utf-8",
    )
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
