"""Lesion-probability head on frozen none (OrigVAE).

BP 30%: adds a none+cls column to results/bp/panel.png.
COVID: heads for ratios that already have a finished none checkpoint.
Same loss as S5+cls: BCEWithLogits, p = sigmoid(Head(mu)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from pipeline.cases import overlay_rgb
from pipeline.data import datasets
from pipeline.metrics import binary_auroc, metrics_bundle
from pipeline.models_presence import LesionPresenceHead
from pipeline.stage_panel import OUT as BP_OUT
from pipeline.stage_panel import load_none, load_s5, xy
from pipeline.stage_segment import make_bundle
from pipeline.engine import (
    DEVICE,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    safe_load,
    safe_print,
)

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 40
ES = 10
POS_WEIGHT = 2.0
COVID_RATIOS = (0.1, 0.5)


def none_mu_logits(model, x):
    mu, _lv, _z = model.encoder(x)
    return mu, model.decoder(mu)


def s5_mu_logits(model, x):
    mu, _lv, _z, skips = model.encoder.forward_features(x)
    return mu, model.decoder(mu, skips)


def train_head(model, forward, train_loader, val_loader, head_path: Path, report_path: Path):
    if head_path.exists() and report_path.exists():
        meta = json.loads(report_path.read_text(encoding="utf-8"))
        head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
        head.load_state_dict(safe_load(head_path, map_location="cpu")["head"])
        head.eval()
        safe_print(f"skip {report_path} tau={meta['tau']:.2f}")
        return head, float(meta["tau"]), meta

    head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_auroc, bad, best_ep = -1.0, 0, 0
    for ep in range(1, EPOCHS + 1):
        head.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = xy(batch)
            with torch.no_grad():
                mu, _logits = forward(model, x)
            logit = head(mu)
            target = (y.flatten(1).sum(1) > 0).float()
            loss = bce(logit, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        head.eval()
        scores, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = xy(batch)
                mu, _logits = forward(model, x)
                scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
                gts.append((y.flatten(1).sum(1) > 0).detach().cpu().numpy())
        auroc = binary_auroc(np.concatenate(gts), np.concatenate(scores))
        safe_print(f"[{head_path.parent.name} cls {ep}/{EPOCHS}] bce={total/max(1,n):.4f} auroc={auroc:.4f}")
        if auroc > best_auroc + 1e-4:
            best_auroc, bad, best_ep = auroc, 0, ep
            atomic_torch_save({"head": head.state_dict(), "ep": ep, "auroc": auroc}, head_path)
        else:
            bad += 1
            if bad >= ES:
                safe_print(f"early stop @ {ep}")
                break
    head.load_state_dict(safe_load(head_path, map_location="cpu")["head"])
    head.eval()
    tau, test_stats = None, None
    return head, best_ep, {"best_ep": best_ep, "val_auroc": best_auroc}


@torch.inference_mode()
def eval_gate(model, head, forward, loader):
    dices, gts, masses, scores = [], [], [], []
    for batch in loader:
        x, y = xy(batch)
        mu, logits = forward(model, x)
        dices.append(batch_dice(logits, y))
        ps, gs = batch_pred_gt_sums(logits, y)
        masses.append(ps)
        gts.append(gs)
        scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
    dice = np.concatenate(dices)
    gt = np.concatenate(gts)
    mass = np.concatenate(masses)
    score = np.concatenate(scores)
    best_thr, best_key = 0.5, None
    for i in range(1, 100):
        thr = i / 100.0
        keep = score >= thr
        m = metrics_bundle(gt, mass, dice, keep)
        key = (m["overall_dice"], -m["empty_fp_rate"])
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr
    test_m = metrics_bundle(gt, mass, dice, score >= best_thr)
    return best_thr, {
        "dice": float(test_m["overall_dice"]),
        "nonempty_dice": float(test_m["nonempty_dice"]),
        "empty_fp_rate": float(test_m["empty_fp_rate"]),
    }


def train_bp_none():
    data = datasets.bp_train_30(batch_size=8, num_workers=0, pin_memory=False)
    model = load_none()
    head_path = BP_OUT / "none_head_best.pt"
    report_path = BP_OUT / "none_cls_report.json"
    head, best_ep, meta = train_head(
        model, none_mu_logits, data.train_loader, data.val_loader, head_path, report_path,
    )
    if "tau" not in meta:
        tau, _stats = eval_gate(model, head, none_mu_logits, data.val_loader)
        dices, gts, masses, scores = [], [], [], []
        with torch.no_grad():
            for batch in data.test_loader:
                x, y = xy(batch)
                mu, logits = none_mu_logits(model, x)
                dices.append(batch_dice(logits, y))
                ps, gs = batch_pred_gt_sums(logits, y)
                masses.append(ps)
                gts.append(gs)
                scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
        test_m = metrics_bundle(
            np.concatenate(gts), np.concatenate(masses), np.concatenate(dices),
            np.concatenate(scores) >= tau,
        )
        meta = {
            "dataset": "bp_train_30",
            "loss": "BCEWithLogits",
            "probability": "p = sigmoid(Head(mu)) = P(lesion present | slice)",
            "tau": tau,
            "best_ep": best_ep if isinstance(best_ep, int) else meta.get("best_ep"),
            "val_auroc": meta.get("val_auroc"),
            "test": {
                "dice": float(test_m["overall_dice"]),
                "nonempty_dice": float(test_m["nonempty_dice"]),
                "empty_fp_rate": float(test_m["empty_fp_rate"]),
            },
        }
        report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        safe_print(f"bp none+cls tau={tau:.2f} test dice={meta['test']['dice']:.4f}")
    else:
        tau = float(meta["tau"])
    return model, head, tau


def train_covid(ratios=COVID_RATIOS):
    points = []
    for ratio in ratios:
        pct = int(round(ratio * 100))
        _full, train_loader, val_loader, test_loader, n_train = make_bundle(ratio, 4)
        ckpt = ROOT / "results" / "covid" / "label_ratio" / f"r{pct:02d}" / "none" / "seg_best.pt"
        out = ROOT / "results" / "covid" / "label_ratio" / f"r{pct:02d}" / "none_cls"
        out.mkdir(parents=True, exist_ok=True)
        from pipeline.stage_panel import load_none as _ln
        model = _ln()
        # load_none points at the BP checkpoint. Load the COVID one instead.
        state = safe_load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        head_path = out / "head_best.pt"
        report_path = out / "report.json"
        head, best_ep, meta = train_head(
            model, none_mu_logits, train_loader, val_loader, head_path, report_path,
        )
        if "tau" not in meta:
            tau, _val = eval_gate(model, head, none_mu_logits, val_loader)
            dices, gts, masses, scores = [], [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = xy(batch)
                    mu, logits = none_mu_logits(model, x)
                    dices.append(batch_dice(logits, y))
                    ps, gs = batch_pred_gt_sums(logits, y)
                    masses.append(ps)
                    gts.append(gs)
                    scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
            test_m = metrics_bundle(
                np.concatenate(gts), np.concatenate(masses), np.concatenate(dices),
                np.concatenate(scores) >= tau,
            )
            meta = {
                "dataset": "covid",
                "ratio": ratio,
                "mode": "none_cls",
                "n_train": n_train,
                "loss": "BCEWithLogits",
                "probability": "p = sigmoid(Head(mu)) = P(lesion present | slice)",
                "tau": tau,
                "best_ep": best_ep if isinstance(best_ep, int) else meta.get("best_ep"),
                "val_auroc": meta.get("val_auroc"),
                "test": {
                    "dice": float(test_m["overall_dice"]),
                    "nonempty_dice": float(test_m["nonempty_dice"]),
                    "empty_fp_rate": float(test_m["empty_fp_rate"]),
                },
            }
            report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        points.append(meta)
        safe_print(f"covid r{pct} none+cls dice={meta['test']['dice']:.4f} tau={meta['tau']:.2f}")
        del model, head
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return points


@torch.inference_mode()
def render_bp(none, none_head, none_tau):
    s5 = load_s5()
    s5_meta = json.loads((BP_OUT / "cls_report.json").read_text(encoding="utf-8"))
    s5_tau = float(s5_meta["tau"])
    s5_head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
    s5_head.load_state_dict(safe_load(BP_OUT / "head_best.pt", map_location="cpu")["head"])
    s5_head.eval()
    data = datasets.bp_train_30(batch_size=8, num_workers=0, pin_memory=False)
    names = list(data.test.filenames)
    rows = []
    seen = 0
    for batch in data.test_loader:
        x, y = xy(batch)
        mu_s, logits_s = s5_mu_logits(s5, x)
        mu_n, logits_n = none_mu_logits(none, x)
        ps = torch.sigmoid(s5_head(mu_s))
        pn = torch.sigmoid(none_head(mu_n))
        pred_s = (torch.sigmoid(logits_s) > 0.5).float()
        pred_n = (torch.sigmoid(logits_n) > 0.5).float()
        for i in range(x.shape[0]):
            gt = y[i, 0].detach().cpu().numpy() > 0.5
            sm = pred_s[i, 0].detach().cpu().numpy() > 0.5
            nm = pred_n[i, 0].detach().cpu().numpy() > 0.5
            p_s = float(ps[i])
            p_n = float(pn[i])
            sg = sm if p_s >= s5_tau else np.zeros_like(sm)
            ng = nm if p_n >= none_tau else np.zeros_like(nm)

            def dsc(mask):
                inter = np.logical_and(mask, gt).sum()
                return float((2 * inter + 1e-6) / (mask.sum() + gt.sum() + 1e-6))

            rows.append({
                "name": names[seen],
                "gray": (x[i, 0].detach().cpu().numpy() * 255).astype(np.uint8),
                "gt_mask": gt,
                "none_mask": nm,
                "s5_mask": sm,
                "none_g": ng,
                "s5_g": sg,
                "gt": int(gt.sum()),
                "d_s5": dsc(sg),
                "d_s5_raw": dsc(sm),
                "d_n": dsc(ng),
                "s5_pix": int(sg.sum()),
                "n_pix": int(ng.sum()),
                "p_s": p_s,
                "p_n": p_n,
            })
            seen += 1
    cases = pick_cases(rows)
    path = draw(cases, s5_tau, none_tau)
    for title, row in cases:
        safe_print(
            f"{title} {row['name']} s5cls={row['d_s5']:.3f} nonecls={row['d_n']:.3f} "
            f"p_s={row['p_s']:.2f} p_n={row['p_n']:.2f}"
        )
    safe_print("panel", path)


def pick_cases(rows):
    used = set()

    def take(pool, key):
        pool = [r for r in pool if r["name"] not in used]
        if not pool:
            return None
        pool.sort(key=key, reverse=True)
        used.add(pool[0]["name"])
        return pool[0]

    case1 = take(
        [r for r in rows if r["gt"] > 400 and r["d_s5"] >= 0.55 and r["d_s5"] - r["d_n"] >= 0.25],
        lambda r: r["d_s5"] - r["d_n"],
    )
    case2 = take(
        [r for r in rows if r["gt"] == 0 and r["s5_pix"] < 20 and r["n_pix"] > 300],
        lambda r: r["n_pix"],
    )
    case3 = take(
        [r for r in rows if r["gt"] > 400 and r["d_s5"] - r["d_n"] >= 0.15],
        lambda r: r["d_s5"] - r["d_n"],
    )
    case4 = take(
        [r for r in rows if r["gt"] > 200 and r["d_s5"] > r["d_n"]],
        lambda r: r["d_s5"] - r["d_n"],
    )
    # If a role is missing, fill with the next largest S5+cls advantage.
    rest = [r for r in rows if r["name"] not in used]
    rest.sort(key=lambda r: r["d_s5"] - r["d_n"], reverse=True)
    chosen = []
    for title, row in (("Case1", case1), ("Case2", case2), ("Case3", case3), ("Case4", case4)):
        if row is None:
            row = rest.pop(0)
            used.add(row["name"])
        chosen.append((title, row))
    return chosen


def draw(cases, s5_tau, none_tau):
    cell, pad = 150, 4
    label_w, label_h = 64, 36
    cols = ("input", "gt", "none", "s5", "none_cls", "s5_cls")
    headers = ("Input", "GT", "none", "S5", "none+cls", "S5+cls")
    image = Image.new("RGB", (label_w + len(cols) * (cell + pad), label_h + len(cases) * (cell + pad)), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(headers):
        draw.text((label_w + i * (cell + pad) + 4, 4), text, fill=(255, 220, 0))
    draw.text((label_w + 4, 18), "green TP  red FP  white FN", fill=(180, 180, 180))
    meta = []
    for r, (title, row) in enumerate(cases):
        y = label_h + r * (cell + pad)
        draw.text((4, y + 8), title, fill=(255, 220, 0))
        gray = np.stack([row["gray"]] * 3, axis=-1)
        gt = np.zeros((256, 256, 3), np.uint8)
        gt[row["gt_mask"]] = (255, 255, 255)
        tiles = {
            "input": gray,
            "gt": gt,
            "none": overlay_rgb(row["none_mask"], row["gt_mask"]),
            "s5": overlay_rgb(row["s5_mask"], row["gt_mask"]),
            "none_cls": overlay_rgb(row["none_g"], row["gt_mask"]),
            "s5_cls": overlay_rgb(row["s5_g"], row["gt_mask"]),
        }
        for c, key in enumerate(cols):
            im = Image.fromarray(tiles[key]).resize((cell, cell), Image.NEAREST)
            image.paste(im, (label_w + c * (cell + pad), y))
        meta.append({
            "case": title,
            "file": row["name"],
            "s5_dice": row["d_s5_raw"],
            "s5_cls_dice": row["d_s5"],
            "none_cls_dice": row["d_n"],
            "p_s5": row["p_s"],
            "p_none": row["p_n"],
        })
    image.save(BP_OUT / "panel.png")
    (BP_OUT / "cases_none_cls.json").write_text(json.dumps({
        "s5_tau": s5_tau,
        "none_tau": none_tau,
        "cases": meta,
    }, indent=2), encoding="utf-8")
    return BP_OUT / "panel.png"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", default="", help="COVID ratios, e.g. 0.3,0.7,0.9. Skips BP.")
    p.add_argument("--panel", action="store_true", help="Redraw BP panel only.")
    args = p.parse_args()
    if args.panel:
        none, head, tau = train_bp_none()
        render_bp(none, head, tau)
        return
    if args.ratios.strip():
        train_covid([float(x) for x in args.ratios.split(",") if x.strip()])
        return
    none, head, tau = train_bp_none()
    render_bp(none, head, tau)
    del none, head
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    train_covid()


if __name__ == "__main__":
    main()
