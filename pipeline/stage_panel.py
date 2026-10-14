"""BP 30% Fig. 5 style cases.

Frozen S5 (already trained) plus a new lesion-probability head.
Four test slices:
  Case1  lesion kept, mostly true positive
  Case2  empty slice that raw S5 paints, and the head blanks
  Case3  lesion kept, with a visible miss or false edge
  Case4  smaller lesion kept
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
from pipeline.metrics import binary_auroc
from pipeline.models_presence import LesionPresenceHead
from pipeline.models_skip import VAE as SkipVAE
from pipeline.engine import (
    DEVICE,
    OrigVAE,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    safe_load,
    safe_print,
)

ROOT = Path(__file__).resolve().parents[1]
S5_CKPT = ROOT / "old" / "parked" / "results" / "bp_original" / "s5_concat" / "seg_best.pt"
NONE_CKPT = ROOT / "old" / "parked" / "results" / "bp_original" / "none" / "seg_best.pt"
OUT = ROOT / "results" / "bp"
EPOCHS = 40
ES = 10
POS_WEIGHT = 2.0


def load_s5():
    model = SkipVAE(base=32).to(DEVICE)
    state = safe_load(S5_CKPT, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_none():
    model = OrigVAE(base=32).to(DEVICE)
    state = safe_load(NONE_CKPT, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def xy(batch):
    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    return x, y


@torch.inference_mode()
def s5_forward(model, x):
    mu, _lv, _z, skips = model.encoder.forward_features(x)
    logits = model.decoder(mu, skips)
    return mu, logits


def train_head(model, train_loader, val_loader):
    OUT.mkdir(parents=True, exist_ok=True)
    head_path = OUT / "head_best.pt"
    report_path = OUT / "cls_report.json"
    if head_path.exists() and report_path.exists():
        meta = json.loads(report_path.read_text(encoding="utf-8"))
        head = LesionPresenceHead(32 * 32, hidden_dim=128, dropout=0.5).to(DEVICE)
        head.load_state_dict(safe_load(head_path, map_location="cpu")["head"])
        head.eval()
        safe_print(f"skip existing head tau={meta['tau']:.2f}")
        return head, float(meta["tau"])

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
                mu, _logits = s5_forward(model, x)
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
                mu, _logits = s5_forward(model, x)
                scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
                gts.append((y.flatten(1).sum(1) > 0).detach().cpu().numpy())
        auroc = binary_auroc(np.concatenate(gts), np.concatenate(scores))
        safe_print(f"[bp cls {ep}/{EPOCHS}] bce={total/max(1,n):.4f} auroc={auroc:.4f}")
        if auroc > best_auroc + 1e-4:
            best_auroc, bad, best_ep = auroc, 0, ep
            atomic_torch_save({"head": head.state_dict(), "ep": ep, "auroc": auroc}, head_path)
        else:
            bad += 1
            if bad >= ES:
                safe_print(f"early stop bp cls @ {ep}")
                break
    head.load_state_dict(safe_load(head_path, map_location="cpu")["head"])
    head.eval()
    tau = pick_tau(model, head, val_loader)
    report_path.write_text(json.dumps({
        "dataset": "bp_train_30",
        "loss": "BCEWithLogits",
        "probability": "p = sigmoid(Head(mu)) = P(lesion present | slice)",
        "tau": tau,
        "best_ep": best_ep,
        "val_auroc": best_auroc,
    }, indent=2), encoding="utf-8")
    safe_print(f"bp cls tau={tau:.2f} auroc={best_auroc:.4f}")
    return head, tau


@torch.inference_mode()
def pick_tau(model, head, loader) -> float:
    dices, gts, scores = [], [], []
    for batch in loader:
        x, y = xy(batch)
        mu, logits = s5_forward(model, x)
        dices.append(batch_dice(logits, y))
        _ps, gs = batch_pred_gt_sums(logits, y)
        gts.append(gs)
        scores.append(torch.sigmoid(head(mu)).detach().cpu().numpy())
    dice = np.concatenate(dices)
    gt = np.concatenate(gts)
    score = np.concatenate(scores)
    best_thr, best_key = 0.5, None
    for i in range(1, 100):
        thr = i / 100.0
        keep = score >= thr
        out = dice.copy()
        out[~keep & (gt == 0)] = 1.0
        out[~keep & (gt > 0)] = 0.0
        key = (float(out.mean()), -float(((~keep) & (gt == 0)).mean()))
        if best_key is None or key > best_key:
            best_key, best_thr = key, thr
    return best_thr


@torch.inference_mode()
def collect_test(model, head, none, loader, names, tau):
    rows = []
    seen = 0
    for batch in loader:
        x, y = xy(batch)
        mu, logits = s5_forward(model, x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        score = torch.sigmoid(head(mu))
        mu_n, _lv, _z = none.encoder(x)
        none_logits = none.decoder(mu_n)
        none_pred = (torch.sigmoid(none_logits) > 0.5).float()
        b = x.shape[0]
        for i in range(b):
            gt = y[i, 0].detach().cpu().numpy() > 0.5
            pr = pred[i, 0].detach().cpu().numpy() > 0.5
            pn = none_pred[i, 0].detach().cpu().numpy() > 0.5
            p = float(score[i].detach().cpu())
            keep = p >= tau
            gated = pr if keep else np.zeros_like(pr)
            inter = np.logical_and(pr, gt).sum()
            union = pr.sum() + gt.sum()
            dice = float((2 * inter + 1e-6) / (union + 1e-6))
            n_inter = np.logical_and(pn, gt).sum()
            n_union = pn.sum() + gt.sum()
            none_dice = float((2 * n_inter + 1e-6) / (n_union + 1e-6))
            rows.append({
                "name": names[seen],
                "gt": int(gt.sum()),
                "pred": int(pr.sum()),
                "none": int(pn.sum()),
                "dice": dice,
                "none_dice": none_dice,
                "xor": int(np.logical_xor(pr, pn).sum()),
                "p": p,
                "keep": bool(keep),
                "gray": (x[i, 0].detach().cpu().numpy() * 255).astype(np.uint8),
                "gt_mask": gt,
                "s5": pr,
                "gated": gated,
                "none_mask": pn,
            })
            seen += 1
    return rows


def choose(rows):
    """Pick slices where the none mask and the S5 mask are not the same blob."""
    used = set()

    def take(pool, key):
        pool = [r for r in pool if r["name"] not in used]
        pool.sort(key=key, reverse=True)
        chosen = pool[0]
        used.add(chosen["name"])
        return chosen

    # S5 covers the lesion, none misses it.
    case1 = take(
        [r for r in rows if r["gt"] > 400 and r["dice"] >= 0.55 and r["none_dice"] <= 0.25],
        lambda r: r["dice"] - r["none_dice"],
    )
    # Empty slice: only one of the two models paints a false lesion.
    case2 = take(
        [r for r in rows if r["gt"] == 0 and abs(r["pred"] - r["none"]) > 400],
        lambda r: abs(r["pred"] - r["none"]),
    )
    # none covers the lesion, S5 misses it.
    case3 = take(
        [r for r in rows if r["gt"] > 400 and r["none_dice"] >= 0.55 and r["dice"] <= 0.25],
        lambda r: r["none_dice"] - r["dice"],
    )
    # Both fire, but the blobs disagree.
    case4 = take(
        [r for r in rows if r["gt"] > 400 and r["xor"] > 800 and min(r["dice"], r["none_dice"]) > 0.15],
        lambda r: r["xor"],
    )
    return [("Case1", case1), ("Case2", case2), ("Case3", case3), ("Case4", case4)]


def panel(cases):
    cell = 160
    pad = 4
    label_w, label_h = 64, 36
    cols = ("input", "gt", "none", "s5", "s5cls")
    headers = ("Input", "GT", "none", "S5", "S5+cls")
    w = label_w + len(cols) * (cell + pad)
    h = label_h + len(cases) * (cell + pad)
    image = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(headers):
        draw.text((label_w + i * (cell + pad) + 8, 4), text, fill=(255, 220, 0))
    draw.text((label_w + 8, 18), "green TP  red FP  white FN", fill=(180, 180, 180))
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
            "s5": overlay_rgb(row["s5"], row["gt_mask"]),
            "s5cls": overlay_rgb(row["gated"], row["gt_mask"]),
        }
        for c, key in enumerate(cols):
            im = Image.fromarray(tiles[key]).resize((cell, cell), Image.NEAREST)
            image.paste(im, (label_w + c * (cell + pad), y))
            Image.fromarray(tiles[key]).save(OUT / f"{title}_{key}.png")
        meta.append({
            "case": title,
            "file": row["name"],
            "gt": row["gt"],
            "s5_pred": row["pred"],
            "p": row["p"],
            "keep": row["keep"],
            "s5_dice": row["dice"],
        })
    image.save(OUT / "panel.png")
    (OUT / "cases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return OUT / "panel.png"


def main():
    data = datasets.bp_train_30(batch_size=8, num_workers=0, pin_memory=False)
    safe_print(f"bp30 train/val/test {len(data.train)}/{len(data.val)}/{len(data.test)}")
    model = load_s5()
    head, tau = train_head(model, data.train_loader, data.val_loader)
    none = load_none()
    rows = collect_test(model, head, none, data.test_loader, list(data.test.filenames), tau)
    cases = choose(rows)
    path = panel(cases)
    for title, row in cases:
        safe_print(
            f"{title} {row['name']} gt={row['gt']} s5={row['pred']} "
            f"p={row['p']:.2f} keep={row['keep']} "
            f"s5_dice={row['dice']:.3f} none_dice={row['none_dice']:.3f} xor={row['xor']}"
        )
    safe_print("panel", path)


if __name__ == "__main__":
    main()
