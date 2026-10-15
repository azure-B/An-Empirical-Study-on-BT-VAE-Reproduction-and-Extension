"""Pure CNN lesion classifier, no VAE encoder.

The conv stack is the S5 decoder's residual block in the down direction
(ResBlockDown, same channels as the decoder's ups). The last map goes into
the same GAP+GMP head used by S5+classification. Loss is BCEWithLogits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pipeline.config import RESULT_DIR
from pipeline.engine import DEVICE, atomic_torch_save, safe_load, safe_print
from pipeline.metrics import binary_auroc
from pipeline.models_presence import LesionPresenceHead
from pipeline.models_skip import ResBlockDown
from pipeline.stage_segment import CLS_EPOCHS, CLS_ES, CLS_POS_WEIGHT, RATIOS, make_bundle

OUT_NAME = "cnn_cls"


class PurePresenceCNN(nn.Module):
    """Image to presence logit. No μ, no skip, no VAE encoder."""

    def __init__(self, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.down = nn.Sequential(
            ResBlockDown(base, 2 * base),
            nn.Dropout2d(p=dropout),
            ResBlockDown(2 * base, 4 * base),
            nn.Dropout2d(p=dropout),
            ResBlockDown(4 * base, 8 * base),
            nn.Dropout2d(p=dropout),
            ResBlockDown(8 * base, 16 * base),
            nn.Dropout2d(p=dropout),
            ResBlockDown(16 * base, 32 * base),
            nn.Dropout2d(p=dropout),
        )
        self.head = LesionPresenceHead(32 * base, hidden_dim=128, dropout=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.stem(x))
        return self.head(h)


def _xy(batch):
    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    return x, y


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        x, y = _xy(batch)
        scores.append(torch.sigmoid(model(x)).detach().cpu().numpy())
        labels.append((y.flatten(1).sum(1) > 0).detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(bool)


def pick_tau(scores, labels):
    best = None
    pos, neg = labels, ~labels
    for i in range(1, 100):
        thr = i / 100.0
        keep = scores >= thr
        recall = float(keep[pos].mean()) if pos.any() else 0.0
        reject = float((~keep[neg]).mean()) if neg.any() else 0.0
        key = (recall + reject, reject)
        if best is None or key > best["key"]:
            best = {"thr": thr, "key": key, "recall": recall, "reject": reject}
    return best["thr"], best["recall"], best["reject"]


def train_one(ratio: float):
    pct = int(round(ratio * 100))
    out = RESULT_DIR / "label_ratio" / f"r{pct:02d}" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        safe_print(f"skip {report_path}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    _full, train_loader, val_loader, test_loader, n_train = make_bundle(ratio, 8)
    model = PurePresenceCNN().to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLS_POS_WEIGHT], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_auroc, bad, best_ep = -1.0, 0, 0
    best_path = out / "cnn_best.pt"
    for ep in range(1, CLS_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = _xy(batch)
            target = (y.flatten(1).sum(1) > 0).float()
            opt.zero_grad(set_to_none=True)
            loss = bce(model(x), target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * target.size(0)
            n += int(target.size(0))
        vs, vg = collect(model, val_loader)
        auroc = binary_auroc(vg, vs)
        safe_print(f"[cnn r{pct} {ep}/{CLS_EPOCHS}] bce={total/max(n,1):.4f} auroc={auroc:.4f}")
        if auroc > best_auroc + 1e-4:
            best_auroc, bad, best_ep = auroc, 0, ep
            atomic_torch_save({"model": model.state_dict(), "ep": ep, "auroc": auroc}, best_path)
        else:
            bad += 1
            if bad >= CLS_ES:
                safe_print(f"early stop cnn r{pct} @ {ep}")
                break
    model.load_state_dict(safe_load(best_path, map_location="cpu")["model"])
    vs, vg = collect(model, val_loader)
    tau, _vr, _vj = pick_tau(vs, vg)
    ts, tg = collect(model, test_loader)
    keep = ts >= tau
    pos, neg = tg, ~tg
    report = {
        "ratio": ratio,
        "n_train": n_train,
        "model": "PurePresenceCNN",
        "note": "ResBlockDown stack matching the decoder channels, then the S5 presence head. No VAE encoder.",
        "loss": "BCEWithLogits",
        "tau_rule": "validation Youden (lesion recall + empty reject)",
        "tau": tau,
        "best_ep": best_ep,
        "val_auroc": best_auroc,
        "test_auroc": binary_auroc(tg, ts),
        "test_lesion_recall": float(keep[pos].mean()) if pos.any() else 0.0,
        "test_empty_reject": float((~keep[neg]).mean()) if neg.any() else 0.0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print(f"TEST cnn r{pct} auroc={report['test_auroc']:.4f} tau={tau:.2f}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    rows = [train_one(r) for r in RATIOS]
    dest = RESULT_DIR / "label_ratio" / "cnn_cls_curve.json"
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    safe_print("wrote", dest)


if __name__ == "__main__":
    main()
