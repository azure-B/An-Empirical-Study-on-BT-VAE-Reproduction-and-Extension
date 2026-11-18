"""Compare VAE | BT-VAE | BT-VAE+C | BT-VAE+S | BT-VAE+S&C(Ours).

C = lesion binary classification gate; S = skip connections.
Lazy disk loader only (원본 CovidDataset eager RAM 적재 금지).

  py -3.12 -m pipeline compare --ratio 0.3

Checkpoints:
  checkpoints/covid/vae_seg_best.pt                         → VAE
  results/.../rXX/BT-VAE/seg_best.pt                        → BT-VAE  (legacy: none/)
  results/.../rXX/BT-VAE+C/head_best.pt + report.json       → BT-VAE+C (legacy: none_cls/)
  results/.../rXX/BT-VAE+S/seg_best.pt                      → BT-VAE+S (legacy: s5/)
  results/.../rXX/BT-VAE+SC/head_best.pt + report.json      → BT-VAE+S&C(Ours) (legacy: s5_cls/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_load_BT import CovidDataset  # noqa: E402
from pipeline.models_presence import LesionPresenceHead  # noqa: E402
from pipeline.models_skip import VAE as SkipVAE  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_load(path: Path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

OUT_DIR = ROOT / "results" / "covid" / "compare"
METRICS_JSON = OUT_DIR / "compare_metrics.json"
FIG_PNG = OUT_DIR / "segmentation_performance_bar.png"

from pipeline.model_names import DISPLAY, LEGACY_DIRS  # noqa: E402

MODELS = [(name, name) for name in DISPLAY]  # (display, key) — key == display
METRIC_KEYS = ["dice", "miou", "precision", "recall"]
METRIC_LABELS = ["Dice", "mIoU", "Precision", "Recall"]


def _batch_metrics(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> dict[str, float]:
    """pred/target: [B,1,H,W] binary float. Returns per-batch micro averages."""
    p = pred_bin.reshape(-1)
    t = target_bin.reshape(-1)
    inter = (p * t).sum()
    psum, tsum = p.sum(), t.sum()
    union = psum + tsum - inter
    return {
        "dice": float((2 * inter) / (psum + tsum + 1e-6)),
        "miou": float(inter / (union + 1e-6)),
        "precision": float(inter / (psum + 1e-6)),
        "recall": float(inter / (tsum + 1e-6)),
    }


def _load_orig_vae():
    import importlib.util

    path = ROOT / "BT-VAE_CODE" / "models1.py"
    if not path.exists():
        path = ROOT / "original" / "models1.py"
    if not path.exists():
        path = ROOT / "models1.py"
    spec = importlib.util.spec_from_file_location("btvae_models1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.VAE(base=32)


@torch.no_grad()
def eval_segmenter(model, loader, device, head=None, tau: float | None = None) -> dict[str, float]:
    model.eval()
    if head is not None:
        head.eval()
    acc = {k: [] for k in METRIC_KEYS}
    for batch in tqdm(loader, desc="eval", leave=False):
        images, targets, _, _ = batch
        x = images.unsqueeze(1).to(device)
        y = targets.unsqueeze(1).to(device)
        y_bin = (y > 0.5).float()

        # Deterministic μ decode (same as training eval)
        if hasattr(model.encoder, "forward_features"):
            # BT-VAE+S / BT-VAE+S&C
            mu, _lv, _z, skips = model.encoder.forward_features(x)
            logits = model.decoder(mu, skips)
        else:
            # VAE / BT-VAE / BT-VAE+C  (BT-VAE_CODE Decoder(mu))
            mu, _lv, _z = model.encoder(x)
            logits = model.decoder(mu)

        pred = (torch.sigmoid(logits) > 0.5).float()
        if head is not None and tau is not None:
            # C: blank mask when P(lesion) < τ
            keep = torch.sigmoid(head(mu)) >= float(tau)
            pred = pred.clone()
            pred[~keep] = 0

        m = _batch_metrics(pred, y_bin)
        for k in METRIC_KEYS:
            acc[k].append(m[k])
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def _ratio_dir(ratio: float) -> Path:
    return ROOT / "results" / "covid" / "label_ratio" / f"r{int(round(ratio * 100)):02d}"


def _first_existing(root: Path, names: tuple[str, ...], filename: str) -> Path | None:
    for n in names:
        p = root / n / filename
        if p.exists():
            return p
    return None


def resolve_ckpts(ratio: float) -> dict[str, dict]:
    """Map display name → {seg, head?, tau?, kind} where kind is BT-VAE|BT-VAE+S."""
    r = _ratio_dir(ratio)
    out = {}

    vae = ROOT / "checkpoints" / "covid" / "vae_seg_best.pt"
    if vae.exists():
        out["VAE"] = {"seg": vae, "kind": "BT-VAE"}  # same arch, no BT init

    bt = _first_existing(r, LEGACY_DIRS["BT-VAE"], "seg_best.pt")
    if bt is not None:
        out["BT-VAE"] = {"seg": bt, "kind": "BT-VAE"}

    bt_c_h = _first_existing(r, LEGACY_DIRS["BT-VAE+C"], "head_best.pt")
    bt_c_r = _first_existing(r, LEGACY_DIRS["BT-VAE+C"], "report.json")
    if bt is not None and bt_c_h is not None and bt_c_r is not None:
        tau = json.loads(bt_c_r.read_text(encoding="utf-8")).get("tau", 0.5)
        out["BT-VAE+C"] = {"seg": bt, "head": bt_c_h, "tau": tau, "kind": "BT-VAE"}

    bt_s = _first_existing(r, LEGACY_DIRS["BT-VAE+S"], "seg_best.pt")
    if bt_s is not None:
        out["BT-VAE+S"] = {"seg": bt_s, "kind": "BT-VAE+S"}

    bt_sc_h = _first_existing(r, LEGACY_DIRS["BT-VAE+S&C(Ours)"], "head_best.pt")
    bt_sc_r = _first_existing(r, LEGACY_DIRS["BT-VAE+S&C(Ours)"], "report.json")
    if bt_s is not None and bt_sc_h is not None and bt_sc_r is not None:
        tau = json.loads(bt_sc_r.read_text(encoding="utf-8")).get("tau", 0.5)
        out["BT-VAE+S&C(Ours)"] = {"seg": bt_s, "head": bt_sc_h, "tau": tau, "kind": "BT-VAE+S"}
    return out


def build_model(kind: str, device: str):
    if kind == "BT-VAE+S":
        return SkipVAE(base=32, dropout_rate=0.1).to(device)
    return _load_orig_vae().to(device)


def run_eval(ratio: float, batch_size: int = 4) -> dict:
    test_dir = ROOT / "datasets" / "covid_8_1_1" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"{test_dir} 없음. 먼저:\n  py -3 scripts/download_and_prepare.py"
        )
    ds = CovidDataset(test_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[data] lazy test n={len(ds)} device={DEVICE}")

    ckpts = resolve_ckpts(ratio)
    results = {}
    for label, key in MODELS:
        if key not in ckpts:
            print(f"[skip] {label}: checkpoint 없음")
            continue
        info = ckpts[key]
        model = build_model(info["kind"], DEVICE)
        st = safe_load(info["seg"], map_location="cpu")
        state = st.get("model", st.get("vae", st))
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # common keys
        if isinstance(state, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in state):
            model.load_state_dict(state, strict=False)
        elif isinstance(state, dict) and "encoder" in st:
            model.encoder.load_state_dict(st["encoder"], strict=False)
            if "decoder" in st:
                model.decoder.load_state_dict(st["decoder"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model.to(DEVICE)

        head = None
        tau = None
        if "head" in info:
            # infer channels from a dummy forward
            with torch.no_grad():
                x0 = next(iter(loader))[0][:1].unsqueeze(1).to(DEVICE)
                if info["kind"] == "BT-VAE+S":
                    mu, _, _, _ = model.encoder.forward_features(x0)
                else:
                    mu, _, _ = model.encoder(x0)
                ch = int(mu.shape[1])
            head = LesionPresenceHead(ch).to(DEVICE)
            hst = safe_load(info["head"], map_location="cpu")
            head.load_state_dict(hst["head"] if "head" in hst else hst, strict=True)
            tau = float(info["tau"])

        metrics = eval_segmenter(model, loader, DEVICE, head=head, tau=tau)
        results[label] = metrics
        print(f"[{label}] " + " ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS))
        del model, head
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"ratio": ratio, "device": DEVICE, "n_test": len(ds), "models": results}
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_JSON}")
    return payload


def plot_bar(payload: dict | None = None) -> Path:
    if payload is None:
        if not METRICS_JSON.exists():
            raise FileNotFoundError(METRICS_JSON)
        payload = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

    models = payload.get("models") or {}
    labels = [lab for lab, _ in MODELS if lab in models]
    if not labels:
        raise RuntimeError("그릴 모델 결과가 없습니다.")

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]
    x = np.arange(len(METRIC_LABELS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    for i, lab in enumerate(labels):
        vals = [models[lab][k] for k in METRIC_KEYS]
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=lab, color=colors[i % len(colors)], edgecolor="white", linewidth=0.4)

    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_title("Segmentation Performance by Model and Metric")
    ax.yaxis.grid(True, linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=min(5, len(labels)), frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_PNG}")
    return FIG_PNG


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    if args.plot_only:
        plot_bar()
        return
    payload = run_eval(args.ratio, args.batch_size)
    if payload.get("models"):
        plot_bar(payload)
    else:
        print("체크포인트가 없어 그래프를 건너뜁니다. 학습 후 다시 실행하세요.")


if __name__ == "__main__":
    main()
