"""Re-evaluate every BP experiment on val/test with the same metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_load_BT import datasets
from eval_bp_best import DEVICE, metrics_batch, summarize
from models_bp_skip import VAE as SkipVAE
from models_presence import PresenceVAE
from models_proto import ProtoHead, encode_spatial
from models_region_seed import VAE as SeedVAE
from train_original_bp import (
    BASE,
    BT_BEST,
    OrigVAE,
    RESULT_DIR,
    load_bt_into_encoder,
    safe_load,
    safe_print,
)

ROOT = Path(__file__).resolve().parent
OUT = RESULT_DIR / "COMPARE.json"


@torch.no_grad()
def run_predict(predict, loader, name: str):
    bags = {k: [] for k in ["dice", "iou", "prec", "rec", "gt", "pred"]}
    for batch in loader:
        flair, seg = batch[0], batch[1]
        if flair.ndim == 3:
            flair = flair.unsqueeze(1)
        if seg.ndim == 3:
            seg = seg.unsqueeze(1)
        out = predict(flair.to(DEVICE))
        m = metrics_batch(out, seg.to(DEVICE))
        for k in bags:
            bags[k].append(m[k])
    arrs = {k: np.concatenate(v) for k, v in bags.items()}
    s = summarize(arrs)
    safe_print(name, json.dumps({k: s[k] for k in ("mean_dice", "mean_dice_nonempty", "empty_fp_rate")}, indent=2))
    return s


def load_vae(cls, ckpt: Path):
    m = cls(base=BASE)
    st = safe_load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"])
    ep = st.get("best_ep", st.get("ep"))
    del st
    m.to(DEVICE).eval()
    return m, ep


def main():
    data = datasets.bp_train_30(batch_size=8, num_workers=0)
    rows = []

    def add(name, note, ckpt, predict, extra=None):
        safe_print("====", name)
        rec = {
            "name": name,
            "note": note,
            "ckpt": str(ckpt) if ckpt else None,
            "val": run_predict(predict, data.val_loader, f"VAL {name}"),
            "test": run_predict(predict, data.test_loader, f"TEST {name}"),
        }
        if extra:
            rec.update(extra)
        rows.append(rec)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    none, ep = load_vae(OrigVAE, RESULT_DIR / "none" / "seg_best.pt")
    add("none", "original VAE, full FT, val-loss best @48", RESULT_DIR / "none" / "seg_best.pt",
        lambda x, m=none: m(x)[0], {"best_ep": ep})

    s5, ep = load_vae(SkipVAE, RESULT_DIR / "s5_concat" / "seg_best.pt")
    add("s5_concat", "B0–B3 concat skip, stopped, val-loss best @20", RESULT_DIR / "s5_concat" / "seg_best.pt",
        lambda x, m=s5: m(x)[0], {"best_ep": ep})
    del s5
    torch.cuda.empty_cache()

    seed, ep = load_vae(SeedVAE, ROOT / "results" / "bp_region_seed" / "seg_best.pt")
    add("region_seed", "learnable h, stopped ~ep30, best @23", ROOT / "results" / "bp_region_seed" / "seg_best.pt",
        lambda x, m=seed: m(x)[0], {"best_ep": ep})
    del seed
    torch.cuda.empty_cache()

    frz, ep = load_vae(OrigVAE, RESULT_DIR / "none_freeze" / "seg_best.pt")
    add("none_freeze", "BT encoder frozen, decoder only, stopped ~ep37", RESULT_DIR / "none_freeze" / "seg_best.pt",
        lambda x, m=frz: m(x)[0], {"best_ep": ep})
    del frz
    torch.cuda.empty_cache()

    efp, ep = load_vae(OrigVAE, RESULT_DIR / "none_emptyfp" / "seg_last.pt")
    add("none_emptyfp", "none best + empty-FP loss λ=6, last ep (floor never met)",
        RESULT_DIR / "none_emptyfp" / "seg_last.pt",
        lambda x, m=efp: m(x)[0], {"best_ep": ep})
    del efp
    torch.cuda.empty_cache()

    tau = 0.2
    wrap = PresenceVAE(none, base=BASE)
    st = safe_load(RESULT_DIR / "none_presence" / "head_best.pt", map_location="cpu")
    wrap.head.load_state_dict(st["head"])
    wrap.to(DEVICE).eval()
    wrap.gate_thr = tau
    add("none_presence", f"frozen none + presence gate τ={tau}", RESULT_DIR / "none_presence" / "head_best.pt",
        lambda x, m=wrap: m(x)[0], {"tau": tau, "best_ep": st.get("best_ep")})
    del wrap
    torch.cuda.empty_cache()

    from train_original_bp import OrigEncoder
    enc = OrigEncoder(base=BASE)
    load_bt_into_encoder(enc, BT_BEST)
    enc.to(DEVICE).eval()
    head = ProtoHead().to(DEVICE)
    st = safe_load(RESULT_DIR / "bt_proto" / "head_best.pt", map_location="cpu")
    head.load_state_dict(st["head"])
    head.eval()

    def proto_pred(x, e=enc, h=head):
        mu, mid = encode_spatial(e, x)
        return h(mu, mid, out_size=x.shape[-2:])

    add("bt_proto", "frozen BT spatial μ+16 + FG/BG proto, nonempty best @75",
        RESULT_DIR / "bt_proto" / "head_best.pt", proto_pred, {"best_ep": st.get("best_ep")})

    neigh_path = RESULT_DIR / "none_neighbor" / "report.json"
    if neigh_path.exists():
        neigh = json.loads(neigh_path.read_text(encoding="utf-8"))
        for rule in ("isolated_kill", "isolated_small_800"):
            rows.append({
                "name": f"neighbor_{rule}",
                "note": "post-hoc ±1 on none (no retrain); from none_neighbor/report.json",
                "ckpt": None,
                "val": {
                    "mean_dice": neigh["val"]["rules"][rule]["dice_mean"],
                    "mean_dice_nonempty": neigh["val"]["rules"][rule]["mean_dice_nonempty"],
                    "empty_fp_rate": neigh["val"]["rules"][rule]["empty_fp_rate"],
                    "mean_iou": None,
                    "mean_prec": None,
                    "mean_rec": None,
                    "mean_prec_nonempty": None,
                    "mean_rec_nonempty": None,
                },
                "test": {
                    "mean_dice": neigh["test"]["rules"][rule]["dice_mean"],
                    "mean_dice_nonempty": neigh["test"]["rules"][rule]["mean_dice_nonempty"],
                    "empty_fp_rate": neigh["test"]["rules"][rule]["empty_fp_rate"],
                    "mean_iou": None,
                    "mean_prec": None,
                    "mean_rec": None,
                    "mean_prec_nonempty": None,
                    "mean_rec_nonempty": None,
                },
            })

    report = {
        "dataset": "bp_train_30",
        "protocol": "same val/test loaders; dice empty-empty=1; thr=0.5",
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print("wrote", OUT)


if __name__ == "__main__":
    main()
