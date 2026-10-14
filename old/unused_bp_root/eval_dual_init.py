"""Frozen S5 + none gated merge. No training."""
from __future__ import annotations

import json

import torch

from data_load_BT import datasets
from models_dual_dec import DualVAE
from train_dual_dec import BATCH, load_s5_none, pack, eval_heads
from train_original_bp import BASE, DEVICE, NUM_WORKERS, RESULT_DIR, safe_print

OUT = RESULT_DIR / "dual_dec"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=BATCH, num_workers=NUM_WORKERS)
    vae = DualVAE(base=BASE)
    load_s5_none(vae)
    vae.to(DEVICE).eval()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    rec = {
        "note": "frozen s5+none, skip * sigmoid(max(plain))",
        "val": {k: pack(v) for k, v in eval_heads(vae, data.val_loader).items()},
        "test": {k: pack(v) for k, v in eval_heads(vae, data.test_loader).items()},
    }
    (OUT / "init_merge.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    safe_print("VAL ", rec["val"])
    safe_print("TEST", rec["test"])


if __name__ == "__main__":
    main()
