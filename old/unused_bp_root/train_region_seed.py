"""BP SEG with original structure + region seed (no skip). original/ is not modified."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_region_seed import VAE as SeedVAE
from train_original_bp import (
    BASE,
    BETA_LOCAL,
    BT_BEST,
    BT_LAST,
    DEVICE,
    NUM_WORKERS,
    RECON_WEIGHT,
    SEG_BATCH,
    SEG_EPOCHS,
    SEG_ES_PATIENCE,
    SEG_LR,
    atomic_torch_save,
    evaluate_test,
    load_bt_into_encoder,
    safe_load,
    safe_print,
    seg_step_loss,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "bp_region_seed"
H_INIT = 0.3
# Original-sized decoder + 1 scalar. Batch 16 matches none; avoid mid-run resume if possible.
SEED_BATCH = 16


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg_best = OUT_DIR / "seg_best.pt"
    seg_last = OUT_DIR / "seg_last.pt"
    bt_ckpt = BT_BEST if BT_BEST.exists() else BT_LAST
    if not bt_ckpt.exists():
        raise FileNotFoundError(f"BT ckpt missing: {bt_ckpt}")

    data = datasets.bp_train_30(batch_size=SEED_BATCH, num_workers=NUM_WORKERS)
    safe_print(
        "DEVICE:", DEVICE, "SEG region-seed BP train_30:",
        len(data.train), len(data.val), len(data.test), f"batch={SEED_BATCH}",
    )
    safe_print("BT encoder:", bt_ckpt, "h_init:", H_INIT)

    vae = SeedVAE(base=BASE, h_init=H_INIT).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=SEG_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.95, patience=3, min_lr=1e-8)

    start_ep = 1
    best_loss = float("inf")
    best_ep = 0
    if seg_last.exists():
        st = safe_load(seg_last, map_location="cpu")
        vae.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            try:
                sched.load_state_dict(st["scheduler"])
            except Exception:
                pass
        start_ep = int(st.get("ep", 0)) + 1
        best_loss = float(st.get("best_loss", best_loss))
        best_ep = int(st.get("best_ep", 0))
        del st
        safe_print(f"resume from ep {start_ep}, best_loss={best_loss:.4f} h={vae.decoder.h_value():.4f}")
    else:
        load_bt_into_encoder(vae.encoder, bt_ckpt)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    history = []
    for ep in range(start_ep, SEG_EPOCHS + 1):
        vae.train()
        tr_loss = tr_dice = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            flair = batch[0].to(DEVICE)
            seg = batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, dice_v, _ = seg_step_loss(vae, flair, seg)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            tr_dice += dice_v
        ntr = max(1, len(data.train_loader))
        train_loss = tr_loss / ntr

        vae.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in data.val_loader:
                loss, _, _ = seg_step_loss(vae, batch[0].to(DEVICE), batch[1].to(DEVICE))
                va_loss += float(loss.item())
        val_loss = va_loss / max(1, len(data.val_loader))
        sched.step(val_loss)
        h_now = vae.decoder.h_value()
        safe_print(
            f"[SEG seed] ep {ep}/{SEG_EPOCHS} train={train_loss:.4f} val={val_loss:.4f} "
            f"dice_term={tr_dice/ntr:.4f} h={h_now:.4f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": train_loss, "val_loss": val_loss, "h": h_now})

        payload = {
            "ep": ep,
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_loss": best_loss,
            "best_ep": best_ep,
            "h": h_now,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            best_ep = ep
            payload["best_loss"] = best_loss
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, seg_best)
            safe_print(f"  saved SEG_BEST loss={best_loss:.4f} h={h_now:.4f}")
        atomic_torch_save(payload, seg_last)

        if best_ep > 0 and (ep - best_ep) >= SEG_ES_PATIENCE:
            safe_print(f"early stop: no val best for {SEG_ES_PATIENCE} epochs (best={best_loss:.4f} @{best_ep})")
            break

    if seg_best.exists():
        st = safe_load(seg_best)
        vae.load_state_dict(st["model"])
        best_ep = int(st.get("best_ep", best_ep))
        best_loss = float(st.get("best_loss", best_loss))
    metrics = evaluate_test(vae, data.test_loader)
    report = {
        "mode": "region_seed",
        "dataset": "bp_train_30",
        "bt_ckpt": str(bt_ckpt),
        "best_val_loss": best_loss,
        "best_ep": best_ep,
        "h_init": H_INIT,
        "h_best": vae.decoder.h_value(),
        "test": metrics,
        "protocol": {
            "seg_epochs": SEG_EPOCHS,
            "seg_batch": SEG_BATCH,
            "seg_lr": SEG_LR,
            "beta_local": BETA_LOCAL,
            "recon_weight": RECON_WEIGHT,
            "early_stop_patience": SEG_ES_PATIENCE,
            "selection": "best_val_loss",
            "skip": False,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST region_seed:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
