"""Fine-tune S5 skip with the same empty-FP penalty used on none.

Does not overwrite s5_concat/ or none/. Batch 8 for 8GB.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from train_emptyfp import (
    EPOCHS,
    ES_PATIENCE,
    FT_LR,
    eval_split,
    step_loss,
)
from train_original_bp import (
    BASE,
    DEVICE,
    NUM_WORKERS,
    RESULT_DIR,
    atomic_torch_save,
    safe_load,
    safe_print,
)

S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
OUT_DIR = RESULT_DIR / "s5_emptyfp"
S5_BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-fp", type=float, default=1.5)
    ap.add_argument("--max-epoch", type=int, default=None)
    ap.add_argument("--es-patience", type=int, default=None)
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    lam = float(args.lambda_fp)
    limit = EPOCHS if args.max_epoch is None else int(args.max_epoch)
    patience = ES_PATIENCE if args.es_patience is None else int(args.es_patience)
    tag = args.tag or f"l{lam:g}"

    if not S5_BEST.exists():
        raise FileNotFoundError(S5_BEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.bp_train_30(batch_size=S5_BATCH, num_workers=NUM_WORKERS)
    safe_print(
        "DEVICE:", DEVICE, "S5 + empty-FP FT from", S5_BEST,
        f"batch={S5_BATCH} lr={FT_LR} λ={lam} max_ep={limit} es={patience} tag={tag}",
    )

    vae = SkipVAE(base=BASE)
    st = safe_load(S5_BEST, map_location="cpu")
    vae.load_state_dict(st["model"])
    s5_ep = int(st.get("best_ep", -1))
    del st
    vae.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    opt = torch.optim.AdamW(vae.parameters(), lr=FT_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    base_val = eval_split(vae, data.val_loader)
    floor = float(base_val["mean_dice_nonempty"]) - 0.02
    safe_print(
        "s5 VAL ",
        f"dice={base_val['dice_mean']:.4f} nonempty={base_val['mean_dice_nonempty']:.4f} "
        f"empty_fp={base_val['empty_fp_rate']:.3f} floor={floor:.4f}",
    )

    best_fp = 2.0
    best_ep = 0
    history = [{"init_val": base_val, "s5_ep": s5_ep}]
    for ep in range(1, limit + 1):
        vae.train()
        tr = tr_extra = 0.0
        ntr = 0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, _, extra = step_loss(vae, flair, seg, empty_fp_weight=lam)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            tr_extra += extra
            ntr += 1
        val = eval_split(vae, data.val_loader)
        sched.step(val["empty_fp_rate"])
        safe_print(
            f"[s5-emptyfp] ep {ep}/{limit} loss={tr/max(1,ntr):.4f} extra={tr_extra/max(1,ntr):.4f} "
            f"val_dice={val['dice_mean']:.4f} nonempty={val['mean_dice_nonempty']:.4f} "
            f"empty_fp={val['empty_fp_rate']:.3f} ({time.time()-t0:.1f}s)"
        )
        history.append({"ep": ep, "train_loss": tr / max(1, ntr), "val": val})
        ok = val["mean_dice_nonempty"] is not None and val["mean_dice_nonempty"] >= floor
        payload = {
            "ep": ep,
            "model": vae.state_dict(),
            "optimizer": opt.state_dict(),
            "best_fp": best_fp,
            "best_ep": best_ep,
            "s5_ckpt": str(S5_BEST),
        }
        if ok and val["empty_fp_rate"] < best_fp:
            best_fp = float(val["empty_fp_rate"])
            best_ep = ep
            payload["best_fp"] = best_fp
            payload["best_ep"] = best_ep
            if not args.no_ckpt:
                atomic_torch_save(payload, OUT_DIR / "seg_best.pt")
            safe_print(f"  saved BEST empty_fp={best_fp:.3f} nonempty={val['mean_dice_nonempty']:.4f}")
        if not args.no_ckpt:
            atomic_torch_save(payload, OUT_DIR / "seg_last.pt")
        if (not args.no_ckpt) and best_ep > 0 and (ep - best_ep) >= patience:
            safe_print(f"early stop @{best_ep} empty_fp={best_fp:.3f}")
            break

    hist_path = OUT_DIR / f"history_{tag}.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.skip_final_eval:
        safe_print("skip final eval; wrote", hist_path)
        return

    if (OUT_DIR / "seg_best.pt").exists():
        st = safe_load(OUT_DIR / "seg_best.pt", map_location="cpu")
        vae.load_state_dict(st["model"])
        best_ep = int(st.get("best_ep", best_ep))
    else:
        safe_print("no ckpt kept nonempty floor; evaluating last")
    vae.to(DEVICE)
    val_b = eval_split(vae, data.val_loader)
    test_b = eval_split(vae, data.test_loader)
    report = {
        "mode": "s5_emptyfp_ft",
        "s5_ckpt": str(S5_BEST),
        "s5_ep": s5_ep,
        "best_ep": best_ep,
        "lambda": lam,
        "lr": FT_LR,
        "batch": S5_BATCH,
        "init_val": base_val,
        "nonempty_floor": floor,
        "val": val_b,
        "test": test_b,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_print("TEST s5-emptyfp:", json.dumps(test_b, indent=2))


if __name__ == "__main__":
    main()
