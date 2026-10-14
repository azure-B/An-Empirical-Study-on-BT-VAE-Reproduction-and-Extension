"""COVID: BT pretrain → none / S5 SEG. Writes ONLY under results/covid_pipeline/.

Does not touch results/bp_original or checkpoints_bp_original.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import covid_config as C
from data_load_BT import datasets
from models_bp_skip import VAE as SkipVAE
from train_original_bp import (
    BASE,
    BT_EPOCHS,
    BT_LR,
    BT_BATCH,
    DEVICE,
    NUM_WORKERS,
    SEG_EPOCHS,
    SEG_LR,
    SEG_BATCH,
    SEG_ES_PATIENCE,
    OrigEncoder,
    OrigVAE,
    atomic_torch_save,
    batch_dice,
    batch_pred_gt_sums,
    bt_batch_loss,
    freeze_bn,
    load_bt_into_encoder,
    safe_load,
    safe_print,
    seg_step_loss,
    split_dice_stats,
)


def get_data(batch_size: int):
    return datasets.covid(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=False)


def run_bt():
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = get_data(BT_BATCH)
    safe_print(
        f"[COVID BT] n={len(data.train)}/{len(data.val)}/{len(data.test)} "
        f"OUT={C.CKPT_DIR}"
    )
    encoder = OrigEncoder(base=BASE).to(DEVICE)
    conv1x1 = nn.Conv2d(2, 1, 1).to(DEVICE)
    freeze_bn(encoder)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(conv1x1.parameters()),
        lr=BT_LR, weight_decay=1e-4,
    )
    start_ep, best_val = 1, float("inf")
    if C.BT_LAST.exists():
        st = safe_load(C.BT_LAST)
        encoder.load_state_dict(st["encoder"])
        if "conv1x1" in st:
            conv1x1.load_state_dict(st["conv1x1"])
        if "optimizer" in st:
            opt.load_state_dict(st["optimizer"])
        start_ep = int(st.get("ep", 0)) + 1
        best_val = float(st.get("best_val", best_val))
        safe_print(f"BT resume ep={start_ep} best={best_val:.6f}")

    for ep in range(start_ep, BT_EPOCHS + 1):
        encoder.train()
        freeze_bn(encoder)
        running, t0 = 0.0, time.time()
        for batch in data.train_loader:
            view1 = batch[2].to(DEVICE, non_blocking=True)
            view2 = batch[3].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = bt_batch_loss(encoder, view1, view2, ep)
            if not torch.isfinite(loss):
                raise ValueError("BT loss NaN/Inf")
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
        train_avg = running / max(1, len(data.train_loader))
        encoder.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in data.val_loader:
                view1 = batch[2].to(DEVICE, non_blocking=True)
                view2 = batch[3].to(DEVICE, non_blocking=True)
                val_running += float(bt_batch_loss(encoder, view1, view2, ep).item())
        val_avg = val_running / max(1, len(data.val_loader))
        safe_print(
            f"[COVID BT] {ep}/{BT_EPOCHS} train={train_avg:.6f} val={val_avg:.6f} "
            f"({time.time()-t0:.1f}s)"
        )
        payload = {
            "ep": ep, "encoder": encoder.state_dict(), "conv1x1": conv1x1.state_dict(),
            "optimizer": opt.state_dict(), "best_val": best_val, "dataset": C.DATASET_NAME,
        }
        if val_avg < best_val:
            best_val = val_avg
            payload["best_val"] = best_val
            atomic_torch_save(payload, C.BT_BEST)
            safe_print(f"  saved BT_BEST val={best_val:.6f}")
        atomic_torch_save(payload, C.BT_LAST)
    return C.BT_BEST if C.BT_BEST.exists() else C.BT_LAST


def run_seg(mode: str, bt_ckpt: Path, max_ep: int | None = None):
    assert mode in {"none", "s5"}
    out_dir = C.RESULT_DIR / ({"none": "none", "s5": "s5_concat"}[mode])
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_best, seg_last = out_dir / "seg_best.pt", out_dir / "seg_last.pt"
    seg_bs = 8 if mode == "s5" else min(SEG_BATCH, 16)
    data = get_data(seg_bs)
    safe_print(
        f"[COVID SEG {mode}] n={len(data.train)}/{len(data.val)}/{len(data.test)} "
        f"batch={seg_bs} OUT={out_dir}"
    )
    vae = (OrigVAE if mode == "none" else SkipVAE)(base=BASE).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=SEG_LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.95, patience=3, min_lr=1e-8)
    start_ep, best_loss, best_ep = 1, float("inf"), 0
    if seg_last.exists():
        st = safe_load(seg_last, map_location="cpu")
        vae.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        start_ep = int(st.get("ep", 0)) + 1
        best_loss = float(st.get("best_loss", best_loss))
        best_ep = int(st.get("best_ep", 0))
        del st
        safe_print(f"SEG resume {mode} ep={start_ep} best_loss={best_loss:.4f}")
    else:
        load_bt_into_encoder(vae.encoder, bt_ckpt)

    history = []
    limit = SEG_EPOCHS if max_ep is None else min(int(max_ep), SEG_EPOCHS)
    bad = 0
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    for ep in range(start_ep, limit + 1):
        vae.train()
        tr_loss = tr_dice = 0.0
        t0 = time.time()
        for batch in data.train_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss, dice_v, _kl = seg_step_loss(vae, flair, seg)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            tr_dice += dice_v
        ntr = max(1, len(data.train_loader))
        vae.eval()
        va_loss, va_dices, va_gt, va_pred = 0.0, [], [], []
        with torch.no_grad():
            for batch in data.val_loader:
                flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss, _, _, out, seg_b = seg_step_loss(vae, flair, seg, return_out=True)
                va_loss += float(loss.item())
                va_dices.append(batch_dice(out, seg_b))
                ps, gs = batch_pred_gt_sums(out, seg_b)
                va_pred.append(ps)
                va_gt.append(gs)
        nva = max(1, len(data.val_loader))
        va_loss /= nva
        st = split_dice_stats(
            np.concatenate(va_dices), np.concatenate(va_gt), np.concatenate(va_pred)
        )
        sched.step(va_loss)
        rec = {
            "ep": ep,
            "train_loss": tr_loss / ntr,
            "train_dice": tr_dice / ntr,
            "val_loss": va_loss,
            "val_dice": st["dice_mean"],
            "val_ne": st["mean_dice_nonempty"],
            "val_empty_fp": st["empty_fp_rate"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        safe_print(
            f"[COVID SEG {mode} {ep}/{limit}] loss={rec['train_loss']:.4f}/"
            f"{va_loss:.4f} ne={st['mean_dice_nonempty']:.4f} "
            f"fp={st['empty_fp_rate']:.3f} ({rec['sec']:.1f}s)"
        )
        payload = {
            "model": vae.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "ep": ep,
            "best_loss": best_loss, "best_ep": best_ep,
            "dataset": C.DATASET_NAME, "mode": mode,
        }
        atomic_torch_save(payload, seg_last)
        if va_loss < best_loss - 1e-5:
            best_loss, best_ep, bad = va_loss, ep, 0
            payload["best_loss"] = best_loss
            payload["best_ep"] = best_ep
            atomic_torch_save(payload, seg_best)
            safe_print(f"  * best val_loss={best_loss:.4f} ne={st['mean_dice_nonempty']:.4f}")
        else:
            bad += 1
            if bad >= SEG_ES_PATIENCE:
                safe_print(f"early stop SEG {mode} @ ep {ep}")
                break

    # TEST eval
    st_best = safe_load(seg_best if seg_best.exists() else seg_last, map_location="cpu")
    vae.load_state_dict(st_best["model"])
    vae.eval()
    te_dices, te_gt, te_pred = [], [], []
    with torch.no_grad():
        for batch in data.test_loader:
            flair, seg = batch[0].to(DEVICE), batch[1].to(DEVICE)
            _, _, _, out, seg_b = seg_step_loss(vae, flair, seg, return_out=True)
            te_dices.append(batch_dice(out, seg_b))
            ps, gs = batch_pred_gt_sums(out, seg_b)
            te_pred.append(ps)
            te_gt.append(gs)
    test_st = split_dice_stats(
        np.concatenate(te_dices), np.concatenate(te_gt), np.concatenate(te_pred)
    )
    report = {
        "dataset": C.DATASET_NAME,
        "mode": mode,
        "best_ep": best_ep,
        "best_val_loss": best_loss,
        "test": {
            "dice": test_st["dice_mean"],
            "nonempty_dice": test_st["mean_dice_nonempty"],
            "empty_fp_rate": test_st["empty_fp_rate"],
        },
        "history_tail": history[-5:],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "COMPARE.md").write_text(
        f"# COVID SEG `{mode}`\n\n"
        f"dataset=`{C.DATASET_NAME}` · separate from BP\n\n"
        f"TEST ne={test_st['mean_dice_nonempty']:.4f} "
        f"empty_fp={test_st['empty_fp_rate']:.3f} "
        f"dice={test_st['dice_mean']:.4f}\n",
        encoding="utf-8",
    )
    safe_print(
        f"COVID {mode} TEST ne={test_st['mean_dice_nonempty']:.4f} "
        f"fp={test_st['empty_fp_rate']:.3f}"
    )
    return seg_best


def write_readme():
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULT_DIR / "README.md").write_text(
        "\n".join([
            "# COVID pipeline (isolated from BP)",
            "",
            "| | BP | COVID |",
            "|---|---|---|",
            "| results | `results/bp_original/` | `results/covid_pipeline/` |",
            "| BT ckpt | `checkpoints_bp_original/` | `checkpoints_covid/` |",
            "| data | `bp_train_30` | `covid` (`covid_8_1_1`) |",
            "",
            "Do not mix checkpoints across domains.",
            "",
            "## Train",
            "```bash",
            "python train_covid_seg.py --phase all",
            "python train_covid_gate.py   # after S5 exists",
            "```",
            "",
        ]),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["all", "bt", "none", "s5", "readme"],
        default="all",
    )
    ap.add_argument("--max-seg-epoch", type=int, default=None)
    args = ap.parse_args()
    write_readme()
    safe_print("DEVICE", DEVICE, "COVID RESULT_DIR", C.RESULT_DIR)

    if args.phase == "readme":
        return

    if args.phase in {"all", "bt"}:
        bt_ckpt = run_bt()
    else:
        bt_ckpt = C.BT_BEST if C.BT_BEST.exists() else C.BT_LAST
        if not bt_ckpt.exists():
            raise FileNotFoundError(bt_ckpt)

    if args.phase in {"all", "none"}:
        run_seg("none", bt_ckpt, max_ep=args.max_seg_epoch)
    if args.phase in {"all", "s5"}:
        run_seg("s5", bt_ckpt, max_ep=args.max_seg_epoch)

    # root COMPARE stub
    lines = ["# COVID pipeline SEG", "", "Isolated from `results/bp_original`.", ""]
    for name in ("none", "s5_concat"):
        rp = C.RESULT_DIR / name / "report.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            t = r.get("test", {})
            lines.append(
                f"- **{name}**: ne={t.get('nonempty_dice', float('nan')):.4f} "
                f"fp={t.get('empty_fp_rate', float('nan')):.3f}"
            )
    (C.RESULT_DIR / "COMPARE_SEG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print("wrote", C.RESULT_DIR / "COMPARE_SEG.md")


if __name__ == "__main__":
    main()
