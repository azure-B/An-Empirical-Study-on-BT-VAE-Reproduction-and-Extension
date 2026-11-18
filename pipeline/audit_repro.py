"""Reproduction-audit helpers for BT-VAE vs current pipeline.

Separates original notebook behavior from current defaults without changing
model architecture, BT/seg losses, preprocessing, or base eval formulas.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

CKPT_FORMAT_VERSION = 2
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class RunConfig:
    """Explicit run settings (prefer passing this over scattering globals)."""

    dataset: str = "bp"
    run_id: Optional[str] = None
    seed: int = 0
    eval_seed: Optional[int] = None
    seg_scheduler_steps_per_epoch: int = 1
    bt_bn_mode: str = "frozen_stats"  # frozen_stats | original
    weights_only_restart: bool = False
    # mirrored train knobs used in fingerprints (not eval-only)
    bt_batch: int = 16
    bt_lr: float = 3e-4
    seg_lr: float = 1e-4
    seg_batch: int = 16
    recon_weight: float = 0.99
    beta_local: float = 0.01
    early_stop: bool = True  # segmentation ES
    bt_early_stop: bool = False  # original notebook: BT runs full epochs
    es_patience: int = 30
    paper_ratio: float = 0.3

    def __post_init__(self) -> None:
        if self.eval_seed is None:
            self.eval_seed = int(self.seed)
        if self.seg_scheduler_steps_per_epoch not in (1, 2):
            raise ValueError("seg_scheduler_steps_per_epoch must be 1 or 2")
        if self.bt_bn_mode not in ("frozen_stats", "original"):
            raise ValueError("bt_bn_mode must be frozen_stats|original")
        if self.run_id is not None:
            if not SAFE_RUN_ID.match(self.run_id):
                raise ValueError(
                    f"run-id must match {SAFE_RUN_ID.pattern!r}, got {self.run_id!r}"
                )

    def resolved_eval_seed(self) -> int:
        return int(self.seed if self.eval_seed is None else self.eval_seed)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval_seed"] = self.resolved_eval_seed()
        return d


@dataclass
class PathLayout:
    root: Path
    dataset: str
    run_id: Optional[str]
    ckpt_dir: Path
    result_dir: Path
    out_dir: Path
    audit_dir: Path
    bt_best: Path
    bt_last: Path
    vram_log: Path
    lr_log: Path
    manifest_path: Path

    @classmethod
    def from_config(cls, root: Path, cfg: RunConfig) -> "PathLayout":
        tag = "bp" if cfg.dataset == "bp" else "covid"
        if cfg.run_id:
            ckpt_dir = root / "checkpoints" / f"{tag}_paper" / "runs" / cfg.run_id
            base = root / "results" / tag / "runs" / cfg.run_id
            result_dir = base / "paper_30"
            out_dir = base / "compare_paper"
            audit_dir = base / "audit"
        else:
            ckpt_dir = root / "checkpoints" / f"{tag}_paper"
            result_dir = root / "results" / tag / "paper_30"
            out_dir = root / "results" / tag / "compare_paper"
            audit_dir = out_dir / "audit"
        for d in (ckpt_dir, result_dir, out_dir, audit_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            dataset=cfg.dataset,
            run_id=cfg.run_id,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
            out_dir=out_dir,
            audit_dir=audit_dir,
            bt_best=ckpt_dir / "bt_encoder_best.pt",
            bt_last=ckpt_dir / "bt_encoder_last.pt",
            vram_log=out_dir / "vram.log",
            lr_log=audit_dir / "seg_lr_schedule.jsonl",
            manifest_path=audit_dir / "data_manifest.json",
        )


def git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def membership_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def order_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_ids_from_dataset(ds) -> list[str]:
    """Stable sample identifiers without inventing patient IDs."""
    if hasattr(ds, "filenames"):
        return [str(x) for x in ds.filenames]
    if hasattr(ds, "indices") and hasattr(ds, "base"):
        base = ds.base
        names = list(getattr(base, "filenames", []))
        return [names[i] if i < len(names) else str(i) for i in ds.indices]
    return [str(i) for i in range(len(ds))]


def build_split_manifest(name: str, ds, note: str = "") -> dict[str, Any]:
    ids = sample_ids_from_dataset(ds)
    return {
        "split": name,
        "n": len(ids),
        "ids": ids,
        "membership_hash": membership_hash(ids),
        "order_hash": order_hash(ids),
        "note": note,
    }


def write_data_manifest(
    path: Path,
    *,
    dataset: str,
    full_train,
    labeled_train,
    val,
    test,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """Write current split membership. Original-equality is unverified without source lists."""
    payload = {
        "dataset": dataset,
        "original_split_identity": "unverified",
        "message": "현재 분할 manifest 생성 완료 / 원본과의 동일성 미확인",
        "patient_id_leak_check": "skipped_no_authoritative_patient_ids",
        "full_train": build_split_manifest("full_train", full_train),
        "labeled_30_train": build_split_manifest("labeled_30_train", labeled_train),
        "val": build_split_manifest("val", val),
        "test": build_split_manifest("test", test),
    }
    if extra:
        payload["extra"] = extra
    # overlap on filename stems only if names look shared
    stems = {
        k: {Path(x).stem for x in payload[k]["ids"]}
        for k in ("full_train", "labeled_30_train", "val", "test")
    }
    payload["filename_stem_overlap"] = {
        "labeled_vs_val": sorted(stems["labeled_30_train"] & stems["val"]),
        "labeled_vs_test": sorted(stems["labeled_30_train"] & stems["test"]),
        "val_vs_test": sorted(stems["val"] & stems["test"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_fingerprint(
    cfg: RunConfig,
    *,
    model_name: str,
    split_hash: str,
    bt_ckpt_sha: Optional[str] = None,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """Fingerprint for training reuse — excludes eval-only knobs like eval_seed."""
    return {
        "kind": "train",
        "dataset": cfg.dataset,
        "model": model_name,
        "split_membership_hash": split_hash,
        "seg_scheduler_steps_per_epoch": cfg.seg_scheduler_steps_per_epoch,
        "bt_bn_mode": cfg.bt_bn_mode if model_name != "VAE" else "n/a",
        "bt_batch": cfg.bt_batch,
        "bt_lr": cfg.bt_lr,
        "seg_lr": cfg.seg_lr,
        "seg_batch": cfg.seg_batch,
        "recon_weight": cfg.recon_weight,
        "beta_local": cfg.beta_local,
        "early_stop": cfg.early_stop,
        "bt_early_stop": cfg.bt_early_stop,
        "es_patience": cfg.es_patience,
        "paper_ratio": cfg.paper_ratio,
        "seed": cfg.seed,
        "bt_ckpt_sha256": bt_ckpt_sha or "none",
        "git_commit": code_commit,
        "run_id": cfg.run_id,
    }


def eval_fingerprint(cfg: RunConfig, *, eval_batch: int, code_commit: str = "unknown") -> dict[str, Any]:
    return {
        "kind": "eval",
        "eval_seed": cfg.resolved_eval_seed(),
        "eval_batch": eval_batch,
        "protocol": "test.ipynb sampled-z batch-flat mean",
        "git_commit": code_commit,
        "run_id": cfg.run_id,
    }


def fingerprints_match(a: dict, b: dict, keys: Optional[Iterable[str]] = None) -> bool:
    if not a or not b:
        return False
    keys = list(keys) if keys is not None else sorted(set(a) | set(b))
    for k in keys:
        if a.get(k) != b.get(k):
            return False
    return True


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scheduler_group_lrs(sched) -> list[float]:
    return [float(g["lr"]) for g in sched.optimizer.param_groups]


def snapshot_scheduler(sched) -> dict[str, Any]:
    return {
        "lrs": scheduler_group_lrs(sched),
        "num_bad_epochs": int(getattr(sched, "num_bad_epochs", -1)),
        "best": float(sched.best) if getattr(sched, "best", None) is not None else None,
    }


def step_scheduler_n(
    sched,
    val_loss: float,
    n_steps: int,
) -> list[dict[str, Any]]:
    """Call ReduceLROnPlateau.step(val_loss) exactly n_steps times (no re-eval)."""
    traces = []
    for i in range(int(n_steps)):
        sched.step(val_loss)
        snap = snapshot_scheduler(sched)
        snap["call_index"] = i + 1
        traces.append(snap)
    return traces


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def freeze_bn_affine_only(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad = False


def apply_bt_bn_mode(encoder: nn.Module, mode: str, *, after_train_call: bool) -> None:
    """
    frozen_stats: after encoder.train(), force BN.eval + freeze affine (current).
    original: freeze affine once; after encoder.train() do NOT re-eval BN.
    """
    freeze_bn_affine_only(encoder)
    if mode == "frozen_stats" and after_train_call:
        for m in encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    # original: leave BN in whatever mode train()/eval() set


@torch.no_grad()
def bn_oneshot_probe(
    encoder_factory,
    batch: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    """Compare BN stats update under frozen_stats vs original. No optimizer step.

    Does not claim performance impact — BN behavior only.
    """
    results: dict[str, Any] = {"note": "BN behavior probe only; not a performance claim"}

    def _capture(enc: nn.Module) -> dict[str, dict]:
        out = {}
        for name, m in enc.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                out[name] = {
                    "training": bool(m.training),
                    "weight_requires_grad": bool(m.weight.requires_grad) if m.weight is not None else None,
                    "bias_requires_grad": bool(m.bias.requires_grad) if m.bias is not None else None,
                    "running_mean": m.running_mean.detach().cpu().clone(),
                    "running_var": m.running_var.detach().cpu().clone(),
                    "num_batches_tracked": int(m.num_batches_tracked.item())
                    if m.num_batches_tracked is not None
                    else None,
                }
        return out

    def _delta(before, after) -> list[dict]:
        rows = []
        for name in before:
            b, a = before[name], after[name]
            rows.append(
                {
                    "name": name,
                    "training": a["training"],
                    "weight_requires_grad": a["weight_requires_grad"],
                    "bias_requires_grad": a["bias_requires_grad"],
                    "running_mean_l1": float((a["running_mean"] - b["running_mean"]).abs().sum()),
                    "running_var_l1": float((a["running_var"] - b["running_var"]).abs().sum()),
                    "num_batches_tracked_delta": (
                        None
                        if a["num_batches_tracked"] is None or b["num_batches_tracked"] is None
                        else a["num_batches_tracked"] - b["num_batches_tracked"]
                    ),
                }
            )
        return rows

    x = batch.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(1)

    template = encoder_factory().to(device)
    init_sd = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    del template

    for mode in ("frozen_stats", "original"):
        enc = encoder_factory().to(device)
        enc.load_state_dict(init_sd)
        freeze_bn_affine_only(enc)
        enc.train()
        apply_bt_bn_mode(enc, mode, after_train_call=True)
        before = _capture(enc)
        _ = enc(x)
        after = _capture(enc)
        rows = _delta(before, after)
        results[mode] = {
            "layers": rows,
            "any_stats_changed": any(
                r["running_mean_l1"] > 0
                or r["running_var_l1"] > 0
                or (r["num_batches_tracked_delta"] or 0) > 0
                for r in rows
            ),
            "all_affine_frozen": all(
                (r["weight_requires_grad"] is False) and (r["bias_requires_grad"] is False)
                for r in rows
            ),
            "any_bn_training_true": any(r["training"] for r in rows),
        }
    return results


def provenance_block(
    *,
    ckpt_path: Optional[Path],
    model_name: str,
    epoch: Any,
    cfg: RunConfig,
    bt_source: Optional[Path] = None,
    code_commit: str = "unknown",
    eval_batch: Optional[int] = None,
    warm_start: bool = False,
) -> dict[str, Any]:
    def _file(p: Optional[Path]) -> dict[str, Any]:
        if p is None or not Path(p).exists():
            return {"path": str(p) if p else "unknown", "sha256": "unknown"}
        pp = Path(p)
        return {"path": str(pp), "sha256": sha256_file(pp)}

    return {
        "checkpoint": _file(ckpt_path),
        "model": model_name,
        "epoch": epoch if epoch is not None else "unknown",
        "run_config": cfg.to_dict(),
        "bt_pretrain_source": _file(bt_source) if bt_source else {"path": "none", "sha256": "none"},
        "git_commit": code_commit,
        "eval": {
            "protocol": "test.ipynb sampled-z batch-flat mean",
            "batch_size": eval_batch if eval_batch is not None else "unknown",
            "eval_seed": cfg.resolved_eval_seed(),
        },
        "warm_start_weights_only": bool(warm_start),
    }


def load_state_strict(module: nn.Module, state: dict, *, label: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=True)
    # strict=True raises on mismatch; keep signature for callers that log
    if missing or unexpected:
        raise RuntimeError(f"{label}: unexpected strict outcome missing={missing} unexpected={unexpected}")


def require_full_resume(st: dict, *, label: str, need_scheduler: bool, need_scaler: bool) -> None:
    ver = int(st.get("ckpt_format_version", 0))
    required = ["optimizer", "rng", "ep", "best_ep", "bad", "run_config"]
    if "encoder" not in st and "model" not in st:
        required.append("model_or_encoder")
    missing = [k for k in required if k == "model_or_encoder" or k not in st]
    if "model_or_encoder" in missing:
        missing.remove("model_or_encoder")
        if "encoder" not in st and "model" not in st:
            missing.append("model|encoder")
    if need_scheduler and "scheduler" not in st:
        missing.append("scheduler")
    if need_scaler and "scaler" not in st:
        missing.append("scaler")
    if ver < CKPT_FORMAT_VERSION or missing:
        raise RuntimeError(
            f"{label}: incomplete checkpoint for full resume "
            f"(format={ver}, missing={missing}). "
            f"Use --weights-only-restart for warm-start, or start fresh under a new --run-id."
        )


def report_reuse_decision(
    report_path: Path,
    expected_train_fp: dict,
) -> tuple[bool, str]:
    """Return (may_reuse, reason). Old reports without fingerprint → do not reuse."""
    if not report_path.exists():
        return False, "no report"
    try:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"unreadable report: {e}"
    fp = rep.get("train_fingerprint")
    if not isinstance(fp, dict):
        return False, "legacy report without train_fingerprint — not treated as compatible"
    if not fingerprints_match(fp, expected_train_fp):
        return False, "train_fingerprint mismatch — will not auto-reuse"
    return True, "train_fingerprint match"
