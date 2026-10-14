"""Paths for the four paper models. Weights stay under checkpoints/."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints" / "covid"
RESULT_DIR = ROOT / "results" / "covid"
BT_BEST = CKPT_DIR / "bt_encoder_best.pt"
BT_LAST = CKPT_DIR / "bt_encoder_last.pt"
NONE_BEST = RESULT_DIR / "none" / "seg_best.pt"
S5_BEST = RESULT_DIR / "s5_concat" / "seg_best.pt"
DATASET_NAME = "covid"  # data_load_BT REGISTRY key → covid_8_1_1
