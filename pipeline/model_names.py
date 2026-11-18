"""Canonical model names for this study.

Display names (always use these in logs / charts / reports):
  VAE
  BT-VAE
  BT-VAE+C          # C = lesion presence binary classification gate
  BT-VAE+S          # S = skip connections on BT-VAE_CODE backbone
  BT-VAE+S&C(Ours)  # S + C

Filesystem dirs under results/covid/label_ratio/rXX/ use DIR_* (no &).
Legacy aliases (none / s5 / ...) still resolve for old checkpoints.
"""
from __future__ import annotations

# Official display order for bar charts
DISPLAY = (
    "VAE",
    "BT-VAE",
    "BT-VAE+C",
    "BT-VAE+S",
    "BT-VAE+S&C(Ours)",
)

# Safe directory names (Windows-friendly; no &)
DIR = {
    "VAE": "VAE",
    "BT-VAE": "BT-VAE",
    "BT-VAE+C": "BT-VAE+C",
    "BT-VAE+S": "BT-VAE+S",
    "BT-VAE+S&C(Ours)": "BT-VAE+SC",
}

# Internal training mode → display
MODE_TO_DISPLAY = {
    "vae": "VAE",
    "bt_vae": "BT-VAE",
    "bt_vae_c": "BT-VAE+C",
    "bt_vae_s": "BT-VAE+S",
    "bt_vae_sc": "BT-VAE+S&C(Ours)",
    # legacy
    "none": "BT-VAE",
    "none_cls": "BT-VAE+C",
    "s5": "BT-VAE+S",
    "s5_cls": "BT-VAE+S&C(Ours)",
}

# Display → preferred result subdir
DISPLAY_TO_DIR = DIR

# All dirs that may hold a given display model's seg / head
LEGACY_DIRS = {
    "VAE": ("VAE",),
    "BT-VAE": ("BT-VAE", "none"),
    "BT-VAE+C": ("BT-VAE+C", "none_cls"),
    "BT-VAE+S": ("BT-VAE+S", "s5"),
    "BT-VAE+S&C(Ours)": ("BT-VAE+SC", "s5_cls"),
}


def display(name_or_mode: str) -> str:
    if name_or_mode in DISPLAY:
        return name_or_mode
    return MODE_TO_DISPLAY.get(name_or_mode, name_or_mode)


def dir_name(display_name: str) -> str:
    return DIR[display(display_name)]
