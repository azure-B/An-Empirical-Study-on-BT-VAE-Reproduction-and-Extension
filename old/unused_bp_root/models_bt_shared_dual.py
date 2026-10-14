"""BT encoder shared: μ → S5 only; classification from encoder skips (not μ).

  x → BT Encoder (frozen)
        ├─ skips (e0..e4) → ClsHead(pool skips) → P → τ
        └─ μ + skips → Decoder_skip (S5) → SEG
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from train_original_bp import load_bt_into_encoder


class SkipPresenceHead(nn.Module):
    """Presence logit from multi-scale encoder skips (not μ)."""

    def __init__(self, base: int = 32, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        # e0: base, e1: 2b, e2: 4b, e3: 8b, e4: 16b  (from models_bp_skip ResBlockDown)
        chs = [base, 2 * base, 4 * base, 8 * base, 16 * base]
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        in_dim = 2 * sum(chs)  # avg+max per scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, skips: tuple) -> torch.Tensor:
        parts = []
        for e in skips:
            parts.append(self.avg(e).flatten(1))
            parts.append(self.max(e).flatten(1))
        x = torch.cat(parts, dim=1)
        return self.net(x).squeeze(1)


class BTSharedDualVAE(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, head_dropout: float = 0.5):
        super().__init__()
        self.base = base
        tmpl = SkipVAE(base, dropout_rate)
        self.encoder = tmpl.encoder
        self.decoder_skip = tmpl.decoder
        self.head = SkipPresenceHead(base, hidden_dim=128, dropout=head_dropout)
        self.gate_thr: float | None = None
        del tmpl

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def set_trainable(self, skip_dec: bool, head: bool) -> None:
        for p in self.decoder_skip.parameters():
            p.requires_grad = skip_dec
        for p in self.head.parameters():
            p.requires_grad = head

    def encode_det(self, x: torch.Tensor):
        mu, logvar, _z, skips = self.encoder.forward_features(x)
        return mu, logvar, skips

    def forward_parts(self, x: torch.Tensor):
        mu, logvar, skips = self.encode_det(x)
        # μ (+ skips) → S5 only
        skip_logits = self.decoder_skip(mu, skips)
        # classification from skips only (not μ)
        presence_logit = self.head(skips)
        return skip_logits, presence_logit, mu, logvar, skips

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        skip_logits, presence_logit, mu, logvar, skips = self.forward_parts(x)
        if self.gate_thr is not None:
            keep = torch.sigmoid(presence_logit) >= self.gate_thr
            skip_logits = skip_logits.masked_fill(~keep.view(-1, 1, 1, 1), -20.0)
        if return_parts:
            return skip_logits, presence_logit, mu, logvar, skips
        return skip_logits, mu, logvar


def load_bt_shared(model: BTSharedDualVAE, bt_ckpt, s5_ckpt: dict | None) -> None:
    load_bt_into_encoder(model.encoder, bt_ckpt)
    if s5_ckpt is not None:
        s5 = SkipVAE(model.base)
        s5.load_state_dict(s5_ckpt["model"])
        model.decoder_skip.load_state_dict(s5.decoder.state_dict())
        del s5
