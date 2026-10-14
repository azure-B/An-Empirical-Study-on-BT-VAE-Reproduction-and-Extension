"""Cascade with upgraded cls: Head(μ)+Head(skips), τ calibrated separately (not in BCE).

  x → BT Encoder (frozen) → μ, skips
        ├─ Head_μ(μ) + Head_skip(skips) → presence_logit   # train this
        └─ μ+skips → Decoder_seg (frozen) → mask

Infer: keep SEG iff sigmoid(presence) ≥ τ  (τ picked for high recall)
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead
from models_bt_cascade import BTCascadeVAE
from train_original_bp import OrigVAE, load_bt_into_encoder


class SkipPresenceHead(nn.Module):
    """Presence from multi-scale skips (avg+max pool)."""

    def __init__(self, base: int = 32, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        chs = [base, 2 * base, 4 * base, 8 * base, 16 * base]
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        in_dim = 2 * sum(chs)
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
        return self.net(torch.cat(parts, dim=1)).squeeze(1)


class BTCascadeClsUpgrade(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, head_dropout: float = 0.4):
        super().__init__()
        self.base = base
        skip = SkipVAE(base, dropout_rate)
        plain = OrigVAE(base, dropout_rate)
        self.encoder = skip.encoder
        self.decoder_seg = skip.decoder
        self.decoder_cls = plain.decoder  # optional aux; not required for gate
        self.head_mu = LesionPresenceHead(32 * base, hidden_dim=128, dropout=head_dropout)
        self.head_skip = SkipPresenceHead(base, hidden_dim=128, dropout=head_dropout)
        # fuse: presence = logit_mu + softplus(w) * logit_skip
        self.fuse_w = nn.Parameter(torch.tensor(0.0))
        # τ stored as buffer after calibration (prob space), not trained with BCE
        self.register_buffer("tau_prob", torch.tensor(0.5))
        del skip, plain

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_seg(self) -> None:
        self.decoder_seg.eval()
        for p in self.decoder_seg.parameters():
            p.requires_grad = False

    def set_trainable(self, cls_heads: bool, decoder_cls: bool = False) -> None:
        for p in self.head_mu.parameters():
            p.requires_grad = cls_heads
        for p in self.head_skip.parameters():
            p.requires_grad = cls_heads
        self.fuse_w.requires_grad = cls_heads
        for p in self.decoder_cls.parameters():
            p.requires_grad = decoder_cls

    def encode_det(self, x: torch.Tensor):
        mu, logvar, _z, skips = self.encoder.forward_features(x)
        return mu, logvar, skips

    def presence_logit(self, mu: torch.Tensor, skips: tuple) -> torch.Tensor:
        lm = self.head_mu(mu)
        ls = self.head_skip(skips)
        w = torch.nn.functional.softplus(self.fuse_w)
        return lm + w * ls

    def forward_parts(self, x: torch.Tensor):
        mu, logvar, skips = self.encode_det(x)
        with torch.no_grad():
            seg_logits = self.decoder_seg(mu, skips)
        presence = self.presence_logit(mu, skips)
        return seg_logits, presence, mu, logvar, skips

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        seg_logits, presence, mu, logvar, skips = self.forward_parts(x)
        keep = torch.sigmoid(presence) >= self.tau_prob
        seg_logits = seg_logits.masked_fill(~keep.view(-1, 1, 1, 1), -20.0)
        if return_parts:
            return seg_logits, presence, mu, logvar, skips
        return seg_logits, mu, logvar


def load_from_cascade_focal(model: BTCascadeClsUpgrade, ckpt: dict) -> int:
    """Load encoder + decoder_seg (+ decoder_cls if shapes match) from cascade_focal."""
    src = ckpt["model"] if "model" in ckpt else ckpt
    dst = model.state_dict()
    # map old keys
    mapped = {}
    for k, v in src.items():
        if k.startswith("encoder.") or k.startswith("decoder_seg.") or k.startswith("decoder_cls."):
            if k in dst and dst[k].shape == v.shape:
                mapped[k] = v
    model.load_state_dict(mapped, strict=False)
    return len(mapped)


def warm_from_bt_none(model: BTCascadeClsUpgrade, bt_ckpt, none_ckpt: dict | None) -> None:
    load_bt_into_encoder(model.encoder, bt_ckpt)
    if none_ckpt is not None:
        plain = OrigVAE(model.base)
        plain.load_state_dict(none_ckpt["model"])
        model.decoder_cls.load_state_dict(plain.decoder.state_dict())
        del plain
