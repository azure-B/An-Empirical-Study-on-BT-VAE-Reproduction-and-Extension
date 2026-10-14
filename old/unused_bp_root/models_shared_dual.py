"""Shared-encoder dual decoder: 1 encoder → skip SEG + cls plain + presence head.

Intended structure (NOT two full VAEs):
  x → Encoder (shared, S5)
        → (μ, skips)
             ├─ Decoder_skip(μ, skips) → SEG mask   [freeze from s5_concat]
             └─ Decoder_cls(μ) + Head(μ) → P(present) → τ gate

original/ untouched. Does not overwrite none/ or s5_concat/.
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead
from train_original_bp import OrigVAE


class SharedDualVAE(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, head_dropout: float = 0.5):
        super().__init__()
        self.base = base
        # build from templates then we will replace modules via load
        skip_tmpl = SkipVAE(base, dropout_rate)
        plain_tmpl = OrigVAE(base, dropout_rate)
        self.encoder = skip_tmpl.encoder
        self.decoder_skip = skip_tmpl.decoder
        self.decoder_cls = plain_tmpl.decoder
        self.head = LesionPresenceHead(32 * base, hidden_dim=128, dropout=head_dropout)
        self.gate_thr: float | None = None
        # drop unused plain encoder from tmpl (not registered)
        del skip_tmpl, plain_tmpl

    def freeze_recon(self) -> None:
        """Freeze shared encoder + skip decoder (keep S5 SEG)."""
        self.encoder.eval()
        self.decoder_skip.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder_skip.parameters():
            p.requires_grad = False

    def set_cls_trainable(self, decoder: bool, head: bool = True) -> None:
        for p in self.decoder_cls.parameters():
            p.requires_grad = decoder
        for p in self.head.parameters():
            p.requires_grad = head

    def encode_det(self, x: torch.Tensor):
        """μ, logvar, skips — no sampling for SEG/cls."""
        mu, logvar, _z, skips = self.encoder.forward_features(x)
        return mu, logvar, skips

    def forward_parts(self, x: torch.Tensor):
        mu, logvar, skips = self.encode_det(x)
        skip_logits = self.decoder_skip(mu, skips)
        cls_logits = self.decoder_cls(mu)
        presence_logit = self.head(mu)
        return skip_logits, cls_logits, presence_logit, mu, logvar

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        skip_logits, cls_logits, presence_logit, mu, logvar = self.forward_parts(x)
        if self.gate_thr is not None:
            keep = torch.sigmoid(presence_logit) >= self.gate_thr
            skip_logits = skip_logits.masked_fill(~keep.view(-1, 1, 1, 1), -20.0)
        if return_parts:
            return skip_logits, cls_logits, presence_logit, mu, logvar
        return skip_logits, mu, logvar


def load_shared_from_ckpts(
    model: SharedDualVAE,
    s5_ckpt: dict,
    none_ckpt: dict,
) -> None:
    """encoder+decoder_skip ← S5; decoder_cls ← none decoder."""
    s5 = SkipVAE(model.base)
    s5.load_state_dict(s5_ckpt["model"])
    model.encoder.load_state_dict(s5.encoder.state_dict())
    model.decoder_skip.load_state_dict(s5.decoder.state_dict())
    del s5

    none = OrigVAE(model.base)
    none.load_state_dict(none_ckpt["model"])
    model.decoder_cls.load_state_dict(none.decoder.state_dict())
    del none
