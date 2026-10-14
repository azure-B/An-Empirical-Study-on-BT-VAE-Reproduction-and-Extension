"""Dual-role VAE: S5 skip line (recon) + plain no-skip line (classification + head).

Does NOT modify original/, none/, or s5_concat/ checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead
from train_original_bp import OrigVAE


class DualRoleVAE(nn.Module):
    """
    Line A — skip_net (S5): encoder + concat-skip decoder → SEG mask (recon).
    Line B — cls_net (none/plain): encoder + no-skip decoder; head(μ) → lesion yes/no.

    Inference: if sigmoid(head(μ_cls)) < tau → blank, else return S5 logits.
    """

    def __init__(self, base: int = 32, dropout_rate: float = 0.1, head_dropout: float = 0.5):
        super().__init__()
        self.base = base
        self.skip_net = SkipVAE(base, dropout_rate)
        self.cls_net = OrigVAE(base, dropout_rate)
        self.head = LesionPresenceHead(32 * base, hidden_dim=128, dropout=head_dropout)
        self.gate_thr: float | None = None

    def freeze_skip(self) -> None:
        self.skip_net.eval()
        for p in self.skip_net.parameters():
            p.requires_grad = False

    def set_cls_trainable(self, encoder: bool, decoder: bool, head: bool = True) -> None:
        for p in self.cls_net.encoder.parameters():
            p.requires_grad = encoder
        for p in self.cls_net.decoder.parameters():
            p.requires_grad = decoder
        for p in self.head.parameters():
            p.requires_grad = head

    @torch.no_grad()
    def skip_logits_det(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic S5: decode μ with skips."""
        mu, _logvar, _z, skips = self.skip_net.encoder.forward_features(x)
        return self.skip_net.decoder(mu, skips)

    def cls_forward_det(self, x: torch.Tensor):
        """Deterministic plain line: decoder(μ), head(μ)."""
        mu, logvar, _z = self.cls_net.encoder(x)
        plain_logits = self.cls_net.decoder(mu)
        presence_logit = self.head(mu)
        return plain_logits, presence_logit, mu, logvar

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        """
        Default infer-style forward returns gated S5 logits (+ mu/logvar from cls).
        """
        skip_logits = self.skip_logits_det(x)
        plain_logits, presence_logit, mu, logvar = self.cls_forward_det(x)
        if self.gate_thr is not None:
            keep = torch.sigmoid(presence_logit) >= self.gate_thr
            skip_logits = skip_logits.masked_fill(~keep.view(-1, 1, 1, 1), -20.0)
        if return_parts:
            return skip_logits, plain_logits, presence_logit, mu, logvar
        return skip_logits, mu, logvar
