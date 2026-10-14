"""Skip head (S5) + plain head (none). Merge is a presence gate, not a union.

Shared early decoder was tried first: none's up3–5 on S5 up2 features
collapses (VAL nonempty ~0). So each head keeps its own pretrained net.
original/ is not modified.
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from train_original_bp import OrigVAE


def merge_skip_gated(skip_logits: torch.Tensor, plain_logits: torch.Tensor) -> torch.Tensor:
    """Skip spatial mask, kept only where the no-skip head fires."""
    skip_p = torch.sigmoid(skip_logits)
    gate = torch.sigmoid(plain_logits.amax(dim=(2, 3), keepdim=True))
    merged_p = (skip_p * gate).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(merged_p)


class DualVAE(nn.Module):
    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        self.skip_net = SkipVAE(base, dropout_rate)
        self.plain_net = OrigVAE(base, dropout_rate)

    def forward(self, x, return_heads=False):
        skip_logits, mu, logvar = self.skip_net(x)
        plain_logits, _, _ = self.plain_net(x)
        merged = merge_skip_gated(skip_logits, plain_logits)
        if return_heads:
            return merged, skip_logits, plain_logits, mu, logvar
        return merged, mu, logvar
