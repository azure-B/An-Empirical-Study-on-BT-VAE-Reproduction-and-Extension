"""wprod+mass compressed to a single S5 encoder.

Replaces dual cls_net encoder + Presence25D ResNet with:
  Head_μ(μ_center) × Head_ctx([μ_prev, μ_center, μ_next])
SEG still from frozen S5 decoder(μ_c, skips_c).
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from models_lesion_presence import LesionPresenceHead


class ContextPresenceHead(nn.Module):
    """Presence logit from three spatial μ maps (prev, center, next). No sigmoid."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 3 slices × (avg + max) × C
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, mu: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.avg_pool(mu).flatten(1), self.max_pool(mu).flatten(1)],
            dim=1,
        )

    def forward(
        self, mu_prev: torch.Tensor, mu_c: torch.Tensor, mu_next: torch.Tensor
    ) -> torch.Tensor:
        pooled = torch.cat(
            [self._pool(mu_prev), self._pool(mu_c), self._pool(mu_next)], dim=1
        )
        return self.classifier(pooled).squeeze(1)


class Wprod1Enc(nn.Module):
    """Frozen S5 + two presence heads (μ / 2.5D-ctx). Gate math applied outside."""

    def __init__(self, base: int = 32, dropout_rate: float = 0.1, head_dropout: float = 0.5):
        super().__init__()
        self.base = base
        self.s5 = SkipVAE(base, dropout_rate)
        ch = 32 * base
        self.head_mu = LesionPresenceHead(ch, hidden_dim=128, dropout=head_dropout)
        self.head_ctx = ContextPresenceHead(ch, hidden_dim=128, dropout=head_dropout)

    def freeze_s5(self) -> None:
        self.s5.eval()
        for p in self.s5.parameters():
            p.requires_grad = False

    def encode_mu(self, x: torch.Tensor):
        """Deterministic μ (+ skips). No grad into S5 when frozen."""
        mu, logvar, _z, skips = self.s5.encoder.forward_features(x)
        return mu, logvar, skips

    @torch.no_grad()
    def seg_logits(self, x: torch.Tensor) -> torch.Tensor:
        mu, _lv, skips = self.encode_mu(x)
        return self.s5.decoder(mu, skips)

    def presence_logits(
        self,
        x_prev: torch.Tensor,
        x_c: torch.Tensor,
        x_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logit_μ, logit_ctx). Encodes three slices with shared S5 encoder."""
        mu_p, _, _ = self.encode_mu(x_prev)
        mu_c, _, _ = self.encode_mu(x_c)
        mu_n, _, _ = self.encode_mu(x_next)
        return self.head_mu(mu_c), self.head_ctx(mu_p, mu_c, mu_n)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_c: torch.Tensor,
        x_next: torch.Tensor,
    ):
        """Infer: S5 logits + both presence logits."""
        with torch.no_grad():
            mu_c, _lv, skips = self.encode_mu(x_c)
            seg = self.s5.decoder(mu_c, skips)
            mu_p, _, _ = self.encode_mu(x_prev)
            mu_n, _, _ = self.encode_mu(x_next)
        logit_mu = self.head_mu(mu_c)
        logit_ctx = self.head_ctx(mu_p, mu_c, mu_n)
        return seg, logit_mu, logit_ctx
