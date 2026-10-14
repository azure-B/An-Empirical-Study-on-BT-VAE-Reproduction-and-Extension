"""Lesion-presence classification head on frozen VAE μ. original/ untouched."""
from __future__ import annotations

import torch
from torch import nn


class LesionPresenceHead(nn.Module):
    """Case-level presence logit from spatial μ. No sigmoid inside."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        avg_feature = self.avg_pool(feature).flatten(1)
        max_feature = self.max_pool(feature).flatten(1)
        pooled = torch.cat([avg_feature, max_feature], dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(1)
