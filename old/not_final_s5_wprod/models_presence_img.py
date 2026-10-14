"""FLAIR-only presence classifier (no VAE μ). ResNet18, 1-channel input."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class PresenceImgCls(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        # weights=None: medical domain; ImageNet 3ch mismatch anyway
        m = resnet18(weights=None)
        # 1-channel stem
        w = m.conv1.weight  # unused random init shape ref
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 1),
        )
        self.net = m
        del w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logit [B]."""
        return self.net(x).squeeze(1)
