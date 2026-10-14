"""2.5D FLAIR presence classifier: ResNet18 on (z-1, z, z+1) as 3ch."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class Presence25D(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        del pretrained  # medical domain; random init
        m = resnet18(weights=None)
        # keep 3ch stem (ImageNet layout); re-init for MRI stacks
        nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 1),
        )
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] → logit [B]."""
        return self.net(x).squeeze(1)
