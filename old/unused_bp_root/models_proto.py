"""Spatial prototype readout on frozen BT encoder. Does not modify original/.

Better μ: keep μ as an 8×8 map (no GAP), concat the 16×16 encoder map,
then a small 1×1 projector + FG/BG prototypes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_spatial(encoder: nn.Module, x: torch.Tensor):
    """μ [B,1024,8,8] and last 16×16 map [B,512,16,16]."""
    x = encoder.encoder_initial_flair(x)
    mid16 = None
    for m in encoder.encoder_features_flair:
        x = m(x)
        if x.shape[-1] == 16:
            mid16 = x
    mu = encoder.encoder_mu_flair(x)
    if mid16 is None:
        mid16 = F.interpolate(mu, size=(16, 16), mode="bilinear", align_corners=False)
    return mu, mid16


class ProtoHead(nn.Module):
    def __init__(self, mu_ch: int = 1024, mid_ch: int = 512, proj_ch: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(mu_ch + mid_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, proj_ch, 1, bias=False),
            nn.BatchNorm2d(proj_ch),
        )
        self.fg = nn.Parameter(torch.randn(proj_ch))
        self.bg = nn.Parameter(torch.randn(proj_ch))
        self.scale = nn.Parameter(torch.tensor(5.0))

    def embed(self, mu: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        mu16 = F.interpolate(mu, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        z = self.proj(torch.cat([mu16, mid], dim=1))
        return F.normalize(z, dim=1)

    def logits_at(self, z: torch.Tensor) -> torch.Tensor:
        fg = F.normalize(self.fg, dim=0).view(1, -1, 1, 1)
        bg = F.normalize(self.bg, dim=0).view(1, -1, 1, 1)
        return self.scale * ((z * fg).sum(1, keepdim=True) - (z * bg).sum(1, keepdim=True))

    def forward(self, mu: torch.Tensor, mid: torch.Tensor, out_size=256) -> torch.Tensor:
        logit = self.logits_at(self.embed(mu, mid))
        return F.interpolate(logit, size=out_size, mode="bilinear", align_corners=False)
