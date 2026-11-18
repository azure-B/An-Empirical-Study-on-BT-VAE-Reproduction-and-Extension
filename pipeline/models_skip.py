"""BT-VAE+S: same backbone as BT-VAE_CODE/models1.py + skip connections.

Encoder weight keys match BT-VAE_CODE Encoder so Barlow Twins checkpoints load.
Decoder: concat+3x3 fuse on S1..S5 (e0..e4) with learnable softplus scales α
that can approach 0 (effectively off).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
_models1 = ROOT / "BT-VAE_CODE" / "models1.py"
if not _models1.exists():
    raise FileNotFoundError(f"BT-VAE_CODE/models1.py required: {_models1}")
_spec = importlib.util.spec_from_file_location("btvae_code_models1", _models1)
_m1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_m1)

# Exact blocks from paper code
ResBlockUp = _m1.ResBlockUp
ResBlockDown = _m1.ResBlockDown


def _inv_softplus(y: float) -> float:
    """raw such that softplus(raw) ≈ y (y > 0)."""
    # softplus(x) = log(1 + exp(x)); inverse = log(exp(y) - 1)
    return math.log(math.expm1(y))


class ConcatConvFuse(nn.Module):
    """cat(decoder, skip) -> 3x3 Conv -> BN -> LeakyReLU."""

    def __init__(self, dec_ch: int, skip_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_ch + skip_ch, dec_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class Encoder(nn.Module):
    """BT-VAE_CODE Encoder layout + skip feature taps for +S."""

    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        self.encoder_initial_flair = self._create_initial_encoder(1, base, dropout_rate)
        self.encoder_features_flair = self._create_feature_encoder(base, dropout_rate)
        self.encoder_mu_flair = self._create_mu_encoder(32 * base)
        self.encoder_logvar_flair = self._create_logvar_encoder(32 * base)

    def _create_initial_encoder(self, in_channels, base, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, base, 3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def _create_feature_encoder(self, base, dropout_rate):
        # Identical channel schedule to BT-VAE_CODE/models1.Encoder
        return nn.Sequential(
            ResBlockDown(base, 2 * base),
            nn.Dropout2d(p=dropout_rate),
            ResBlockDown(2 * base, 4 * base),
            nn.Dropout2d(p=dropout_rate),
            ResBlockDown(4 * base, 8 * base),
            nn.Dropout2d(p=dropout_rate),
            ResBlockDown(8 * base, 16 * base),
            nn.Dropout2d(p=dropout_rate),
            ResBlockDown(16 * base, 32 * base),
            nn.Dropout2d(p=dropout_rate),
        )

    def _create_mu_encoder(self, C):
        return nn.Sequential(nn.Conv2d(C, C, 1), nn.BatchNorm2d(C))

    def _create_logvar_encoder(self, C):
        return nn.Sequential(nn.Conv2d(C, C, 1), nn.BatchNorm2d(C))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_features(self, flair):
        e0 = self.encoder_initial_flair(flair)
        e1 = self.encoder_features_flair[0](e0)
        e1d = self.encoder_features_flair[1](e1)
        e2 = self.encoder_features_flair[2](e1d)
        e2d = self.encoder_features_flair[3](e2)
        e3 = self.encoder_features_flair[4](e2d)
        e3d = self.encoder_features_flair[5](e3)
        e4 = self.encoder_features_flair[6](e3d)
        e4d = self.encoder_features_flair[7](e4)
        h = self.encoder_features_flair[8](e4d)
        h = self.encoder_features_flair[9](h)
        mu = self.encoder_mu_flair(h)
        logvar = self.encoder_logvar_flair(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, (e0, e1, e2, e3, e4)

    def forward(self, flair):
        mu, logvar, z, _ = self.forward_features(flair)
        return mu, logvar, z


class Decoder(nn.Module):
    """+S decoder: 5× ResBlockUp + S1..S5 concat fuses with learnable α."""

    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        feat = 32 * base
        # Channel schedule mirrors BT-VAE_CODE Decoder._create_decoder
        self.up1 = ResBlockUp(feat, feat)          # 32b -> 32b
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.up2 = ResBlockUp(feat, feat)          # 32b -> 32b
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.up3 = ResBlockUp(feat, 16 * base)     # 32b -> 16b
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.up4 = ResBlockUp(16 * base, 8 * base) # 16b -> 8b
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.up5 = ResBlockUp(8 * base, base)      # 8b -> base
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.out_conv = nn.Conv2d(base, 1, 3, stride=1, padding=1)

        # S5..S1: after up1..up5 respectively (e4..e0)
        self.fuse4 = ConcatConvFuse(feat, 16 * base)      # after up1 + e4 (S5)
        self.fuse3 = ConcatConvFuse(feat, 8 * base)       # after up2 + e3 (S4)
        self.fuse2 = ConcatConvFuse(16 * base, 4 * base)  # after up3 + e2 (S3)
        self.fuse1 = ConcatConvFuse(8 * base, 2 * base)   # after up4 + e1 (S2)
        self.fuse0 = ConcatConvFuse(base, base)           # after up5 + e0 (S1)

        # α_i = softplus(raw_i); init α≈1 so skips start fully on
        self.skip_raw = nn.Parameter(
            torch.full((5,), float(_inv_softplus(1.0)))
        )

    def skip_alphas(self) -> torch.Tensor:
        """α1..α5 (S1..S5), softplus ≥ 0."""
        return F.softplus(self.skip_raw)

    def forward(self, z, skips):
        e0, e1, e2, e3, e4 = skips
        a1, a2, a3, a4, a5 = self.skip_alphas()

        x = self.drop1(self.up1(z))
        x = self.fuse4(x, e4 * a5)   # S5
        x = self.drop2(self.up2(x))
        x = self.fuse3(x, e3 * a4)   # S4
        x = self.drop3(self.up3(x))
        x = self.fuse2(x, e2 * a3)   # S3
        x = self.drop4(self.up4(x))
        x = self.fuse1(x, e1 * a2)   # S2
        x = self.drop5(self.up5(x))
        x = self.fuse0(x, e0 * a1)   # S1
        return self.out_conv(x)


class VAE(nn.Module):
    """BT-VAE+S model (display name)."""

    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(base, dropout_rate)
        self.decoder = Decoder(base, dropout_rate)

    def skip_alphas(self) -> torch.Tensor:
        return self.decoder.skip_alphas()

    def forward(self, x):
        mu, logvar, z, skips = self.encoder.forward_features(x)
        out = self.decoder(z, skips)
        return out, mu, logvar


# Alias preferred in new code
BTVAE_S = VAE
