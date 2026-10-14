"""BT-VAE skip ablations (no Attention Gate).

Skip index (high-res → low-res):
  B0 = e0 (256), B1 = e1 (128), B2 = e2 (64), B3 = e3 (32), B4 = e4 (16)

Modes / stages:
  none         — S1 original: no skips
  full         — S2 additive residual B0..B4  (reuse prior run if present)
  b0b3         — S3 additive residual B0..B3 (B4 off only)
  b0b3_learn   — S4 B0..B3 zero-init 1x1 + learnable scale (init 0.01)
  b0b3_concat  — S5 B0..B3 concat(decoder, skip) → 3x3 Conv fuse
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SKIP_NONE = "none"
SKIP_FULL = "full"
SKIP_B0B3 = "b0b3"
SKIP_B0B3_LEARN = "b0b3_learn"
SKIP_B0B3_CONCAT = "b0b3_concat"


class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super().__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_in, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(filters_in),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(filters_out),
        )
        self.conv3_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(filters_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)
        return conv2 + conv3


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super().__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_in, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters_in),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(filters_out),
        )
        self.conv3_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters_out),
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)
        return conv2 + conv3


class Encoder(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, latent_ch: int | None = None):
        super().__init__()
        self.base = base
        self.latent_ch = latent_ch or (32 * base)
        feat_ch = 32 * base

        self.encoder_initial_flair = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )
        self.encoder_features_flair = nn.Sequential(
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
        self.encoder_mu_flair = nn.Sequential(
            nn.Conv2d(feat_ch, self.latent_ch, 1),
            nn.BatchNorm2d(self.latent_ch),
        )
        self.encoder_logvar_flair = nn.Sequential(
            nn.Conv2d(feat_ch, self.latent_ch, 1),
            nn.BatchNorm2d(self.latent_ch),
        )
        self.bt_projector = nn.Sequential(
            nn.Linear(feat_ch, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-6.0, 2.0)
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
        logvar = self.encoder_logvar_flair(h).clamp(-6.0, 2.0)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, (e0, e1, e2, e3, e4), h

    def forward_bt(self, flair):
        mu, logvar, _, _, h = self.forward_features(flair)
        proj = self.bt_projector(F.adaptive_avg_pool2d(h, 1).flatten(1))
        return proj, mu, logvar

    def forward(self, flair):
        mu, logvar, z, _, _ = self.forward_features(flair)
        return mu, logvar, z


def _zero_init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ConcatConvFuse(nn.Module):
    """cat(decoder, skip) → 3x3 Conv → BN → LeakyReLU."""

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


class Decoder(nn.Module):
    def __init__(
        self,
        base: int = 32,
        dropout_rate: float = 0.1,
        latent_ch: int | None = None,
        skip_mode: str = SKIP_FULL,
    ):
        super().__init__()
        allowed = {SKIP_NONE, SKIP_FULL, SKIP_B0B3, SKIP_B0B3_LEARN, SKIP_B0B3_CONCAT}
        if skip_mode not in allowed:
            raise ValueError(f"unknown skip_mode={skip_mode}")
        self.skip_mode = skip_mode
        self.base = base
        latent_ch = latent_ch or (32 * base)
        self.latent_ch = latent_ch
        feat_ch = 32 * base

        self.from_latent = (
            nn.Identity()
            if latent_ch == feat_ch
            else nn.Sequential(
                nn.Conv2d(latent_ch, feat_ch, 1, bias=False),
                nn.BatchNorm2d(feat_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        self.up1 = ResBlockUp(feat_ch, feat_ch)
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.up2 = ResBlockUp(feat_ch, feat_ch)
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.up3 = ResBlockUp(feat_ch, 16 * base)
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.up4 = ResBlockUp(16 * base, 8 * base)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.up5 = ResBlockUp(8 * base, base)
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.out_conv = nn.Conv2d(base, 1, 3, stride=1, padding=1)

        # additive / learnable: 1x1 align then residual add
        self.proj4 = nn.Conv2d(16 * base, feat_ch, 1, bias=False)
        self.proj3 = nn.Conv2d(8 * base, feat_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(4 * base, 16 * base, 1, bias=False)
        self.proj1 = nn.Conv2d(2 * base, 8 * base, 1, bias=False)
        self.proj0 = nn.Identity()

        # concat fuse (S5): after up2..up5 → B3..B0
        self.fuse3 = ConcatConvFuse(feat_ch, 8 * base)       # up2: feat + e3
        self.fuse2 = ConcatConvFuse(16 * base, 4 * base)     # up3
        self.fuse1 = ConcatConvFuse(8 * base, 2 * base)      # up4
        self.fuse0 = ConcatConvFuse(base, base)              # up5

        self.scale3 = self.scale2 = self.scale1 = self.scale0 = None
        if skip_mode == SKIP_B0B3_LEARN:
            _zero_init_conv(self.proj3)
            _zero_init_conv(self.proj2)
            _zero_init_conv(self.proj1)
            self.proj0 = _zero_init_conv(nn.Conv2d(base, base, 1, bias=False))
            # Must be >0: if scale=0 and proj=0, BOTH get zero grad (dead skip path)
            _s0 = torch.tensor([0.01])
            self.scale3 = nn.Parameter(_s0.clone())
            self.scale2 = nn.Parameter(_s0.clone())
            self.scale1 = nn.Parameter(_s0.clone())
            self.scale0 = nn.Parameter(_s0.clone())

        # (use_b4, use_b3, use_b2, use_b1, use_b0)
        if skip_mode == SKIP_NONE:
            self.active = (False, False, False, False, False)
        elif skip_mode == SKIP_FULL:
            self.active = (True, True, True, True, True)
        else:
            # B0..B3 on, B4 off only
            self.active = (False, True, True, True, True)

    def scale_values(self) -> dict[str, float] | None:
        if self.skip_mode != SKIP_B0B3_LEARN:
            return None
        with torch.no_grad():
            return {
                "s3": float(self.scale3.item()),
                "s2": float(self.scale2.item()),
                "s1": float(self.scale1.item()),
                "s0": float(self.scale0.item()),
            }

    def _add_skip(self, x, e, proj, use: bool, scale: nn.Parameter | None = None):
        if not use:
            return x
        s = proj(e)
        if scale is not None:
            s = scale.view(1, 1, 1, 1) * s
        return x + s

    def forward(self, z, skips=None):
        if self.skip_mode != SKIP_NONE and skips is None:
            raise ValueError(f"skip_mode={self.skip_mode} requires encoder skips")
        e0 = e1 = e2 = e3 = e4 = None
        if skips is not None:
            e0, e1, e2, e3, e4 = skips
        use_b4, use_b3, use_b2, use_b1, use_b0 = self.active
        concat = self.skip_mode == SKIP_B0B3_CONCAT

        x = self.from_latent(z)

        x = self.drop1(self.up1(x))
        x = self._add_skip(x, e4, self.proj4, use_b4)  # B4 only in full

        x = self.drop2(self.up2(x))
        if use_b3:
            if concat:
                x = self.fuse3(x, e3)
            else:
                x = self._add_skip(x, e3, self.proj3, True, self.scale3)

        x = self.drop3(self.up3(x))
        if use_b2:
            if concat:
                x = self.fuse2(x, e2)
            else:
                x = self._add_skip(x, e2, self.proj2, True, self.scale2)

        x = self.drop4(self.up4(x))
        if use_b1:
            if concat:
                x = self.fuse1(x, e1)
            else:
                x = self._add_skip(x, e1, self.proj1, True, self.scale1)

        x = self.drop5(self.up5(x))
        if use_b0:
            if concat:
                x = self.fuse0(x, e0)
            else:
                x = self._add_skip(x, e0, self.proj0, True, self.scale0)

        return self.out_conv(x)


class VAE(nn.Module):
    def __init__(
        self,
        base: int = 32,
        dropout_rate: float = 0.1,
        latent_ch: int | None = None,
        skip_mode: str = SKIP_FULL,
    ):
        super().__init__()
        latent_ch = latent_ch or (32 * base)
        self.skip_mode = skip_mode
        self.encoder = Encoder(base, dropout_rate, latent_ch)
        self.decoder = Decoder(base, dropout_rate, latent_ch, skip_mode=skip_mode)

    def forward(self, x, sample: bool | None = None):
        mu, logvar, z, skips, _ = self.encoder.forward_features(x)
        if sample is False:
            z = mu
        if self.skip_mode == SKIP_NONE:
            out = self.decoder(z, None)
        else:
            out = self.decoder(z, skips)
        return out, mu, logvar
