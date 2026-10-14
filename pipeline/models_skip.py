"""BP original + S5-style concat skip (B0..B3, B4 off).

Does NOT modify original/. Encoder module names match original.models1.Encoder
so BT checkpoints load with strict=False / matching keys.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
    """Same weights layout as original.models1.Encoder; forward also returns skips."""

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
    """S5: B0..B3 concat+3x3 fuse, B4 off."""

    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        feat = 32 * base
        self.up1 = ResBlockUp(feat, feat)
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.up2 = ResBlockUp(feat, feat)
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.up3 = ResBlockUp(feat, 16 * base)
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.up4 = ResBlockUp(16 * base, 8 * base)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.up5 = ResBlockUp(8 * base, base)
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.out_conv = nn.Conv2d(base, 1, 3, stride=1, padding=1)

        self.fuse3 = ConcatConvFuse(feat, 8 * base)  # after up2 + e3
        self.fuse2 = ConcatConvFuse(16 * base, 4 * base)
        self.fuse1 = ConcatConvFuse(8 * base, 2 * base)
        self.fuse0 = ConcatConvFuse(base, base)

    def forward(self, z, skips):
        e0, e1, e2, e3, e4 = skips  # e4 unused (B4 off)
        del e4

        x = self.drop1(self.up1(z))
        x = self.drop2(self.up2(x))
        x = self.fuse3(x, e3)
        x = self.drop3(self.up3(x))
        x = self.fuse2(x, e2)
        x = self.drop4(self.up4(x))
        x = self.fuse1(x, e1)
        x = self.drop5(self.up5(x))
        x = self.fuse0(x, e0)
        return self.out_conv(x)


class VAE(nn.Module):
    def __init__(self, base, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(base, dropout_rate)
        self.decoder = Decoder(base, dropout_rate)

    def forward(self, x):
        mu, logvar, z, skips = self.encoder.forward_features(x)
        out = self.decoder(z, skips)
        return out, mu, logvar
