"""BT-VAE + Attention U-Net skips with learnable gamma blend.

Relative to original/models1.py:
- same ResBlockDown/Up Encoder-Decoder backbone
- NEW: multi-scale encoder features e0..e4
- NEW: Attention Gate (Oktay) on each decoder up stage
- NEW: per-stage scalar gamma in [0,1]:
      skip = gamma * A + (1 - gamma) * B, then x = x + skip
      A = attention-gated skip, B = ungated (additive) skip

No deep supervision, no uncertainty gate, no MS-GSF fuse.
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


class AttentionGate(nn.Module):
    """Oktay Attention U-Net gate: alpha = sigma(psi(ReLU(W_x x + W_g g)))."""

    def __init__(self, enc_ch: int, dec_ch: int, inter_ch: int | None = None):
        super().__init__()
        inter_ch = inter_ch or max(enc_ch // 2, 1)
        self.W_x = nn.Sequential(
            nn.Conv2d(enc_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(dec_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: encoder skip, g: decoder gating signal
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.relu(self.W_x(x) + self.W_g(g))
        alpha = self.psi(a)
        return x * alpha


class Encoder(nn.Module):
    """Original-style encoder + multi-scale skips for AG decoder."""

    def __init__(self, base: int = 32, dropout_rate: float = 0.1, latent_ch: int | None = None):
        super().__init__()
        self.base = base
        # original: mu/logvar channels = 32 * base (== 1024 when base=32)
        self.latent_ch = latent_ch or (32 * base)
        feat_ch = 32 * base

        self.encoder_initial_flair = self._create_initial_encoder(1, base, dropout_rate)
        self.encoder_features_flair = self._create_feature_encoder(base, dropout_rate)
        self.encoder_mu_flair = nn.Sequential(
            nn.Conv2d(feat_ch, self.latent_ch, 1),
            nn.BatchNorm2d(self.latent_ch),
        )
        self.encoder_logvar_flair = nn.Sequential(
            nn.Conv2d(feat_ch, self.latent_ch, 1),
            nn.BatchNorm2d(self.latent_ch),
        )
        # BT projector on bottleneck (kept for train_skiponly Barlow path)
        self.bt_projector = nn.Sequential(
            nn.Linear(feat_ch, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

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

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_features(self, flair):
        e0 = self.encoder_initial_flair(flair)       # base, 256
        e1 = self.encoder_features_flair[0](e0)      # 2b, 128
        e1d = self.encoder_features_flair[1](e1)
        e2 = self.encoder_features_flair[2](e1d)     # 4b, 64
        e2d = self.encoder_features_flair[3](e2)
        e3 = self.encoder_features_flair[4](e2d)     # 8b, 32
        e3d = self.encoder_features_flair[5](e3)
        e4 = self.encoder_features_flair[6](e3d)     # 16b, 16
        e4d = self.encoder_features_flair[7](e4)
        h = self.encoder_features_flair[8](e4d)      # 32b, 8
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
        # original API: mu, logvar, z
        mu, logvar, z, _, _ = self.forward_features(flair)
        return mu, logvar, z


class Decoder(nn.Module):
    """Up stack + AG/additive skip blend via learnable gamma, then residual add."""

    def __init__(self, base: int = 32, dropout_rate: float = 0.1, latent_ch: int | None = None):
        super().__init__()
        self.base = base
        latent_ch = latent_ch or (32 * base)
        self.latent_ch = latent_ch
        feat_ch = 32 * base

        # If latent_ch != feat_ch (rare), project; original was equal (32*base)
        self.from_latent = (
            nn.Identity()
            if latent_ch == feat_ch
            else nn.Sequential(
                nn.Conv2d(latent_ch, feat_ch, 1, bias=False),
                nn.BatchNorm2d(feat_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        self.up1 = ResBlockUp(feat_ch, feat_ch)          # 8 -> 16
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.up2 = ResBlockUp(feat_ch, feat_ch)          # 16 -> 32
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.up3 = ResBlockUp(feat_ch, 16 * base)        # 32 -> 64
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.up4 = ResBlockUp(16 * base, 8 * base)       # 64 -> 128
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.up5 = ResBlockUp(8 * base, base)            # 128 -> 256
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.out_conv = nn.Conv2d(base, 1, 3, stride=1, padding=1)

        # Attention gates: (enc_ch, dec_ch after up)
        self.ag4 = AttentionGate(16 * base, feat_ch)
        self.ag3 = AttentionGate(8 * base, feat_ch)
        self.ag2 = AttentionGate(4 * base, 16 * base)
        self.ag1 = AttentionGate(2 * base, 8 * base)
        self.ag0 = AttentionGate(base, base)

        # channel align skip -> decoder channels (shared by A and B)
        self.proj4 = nn.Conv2d(16 * base, feat_ch, 1, bias=False)
        self.proj3 = nn.Conv2d(8 * base, feat_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(4 * base, 16 * base, 1, bias=False)
        self.proj1 = nn.Conv2d(2 * base, 8 * base, 1, bias=False)
        self.proj0 = nn.Identity()

        # Direct scalar gamma in [0,1] (NOT logit+sigmoid).
        # Sigmoid at 0.01 saturates (σ'=σ(1-σ)≈0.01) so γ barely moves; raw param keeps full grad.
        _g0 = torch.tensor([float(0.01)])
        self.gamma4 = nn.Parameter(_g0.clone())
        self.gamma3 = nn.Parameter(_g0.clone())
        self.gamma2 = nn.Parameter(_g0.clone())
        self.gamma1 = nn.Parameter(_g0.clone())
        self.gamma0 = nn.Parameter(_g0.clone())

    def gamma_values(self) -> dict[str, float]:
        """Detached clamped gammas for logging (1=full AG, 0=full additive)."""
        with torch.no_grad():
            return {
                "g4": float(self.gamma4.clamp(0.0, 1.0).item()),
                "g3": float(self.gamma3.clamp(0.0, 1.0).item()),
                "g2": float(self.gamma2.clamp(0.0, 1.0).item()),
                "g1": float(self.gamma1.clamp(0.0, 1.0).item()),
                "g0": float(self.gamma0.clamp(0.0, 1.0).item()),
            }

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        g = gamma.clamp(0.0, 1.0).view(1, 1, 1, 1)
        return g * a + (1.0 - g) * b

    def forward(self, z, skips=None):
        if skips is None:
            raise ValueError("Attention-skip Decoder requires encoder skips (e0..e4).")
        e0, e1, e2, e3, e4 = skips

        x = self.from_latent(z)

        # A = AG(skip), B = ungated skip; residual-add the blend
        x = self.drop1(self.up1(x))
        a4, b4 = self.proj4(self.ag4(e4, x)), self.proj4(e4)
        x = x + self._blend(a4, b4, self.gamma4)

        x = self.drop2(self.up2(x))
        a3, b3 = self.proj3(self.ag3(e3, x)), self.proj3(e3)
        x = x + self._blend(a3, b3, self.gamma3)

        x = self.drop3(self.up3(x))
        a2, b2 = self.proj2(self.ag2(e2, x)), self.proj2(e2)
        x = x + self._blend(a2, b2, self.gamma2)

        x = self.drop4(self.up4(x))
        a1, b1 = self.proj1(self.ag1(e1, x)), self.proj1(e1)
        x = x + self._blend(a1, b1, self.gamma1)

        x = self.drop5(self.up5(x))
        a0, b0 = self.proj0(self.ag0(e0, x)), self.proj0(e0)
        x = x + self._blend(a0, b0, self.gamma0)

        return self.out_conv(x)


class VAE(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, latent_ch: int | None = None):
        super().__init__()
        latent_ch = latent_ch or (32 * base)
        self.encoder = Encoder(base, dropout_rate, latent_ch)
        self.decoder = Decoder(base, dropout_rate, latent_ch)

    def forward(self, x, sample: bool | None = None):
        mu, logvar, z, skips, _ = self.encoder.forward_features(x)
        if sample is False:
            z = mu
        out = self.decoder(z, skips)
        return out, mu, logvar
