"""MS-GSF-BT-VAE: multi-scale probabilistic encoder and uncertainty-gated decoder.
Public API remains compatible with the original notebook:
    Encoder(x) -> mu, logvar, z
    VAE(x) -> logits, mu, logvar
Use VAE(x, return_aux=True) during training for deep supervision.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int, maximum: int = 8) -> int:
    for g in range(min(maximum, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(cin, cout, 3, 1, dropout),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.skip(x) + self.body(x))


class ResidualDown(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(cin, cout, 3, 2, dropout),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
        )
        self.skip = nn.Conv2d(cin, cout, 1, stride=2, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.skip(x) + self.body(x))


class Encoder(nn.Module):
    """Five-scale probabilistic encoder. Input H/W should be divisible by 32."""
    def __init__(self, base: int = 16, dropout_rate: float = 0.05,
                 latent_ch: int | None = None, projector_dim: int = 256):
        super().__init__()
        latent_ch = latent_ch or 8 * base
        c0, c1, c2, c3, c4 = base, 2 * base, 4 * base, 8 * base, 16 * base
        self.base = base
        self.latent_ch = latent_ch
        self.channels = (c0, c1, c2, c3, c4)

        self.stem = nn.Sequential(ConvGNAct(1, c0, dropout=dropout_rate), ResidualBlock(c0, c0))
        self.down1 = ResidualDown(c0, c1, dropout_rate)
        self.down2 = ResidualDown(c1, c2, dropout_rate)
        self.down3 = ResidualDown(c2, c3, dropout_rate)
        self.down4 = ResidualDown(c3, c4, dropout_rate)
        self.bottleneck = nn.Sequential(
            ResidualDown(c4, c4, dropout_rate),
            ResidualBlock(c4, c4, dropout_rate),
        )
        self.mu_head = nn.Conv2d(c4, latent_ch, 1)
        self.logvar_head = nn.Conv2d(c4, latent_ch, 1)
        self.bt_projector = nn.Sequential(
            nn.Linear(c4, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.GELU(),
            nn.Linear(projector_dim, projector_dim),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode_with_skips(self, x: torch.Tensor):
        e0 = self.stem(x)       # 256
        e1 = self.down1(e0)     # 128
        e2 = self.down2(e1)     # 64
        e3 = self.down3(e2)     # 32
        e4 = self.down4(e3)     # 16
        h = self.bottleneck(e4) # 8
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-6.0, 2.0)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, (e0, e1, e2, e3, e4), h

    def forward_bt(self, x: torch.Tensor):
        mu, logvar, _, _, h = self.encode_with_skips(x)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        projection = self.bt_projector(pooled)
        return projection, mu, logvar

    def forward(self, x: torch.Tensor):
        mu, logvar, z, _, _ = self.encode_with_skips(x)
        return mu, logvar, z


class UncertaintyGatedSkip(nn.Module):
    def __init__(self, enc_ch: int, dec_ch: int, out_ch: int):
        super().__init__()
        self.enc_proj = nn.Conv2d(enc_ch, out_ch, 1, bias=False)
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, 1, bias=False)
        self.unc_proj = nn.Conv2d(1, out_ch, 1, bias=False)
        self.psi = nn.Conv2d(out_ch, 1, 1)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, enc: torch.Tensor, dec: torch.Tensor,
                uncertainty: torch.Tensor) -> torch.Tensor:
        dec_up = F.interpolate(dec, size=enc.shape[-2:], mode="bilinear", align_corners=False)
        unc = F.interpolate(uncertainty, size=enc.shape[-2:], mode="bilinear", align_corners=False)
        gamma = F.softplus(self.gamma_raw)
        score = self.enc_proj(enc) + self.dec_proj(dec_up) - gamma * self.unc_proj(unc)
        alpha = torch.sigmoid(self.psi(F.silu(score)))
        return self.enc_proj(enc) * alpha


class UpFuseBlock(nn.Module):
    def __init__(self, dec_ch: int, enc_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, 1, bias=False)
        self.gate = UncertaintyGatedSkip(enc_ch, dec_ch, out_ch)
        self.fuse = ResidualBlock(2 * out_ch, out_ch, dropout)

    def forward(self, dec: torch.Tensor, enc: torch.Tensor,
                uncertainty: torch.Tensor) -> torch.Tensor:
        dec_up = F.interpolate(dec, size=enc.shape[-2:], mode="bilinear", align_corners=False)
        dec_up = self.dec_proj(dec_up)
        enc_gated = self.gate(enc, dec, uncertainty)
        return self.fuse(torch.cat((dec_up, enc_gated), dim=1))


class Decoder(nn.Module):
    def __init__(self, base: int = 16, dropout_rate: float = 0.05,
                 latent_ch: int | None = None):
        super().__init__()
        latent_ch = latent_ch or 8 * base
        c0, c1, c2, c3, c4 = base, 2 * base, 4 * base, 8 * base, 16 * base
        self.from_latent = ResidualBlock(latent_ch, c4, dropout_rate)
        self.up4 = UpFuseBlock(c4, c4, c4, dropout_rate)
        self.up3 = UpFuseBlock(c4, c3, c3, dropout_rate)
        self.up2 = UpFuseBlock(c3, c2, c2, dropout_rate)
        self.up1 = UpFuseBlock(c2, c1, c1, dropout_rate)
        self.up0 = UpFuseBlock(c1, c0, c0, dropout_rate)
        self.head = nn.Conv2d(c0, 1, 1)
        self.aux_64 = nn.Conv2d(c2, 1, 1)
        self.aux_128 = nn.Conv2d(c1, 1, 1)

    def forward(self, z: torch.Tensor, skips, logvar: torch.Tensor,
                return_aux: bool = False):
        e0, e1, e2, e3, e4 = skips
        uncertainty = torch.exp(logvar.clamp(-6.0, 2.0)).mean(dim=1, keepdim=True)
        d = self.from_latent(z)
        d4 = self.up4(d, e4, uncertainty)
        d3 = self.up3(d4, e3, uncertainty)
        d2 = self.up2(d3, e2, uncertainty)
        d1 = self.up1(d2, e1, uncertainty)
        d0 = self.up0(d1, e0, uncertainty)
        logits = self.head(d0)
        if not return_aux:
            return logits
        size = logits.shape[-2:]
        aux = [
            F.interpolate(self.aux_128(d1), size=size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_64(d2), size=size, mode="bilinear", align_corners=False),
        ]
        return logits, aux


class VAE(nn.Module):
    def __init__(self, base: int = 16, dropout_rate: float = 0.05,
                 latent_ch: int | None = None):
        super().__init__()
        self.encoder = Encoder(base, dropout_rate, latent_ch)
        self.decoder = Decoder(base, dropout_rate, latent_ch)

    def forward(self, x: torch.Tensor, sample: bool | None = None,
                return_aux: bool = False):
        mu, logvar, z, skips, _ = self.encoder.encode_with_skips(x)
        if sample is False:
            z = mu
        elif sample is True and not self.training:
            z = self.encoder.reparameterize(mu, logvar)
        decoded = self.decoder(z, skips, logvar, return_aux=return_aux)
        if return_aux:
            logits, aux = decoded
            return logits, mu, logvar, aux
        return decoded, mu, logvar
