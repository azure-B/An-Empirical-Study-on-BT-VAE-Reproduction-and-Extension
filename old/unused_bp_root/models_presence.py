"""Presence head on top of original VAE. Does not modify original/."""
from torch import nn
import torch


class PresenceHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


class PresenceVAE(nn.Module):
    """Wraps original VAE. forward() stays (out, mu, logvar) for existing eval."""

    def __init__(self, vae: nn.Module, base: int = 32):
        super().__init__()
        self.vae = vae
        self.head = PresenceHead(32 * base)
        self.gate_thr: float | None = None

    def presence_logit(self, mu: torch.Tensor) -> torch.Tensor:
        return self.head(mu)

    def forward(self, x):
        out, mu, logvar = self.vae(x)
        if self.gate_thr is not None:
            present = torch.sigmoid(self.head(mu)) >= self.gate_thr
            out = out.masked_fill(~present.view(out.size(0), 1, 1, 1), -20.0)
        return out, mu, logvar
