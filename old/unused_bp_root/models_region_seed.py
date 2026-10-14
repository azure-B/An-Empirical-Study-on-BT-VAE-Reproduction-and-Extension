"""Original BT-VAE decoder + one-parameter region seed (no skips).

Does NOT modify original/. Encoder module names match original.models1.Encoder
so BT checkpoints load. Decoder blocks match original.models1.Decoder; the only
added parameter is `h` (growth allowance).

seed = relu(softplus(mean_c(z)) - h)
Each upsample: bilinear up seed → 3x3 max-pool dilate → clip by decoder energy.
Decoder features are gated by (1 + seed). Broadcast, no extra conv.
"""
from torch import nn
import torch
import torch.nn.functional as F


class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()
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
        super(ResBlockDown, self).__init__()
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
    def __init__(self, base, dropout_rate=0.1):
        super(Encoder, self).__init__()
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

    def encode_sequence(self, x, initial, features, mu_layer, logvar_layer):
        x = initial(x)
        x = features(x)
        mu = mu_layer(x)
        logvar = logvar_layer(x)
        return mu, logvar

    def forward(self, flair):
        mu_flair, logvar_flair = self.encode_sequence(
            flair,
            self.encoder_initial_flair,
            self.encoder_features_flair,
            self.encoder_mu_flair,
            self.encoder_logvar_flair,
        )
        z = self.reparameterize(mu_flair, logvar_flair)
        return mu_flair, logvar_flair, z


class Decoder(nn.Module):
    """Original 5× ResBlockUp stack + region-seed grow. Only extra param: h."""

    def __init__(self, base, dropout_rate=0.1, h_init: float = 0.3):
        super(Decoder, self).__init__()
        self.up1 = ResBlockUp(32 * base, 32 * base)
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.up2 = ResBlockUp(32 * base, 32 * base)
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.up3 = ResBlockUp(32 * base, 16 * base)
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.up4 = ResBlockUp(16 * base, 8 * base)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.up5 = ResBlockUp(8 * base, base)
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.out_conv = nn.Conv2d(base, 1, 3, stride=1, padding=1)
        self.h = nn.Parameter(torch.tensor([float(h_init)]))

    def h_value(self) -> float:
        with torch.no_grad():
            return float(self.h.item())

    def _init_seed(self, z: torch.Tensor) -> torch.Tensor:
        energy = F.softplus(z.mean(dim=1, keepdim=True))
        return F.relu(energy - self.h)

    def _grow(self, x: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seed = F.interpolate(seed, size=x.shape[-2:], mode="bilinear", align_corners=False)
        seed = F.max_pool2d(seed, kernel_size=3, stride=1, padding=1)
        energy = F.softplus(x.mean(dim=1, keepdim=True))
        seed = torch.minimum(seed, energy)
        x = x * (1.0 + seed)
        return x, seed

    def forward(self, z):
        seed = self._init_seed(z)
        x = self.drop1(self.up1(z))
        x, seed = self._grow(x, seed)
        x = self.drop2(self.up2(x))
        x, seed = self._grow(x, seed)
        x = self.drop3(self.up3(x))
        x, seed = self._grow(x, seed)
        x = self.drop4(self.up4(x))
        x, seed = self._grow(x, seed)
        x = self.drop5(self.up5(x))
        x, seed = self._grow(x, seed)
        return self.out_conv(x)


class VAE(nn.Module):
    def __init__(self, base, dropout_rate=0.1, h_init: float = 0.3):
        super(VAE, self).__init__()
        self.encoder = Encoder(base, dropout_rate)
        self.decoder = Decoder(base, dropout_rate, h_init=h_init)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        out = self.decoder(z)
        return out, mu, logvar
