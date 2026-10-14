"""BT encoder cascade: shared μ forked to Cls decoder + SEG decoder.

  x → BT Encoder (frozen) → μ (+ skips)
        ├─ μ → Decoder_cls → presence − τ
        └─ μ + skips → Decoder_seg (S5)

Same μ is intentional: one frozen BT latent, two task heads.
Infer: keep SEG iff presence_logit ≥ τ_logit.
"""
from __future__ import annotations

import torch
from torch import nn

from models_bp_skip import VAE as SkipVAE
from train_original_bp import OrigVAE, load_bt_into_encoder


class BTCascadeVAE(nn.Module):
    def __init__(self, base: int = 32, dropout_rate: float = 0.1, tau_init: float = 0.0):
        super().__init__()
        self.base = base
        skip = SkipVAE(base, dropout_rate)
        plain = OrigVAE(base, dropout_rate)
        self.encoder = skip.encoder
        self.decoder_seg = skip.decoder
        self.decoder_cls = plain.decoder
        self.tau_logit = nn.Parameter(torch.tensor(float(tau_init)))
        del skip, plain

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def set_trainable(self, seg: bool, cls: bool, tau: bool | None = None) -> None:
        for p in self.decoder_seg.parameters():
            p.requires_grad = seg
        for p in self.decoder_cls.parameters():
            p.requires_grad = cls
        if tau is None:
            tau = cls
        self.tau_logit.requires_grad = bool(tau)

    def encode_det(self, x: torch.Tensor):
        mu, logvar, _z, skips = self.encoder.forward_features(x)
        return mu, logvar, skips

    def cls_logit_from_map(self, cls_map: torch.Tensor) -> torch.Tensor:
        return cls_map.flatten(1).amax(dim=1)

    def decision_logit(self, presence_logit: torch.Tensor) -> torch.Tensor:
        return presence_logit - self.tau_logit

    def tau_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.tau_logit)

    def forward_parts(self, x: torch.Tensor):
        mu, logvar, skips = self.encode_det(x)
        seg_logits = self.decoder_seg(mu, skips)
        cls_map = self.decoder_cls(mu)
        presence_logit = self.cls_logit_from_map(cls_map)
        decision = self.decision_logit(presence_logit)
        return seg_logits, cls_map, presence_logit, decision, mu, logvar, skips

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        seg_logits, cls_map, presence_logit, decision, mu, logvar, skips = self.forward_parts(x)
        keep = decision >= 0
        seg_logits = seg_logits.masked_fill(~keep.view(-1, 1, 1, 1), -20.0)
        if return_parts:
            return seg_logits, cls_map, presence_logit, decision, mu, logvar, skips
        return seg_logits, mu, logvar


def load_bt_cascade(model: BTCascadeVAE, bt_ckpt, s5_ckpt: dict | None, none_ckpt: dict | None) -> None:
    load_bt_into_encoder(model.encoder, bt_ckpt)
    if s5_ckpt is not None:
        s5 = SkipVAE(model.base)
        s5.load_state_dict(s5_ckpt["model"])
        model.decoder_seg.load_state_dict(s5.decoder.state_dict())
        del s5
    if none_ckpt is not None:
        plain = OrigVAE(model.base)
        plain.load_state_dict(none_ckpt["model"])
        model.decoder_cls.load_state_dict(plain.decoder.state_dict())
        del plain
