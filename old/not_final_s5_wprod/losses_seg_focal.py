"""Dice + Sigmoid Focal for small-lesion SEG; focal BCE for presence cls."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_dice_loss_sigmoid(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Pixel-wise focal BCE-with-logits. alpha = weight on positive class."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    mod = (1 - p_t).pow(gamma)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * mod * ce).mean()


def small_lesion_weight(target: torch.Tensor, power: float = 0.5, eps: float = 1.0) -> torch.Tensor:
    """Per-sample weight: smaller GT area → larger weight (batch)."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    area = target.flatten(1).sum(dim=1).clamp_min(0.0)
    # nonempty only meaningfully weighted; empty → 1
    w = torch.ones_like(area)
    ne = area > 0
    w[ne] = (area[ne] + eps).pow(-power)
    w = w / w.mean().clamp_min(1e-6)
    return w


def dice_focal_seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    gamma: float = 2.0,
    alpha: float = 0.75,
    size_power: float = 0.5,
) -> tuple[torch.Tensor, float, float]:
    """
    Combined SEG loss. Focal emphasizes hard/rare pixels; size_power
    up-weights small lesions in the dice term (per-image weighted).
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()

    # size-weighted soft dice (per image then mean)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice_i = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
    w = small_lesion_weight(target, power=size_power)
    dice = (dice_i * w).mean()

    focal = sigmoid_focal_loss(logits, target, gamma=gamma, alpha=alpha)
    loss = dice_weight * dice + focal_weight * focal
    return loss, float(dice.detach()), float(focal.detach())


def focal_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.6,
) -> torch.Tensor:
    """Case-level focal BCE for presence (logit [B], target [B])."""
    target = target.float()
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    mod = (1 - p_t).pow(gamma)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * mod * ce).mean()
