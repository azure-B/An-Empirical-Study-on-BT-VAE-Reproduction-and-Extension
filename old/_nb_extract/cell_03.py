
import torch
import math
import time
import math
import torch

# LOG2PI = math.log(2.0 * math.pi)

# def gaussian_logpdf(m, logvar, z):
#     """
#     m, logvar, z: (B, C, 1, 1) 또는 (B, C)
#     반환: (B, C)  각 채널별 log p(z|m, logvar)
#     """
#     # (B,C,1,1) -> (B,C)
#     if m.ndim == 4:
#         s = lambda t: t.squeeze(-1).squeeze(-1)
#         m, logvar, z = map(s, (m, logvar, z))

#     # 수치 안정화: logvar 클램프 & var 최소값 보호
#     logvar = logvar.clamp(-6.0, 1.0)
#     var = torch.exp(logvar)  # (B,C)

#     # log N(z; m, var) = -0.5 * [ (z-m)^2/var + log(2π) + log(var) ]
#     diff2 = (z - m) ** 2
#     logpdf = -0.5 * (diff2 / (var + 1e-12) + LOG2PI + logvar)
    
#     return logpdf  # (B, C)


# def barlow_from_logpdf(l1, l2, lambda_offdiag=0.005, eps=1e-4):
#     # l1, l2: (B, C) 또는 (B, C, 1, 1)
#     if l1.ndim == 4:
#         s = lambda t: t.squeeze(-1).squeeze(-1)
#         l1, l2 = map(s, (l1, l2))

#     B = l1.size(0)

#     # 배치 표준화(채널별 0-mean, 1-std)
#     z1 = (l1 - l1.mean(dim=0)) / (l1.std(dim=0) + eps)
#     z2 = (l2 - l2.mean(dim=0)) / (l2.std(dim=0) + eps)

#     # 교차상관행렬 C=(1/B) z1^T z2  → (C,C)
#     C = (z1.T @ z2) / float(B)

#     diag = torch.diag(C)
#     diag_loss = (1.0 - diag).pow(2).sum()           # 대각은 1로
#     off = C - torch.diag(diag)
#     off_loss = (off.pow(2)).sum()                   # 비대각은 0으로

#     return diag_loss + lambda_offdiag * off_loss




conv1x1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

# def concat_v1v2(view1, view2):
#     # (B,1,H,W) + (B,1,H,W) → (B,2,H,W)
#     view3 = torch.cat([view1, view2], dim=1)
#     # 1×1 conv로 채널을 2→1 축소
#     return conv1x1(view3)

# def elementwise_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     return x * y   # torch.mul(x, y) 도 동일




# import math

# import torch

# def kl_diag_gauss(mu_p, logvar_p, mu_q, logvar_q, reduce='mean', eps=1e-12):
#     """
#     KL( N(mu_p, diag(exp(logvar_p))) || N(mu_q, diag(exp(logvar_q))) )
#     입력: (B,C,1,1) 또는 (B,C) 또는 (...,C)
#     반환: (B,) 또는 스칼라(mean/sum) 또는 전체( none )
#     """
#     # (...,C,1,1) -> (...,C)
#     if mu_p.ndim >= 3 and mu_p.shape[-1] == 1 and mu_p.shape[-2] == 1:
#         squeeze2 = lambda x: x.squeeze(-1).squeeze(-1)
#         mu_p, logvar_p, mu_q, logvar_q = map(squeeze2, (mu_p, logvar_p, mu_q, logvar_q))

#     var_p = torch.exp(logvar_p).clamp_min(eps)
#     var_q = torch.exp(logvar_q).clamp_min(eps)

#     diff2 = (mu_q - mu_p) ** 2
#     # per-dim KL
#     kl_per_dim = (var_p + diff2) / var_q + (logvar_q - logvar_p) - 1.0
#     kl = 0.5 * kl_per_dim.sum(dim=-1)  # sum over last dim (features)

#     if reduce == 'mean':
#         return kl.mean()
#     elif reduce == 'none':
#         return kl
#     elif reduce == 'sum':
#         return kl.sum()
#     else:
#         raise ValueError("reduce must be 'mean' | 'sum' | 'none'")

import torch

def kl_gaussian(mu1, sigma1, mu2, sigma2, eps: float = 1e-12):
    """
    KL( N(mu1, sigma1^2) || N(mu2, sigma2^2) )
    모든 텐서는 브로드캐스팅 가능. 반환값은 입력과 동일한 브로드캐스팅 shape.

    Args:
        mu1, sigma1, mu2, sigma2: 텐서 (예: [..., C])
        eps: 수치 안정성용 작은 값
    Returns:
        kl: 텐서 (mu/sigma와 동일한 브로드캐스팅 shape)
    """
    var1 = (sigma1 ** 2).clamp_min(eps)
    var2 = (sigma2 ** 2).clamp_min(eps)
    diff2 = (mu1 - mu2) ** 2
    return 0.5 * (torch.log(var2) - torch.log(var1) + (var1 + diff2) / var2 - 1.0)


def kl_matrix_loss_weighted(K, w_diag=1.0, w_off=1.0, target_off=3.0):
    """
    diag는 0으로, off는 target_off로. 각 항에 가중치 부여 가능.
    """
    C = K.size(0)
    eye = torch.eye(C, device=K.device, dtype=K.dtype).bool()
    diag_loss = (K.diagonal() ** 2).mean()                    # (K[ii]-0)^2
    off_loss  = ((K[~eye] - target_off) ** 2).mean()          # (K[ij]-1)^2, i!=j
    return w_diag * diag_loss + w_off * off_loss


def cross_corr_mu_loss(m1v, m2v, eps=1e-12, lambda_off=1.0):
    """
    μ 기반 cross-correlation loss
    Barlow Twins 구조(대각=1, 비대각=0)
    """
    # [B,C] 입력 가정
    B = m1v.size(0)
    # 배치 정규화
    z1n = (m1v - m1v.mean(0)) / (m1v.std(0) + eps)
    z2n = (m2v - m2v.mean(0)) / (m2v.std(0) + eps)

    # C matrix
    C = (z1n.T @ z2n) / float(B)  # [C,C]

    # 대각/비대각 분리
    D = C.size(0)

    diag = torch.diag(C)
    off  = C - torch.diag(diag)

    diag_loss = (1.0 - diag).pow(2).mean()                    # diag → 1
    off_loss  = off.pow(2).sum() / (D * (D - 1) + eps)         # off → 0

    return diag_loss + lambda_off * off_loss, C







