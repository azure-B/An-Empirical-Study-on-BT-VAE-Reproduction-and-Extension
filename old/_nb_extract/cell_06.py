# # ===== C(교차상관) 계산 & 시각화 유틸 =====
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

# @torch.no_grad()
# def compute_C_mu(encoder, batch):
#     """
#     batch에서 view1, view2를 꺼내 encoder 통과 → μ만으로 C 계산.
#     (네 데이터 구조가 batch[2], batch[3]에 두 view라고 가정)
#     반환: C [C,C]
#     """
#     dev = next(encoder.parameters()).device
#     encoder.eval()

#     v1, v2 = batch[2], batch[3]
#     if v1.ndim == 3: v1 = v1.unsqueeze(1)
#     if v2.ndim == 3: v2 = v2.unsqueeze(1)
#     v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)

#     m1, logv1, z1 = encoder(v1)  # [B,C,H,W] (H=W=8 같은 값)
#     m2, logv2, z2 = encoder(v2)

#     # 네가 쓰는 GAP 경로 그대로
#     s = lambda x: F.adaptive_avg_pool2d(x, 1).flatten(1) if x.dim()==4 else x
#     x1, x2 = s(m1), s(m2)       # [B,C]

#     eps = 1e-4
#     x1n = (x1 - x1.mean(0)) / (x1.std(0) + eps)
#     x2n = (x2 - x2.mean(0)) / (x2.std(0) + eps)
#     B   = x1n.size(0)
#     C   = (x1n.transpose(0,1) @ x2n) / float(B)  # [C,C]
#     return C

# def show_C_heatmap(C, title="Cross-Correlation C (mu)"):
#     C_np = C.detach().float().cpu().numpy()
#     plt.figure(figsize=(6,5))
#     im = plt.imshow(C_np, vmin=-1.0, vmax=1.0, origin="upper", interpolation="nearest")
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

#     # 통계 출력(대각 ~1, 비대각 ~0 확인용)
#     D = C_np.shape[0]
#     diag = np.diag(C_np)
#     off  = C_np - np.diag(diag)
#     off_vals = off[np.triu_indices(D, k=1)]
#     print(
#         f"[C stats] diag: mean={diag.mean():.4f}, std={diag.std():.4f} | "
#         f"off: mean={off_vals.mean():.4f}, rms={np.sqrt((off_vals**2).mean()):.4f}, "
#         f"max|off|={np.abs(off_vals).max():.4f}"
#     )


# batch = next(iter(train_loader))
# C = compute_C_mu(encoder, batch)
# show_C_heatmap(C, title="Cross-Correlation C (mu)")


# # ====== viz_utils.py에 추가(또는 별도 셀) ======
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# # (이미 있다면 중복 정의 불필요)
# def _model_device(m: torch.nn.Module):
#     for p in m.parameters(): return p.device
#     for b in m.buffers():    return b.device
#     return torch.device("cpu")

# @torch.no_grad()
# def _prep_views_from_batch(batch, device):
#     v1, v2 = batch[2], batch[3]   # 너의 데이터 포맷 전제
#     if v1.ndim == 3: v1 = v1.unsqueeze(1)
#     if v2.ndim == 3: v2 = v2.unsqueeze(1)
#     return v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

# # --- KL 대칭행렬 계산 ---
# @torch.no_grad()
# def compute_K_once(encoder, batch, symmetric=True, reduce="mean"):
#     """
#     반환: K [C,C] (reduce='mean' 또는 'sum'), 혹은 [B,C,C] (reduce='none')
#     """
#     dev = _model_device(encoder)
#     encoder.eval()
#     v1, v2 = _prep_views_from_batch(batch, dev)

#     m1, logv1, z1 = encoder(v1)     # [B,C,H,W]
#     m2, logv2, z2 = encoder(v2)

#     _gap = nn.AdaptiveAvgPool2d(1)
#     s = lambda x: _gap(x).flatten(1) if x.dim()==4 else x

#     m1v, m2v = s(m1), s(m2)         # [B,C]
#     lv1, lv2 = s(logv1), s(logv2)   # [B,C]
#     sig1 = (0.5 * lv1).exp()
#     sig2 = (0.5 * lv2).exp()

#     def kl_gaussian(mu1, sigma1, mu2, sigma2, eps=1e-12):
#         v1 = (sigma1**2).clamp_min(eps)
#         v2 = (sigma2**2).clamp_min(eps)
#         d2 = (mu1 - mu2) ** 2
#         return 0.5 * (torch.log(v2) - torch.log(v1) + (v1 + d2) / v2 - 1.0)

#     # [B,C,C]
#     K12_b = kl_gaussian(m1v.unsqueeze(2), sig1.unsqueeze(2),
#                         m2v.unsqueeze(1), sig2.unsqueeze(1))
#     if symmetric:
#         K21_b = kl_gaussian(m2v.unsqueeze(2), sig2.unsqueeze(2),
#                             m1v.unsqueeze(1), sig1.unsqueeze(1))
#         K_b = 0.5 * (K12_b + K21_b)
#     else:
#         K_b = K12_b

#     if reduce == "mean":
#         return K_b.mean(dim=0)      # [C,C]
#     elif reduce == "sum":
#         return K_b.sum(dim=0)       # [C,C]
#     else:
#         return K_b                   # [B,C,C]

# # --- 히트맵 표시 ---
# def show_K_heatmap(K, title="Symmetric KL (mean)", vmax_pct=99, vmin=0.0):
#     K_np = K.detach().float().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)
#     vmax = float(np.percentile(K_np, vmax_pct))
#     plt.figure(figsize=(6,5))
#     im = plt.imshow(K_np, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title(f"{title} | vmax@p{vmax_pct}={vmax:.3f}")
#     plt.tight_layout()
#     plt.show()

#     D = K_np.shape[0]
#     diag = np.diag(K_np)
#     off  = K_np[~np.eye(D, dtype=bool)]
#     print(f"[K stats] diag mean={diag.mean():.4f}, std={diag.std():.4f} | "
#           f"off mean={off.mean():.4f}, p90={np.percentile(off,90):.4f}, "
#           f"p99={np.percentile(off,99):.4f}, max={off.max():.4f}")
    
# batch = next(iter(train_loader))
# K = compute_K_once(encoder, batch, symmetric=True, reduce="mean")
# show_K_heatmap(K, title="Symmetric KL (mean)")


