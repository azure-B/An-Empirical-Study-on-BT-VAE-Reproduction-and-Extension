import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import time


def pretrain_bt(
    encoder: nn.Module,
    train_loader,
    epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    ckpt_path: str | None = "bt_pretrain_bp.pth",
    print_every: int = 50,
    augment_on_gpu: bool = True,
    val_loader=None,   
):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    encoder.to(device).train()
    conv1x1.to(device)

    # BN freeze
    for m in encoder.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    if len(train_loader) == 0:
        raise RuntimeError("train_loader 길이가 0입니다. Dataset/DataLoader 구성을 확인하세요.")

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(conv1x1.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.startswith("cuda")))

    def _progress_line(ep, epochs, it, total, loss, start_time, bar_len=30):
        ratio = it / max(total, 1)
        filled = max(0, min(int(ratio * bar_len), bar_len))
        bar = "█" * filled + "·" * (bar_len - filled)
        elapsed = time.time() - start_time
        ips = it / elapsed if elapsed > 0 else 0.0     # iters/sec
        eta = (total - it) / ips if ips > 0 else 0.0
        print(
            f"\r[ep {ep:3d}/{epochs:3d}] [{bar}] {it:4d}/{total:4d} "
            f"loss={loss:8.5f}  {elapsed:6.1f}s  ETA {eta:5.1f}s",
            end="", flush=True
        )
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        running = 0.0
        epoch_start = time.time()
        encoder.train()
        for it, batch in enumerate(train_loader, 1):
            if isinstance(batch, (list, tuple)):
                view1 = batch[2]   
                view2 = batch[3]   
            else:
                raise ValueError("batch가 tuple이 아닙니다.")
            view1 = view1.to(device, non_blocking=True) #32 1 256 256 
            view2 = view2.to(device, non_blocking=True) #32 1 256 256 

           
            opt.zero_grad(set_to_none=True)
            use_amp = scaler.is_enabled()
            cm = torch.cuda.amp.autocast() if use_amp else nullcontext()

            with cm:
                if view1.ndim == 3: view1 = view1.unsqueeze(1)
                if view2.ndim == 3: view2 = view2.unsqueeze(1)
          

                assert view1.shape[1] == 1 and view2.shape[1] == 1, f"{view1.shape=}, {view2.shape=}"

                m1, v1_, z1 = encoder(view1) #각 32 1024 1 1 에서 -> 32 1024 
                m2, v2_, z2 = encoder(view2)

                v1_ = v1_.clamp(-5, 2); v2_ = v2_.clamp(-5, 2)

                
                # print("m1: ", m1.shape) (b,base^2,8,8)
                
                _gap = nn.AdaptiveAvgPool2d(1)
                s = lambda x: _gap(x).flatten(1) if x.dim() == 4 else x
                m1v, v1v, z1v = s(m1), s(v1_), s(z1)  # 32 1024 (base^2)
                m2v, v2v, z2v = s(m2), s(v2_), s(z2)  # 32 1024
              
                eps = 1e-12
                # m1v, v1v, m2v, v2v: [B, C]  (v*는 logvar)
                sig1 = (0.5 * v1v).exp()   # [B, C]
                sig2 = (0.5 * v2v).exp()   # [B, C]

                    # ➜ per-sample KL 행렬 만들고 배치 평균 내기
                K12_b = kl_gaussian(
                    mu1    = m1v.unsqueeze(2),   # [B,C,1]
                    sigma1 = sig1.unsqueeze(2),  # [B,C,1]
                    mu2    = m2v.unsqueeze(1),   # [B,1,C]
                    sigma2 = sig2.unsqueeze(1),  # [B,1,C]
                )                                # -> [B,C,C]

                K21_b = kl_gaussian(
                        mu1    = m2v.unsqueeze(2),
                        sigma1 = sig2.unsqueeze(2),
                        mu2    = m1v.unsqueeze(1),
                        sigma2 = sig1.unsqueeze(1),
                    )

                K = 0.5 * (K12_b + K21_b)        # [B,C,C]
                K = K.mean(dim=0)

                tgt = 1.0 + min(1.0, ep / 100.0)  # ep 0→100 동안 1.0→2.0

                loss_kl = kl_matrix_loss_weighted(K, w_diag=1.0, w_off=1.0, target_off=tgt)

                loss_cmu, C_mu = cross_corr_mu_loss(m1v, m2v, lambda_off=1.0)  #c_mu는 학습용 X,. 나중에 시각화용용
                lambda_cmu = 0.2  # 0.05 ~ 0.2 권장

                    # ③ 최종 Loss
                loss = loss_kl + lambda_cmu * loss_cmu

                
        
                                

                if not torch.isfinite(loss).item():
                    raise ValueError("Final loss is NaN/Inf")

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

            else:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                opt.step()

            # if it == 1:
            #     with torch.no_grad():
            #         tot = 0.0
            #         for p in encoder.parameters():
            #             if p.grad is not None:
            #                 tot += p.grad.detach().pow(2).sum().item()
                    

            running += loss.item()
            _progress_line(ep, epochs, it, len(train_loader), float(loss.item()), epoch_start)
        print()
        epoch_sec = time.time() - epoch_start
        avg = running / max(1, len(train_loader))
        print(f"=> epoch {ep}  time={epoch_sec:.2f}s  train_avg_loss={avg:.6f}")

        # ======== ★ 검증 루프 (val_loss 기준으로 저장) ========
        if val_loader is not None and len(val_loader) > 0:
            encoder.eval()
            val_running = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    if isinstance(vbatch, (list, tuple)):
                        v1 = vbatch[2]
                        v2 = vbatch[3]
                    else:
                        raise ValueError("val batch가 tuple이 아닙니다.")
                    v1 = v1.to(device, non_blocking=True)
                    v2 = v2.to(device, non_blocking=True)
                   
                   

                    m1, v1_, z1 = encoder(v1)
                    m2, v2_, z2 = encoder(v2)
                    

                    v1_ = v1_.clamp(-5, 2); v2_ = v2_.clamp(-5, 2)
                    
                    _gap = nn.AdaptiveAvgPool2d(1)
                    s = lambda x: _gap(x).flatten(1) if x.dim() == 4 else x
                    m1v, v1v, z1v = s(m1), s(v1_), s(z1) # (B,C,1,1) → (B,C)
                    m2v, v2v, z2v = s(m2), s(v2_), s(z2) # (B,C,1,1) → (B,C)
                    eps = 1e-12
                    # m1v, v1v, m2v, v2v: [B, C]  (v*는 logvar)
                   

                    # 쌍방 KL → 대칭 KL 행렬 K: [C, C]
                    sig1 = (0.5 * v1v).exp()   # [B, C]
                    sig2 = (0.5 * v2v).exp()   # [B, C]

                    # ➜ per-sample KL 행렬 만들고 배치 평균 내기
                    K12_b = kl_gaussian(
                        mu1    = m1v.unsqueeze(2),   # [B,C,1]
                        sigma1 = sig1.unsqueeze(2),  # [B,C,1]
                        mu2    = m2v.unsqueeze(1),   # [B,1,C]
                        sigma2 = sig2.unsqueeze(1),  # [B,1,C]
                    )                                # -> [B,C,C]

                    K21_b = kl_gaussian(
                        mu1    = m2v.unsqueeze(2),
                        sigma1 = sig2.unsqueeze(2),
                        mu2    = m1v.unsqueeze(1),
                        sigma2 = sig1.unsqueeze(1),
                    )

                    K = 0.5 * (K12_b + K21_b)        # [B,C,C]
                    K = K.mean(dim=0)

                    tgt = 1.0 + min(1.0, ep / 100.0)  # ep 0→100 동안 1.0→2.0

                    loss_kl = kl_matrix_loss_weighted(K, w_diag=1.0, w_off=1.0, target_off=tgt)

                    loss_cmu, C_mu = cross_corr_mu_loss(m1v, m2v, lambda_off=1.0)  #c_mu는 학습용 X,. 나중에 시각화용용
                    lambda_cmu = 0.2  # 0.05 ~ 0.2 권장

                    # ③ 최종 Loss
                    loss = loss_kl + lambda_cmu * loss_cmu
                   
                    val_running += loss.item()
                    val_avg = val_running / max(1, len(val_loader))
                print(f"=> epoch {ep} val_loss={val_avg:.6f}")

            if ckpt_path and val_avg < best_val:
                best_val = val_avg
                torch.save({"ep": ep, "encoder": encoder.state_dict()}, ckpt_path)
                print(f"  ↳ saved BEST (by val_loss) to {ckpt_path} (val {best_val:.6f})")
       