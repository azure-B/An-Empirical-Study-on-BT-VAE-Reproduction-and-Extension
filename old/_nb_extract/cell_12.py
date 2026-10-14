import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_dice_loss_sigmoid(pred, target, smooth=1e-6):
    """
    Binary Dice Loss 계산 (시그모이드 기반)
    pred: [B, 1, H, W] - 로짓 값 (Sigmoid 미적용)
    target: [B, 1, H, W] - 단일 채널 마스크 (0: 배경, 1: 전경)
    """
    # Sigmoid 적용
    pred = torch.sigmoid(pred)  # [B, 1, H, W]
    
    # Dice 계산 (채널 차원을 유지하면서 계산)
    pred_flat = pred.reshape(pred.size(0), -1)  # [B, H*W]
    target_flat = target.reshape(target.size(0), -1)  # [B, H*W]

    intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]
    
    dice = (2. * intersection + smooth) / (union + smooth)  # [B]
    loss = 1 - dice.mean()  # scalar
    
    return loss

def combined_binary_loss_sigmoid(pred, target, dice_weight=0.5, ce_weight=0.5):
    """
    이진 분류를 위한 결합 손실 (시그모이드 기반)
    pred: [B, 1, H, W] - logits
    target: [B, 1, H, W] - 단일 채널 마스크 (0: 배경, 1: 전경)
    """
   
    # Dice Loss 계산
    dice = binary_dice_loss_sigmoid(pred, target)
    
    # BCE Loss 계산 (CrossEntropy 대신)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # pred와 target을 같은 형태로 reshape
    pred_reshaped = pred.view(pred.size(0), -1)  # [B, H*W]
    target_reshaped = target.view(target.size(0), -1)  # [B, H*W]
    ce = bce_loss(pred_reshaped, target_reshaped)

    # 전체 loss 계산
    total_loss = dice_weight * dice + ce_weight * ce
    
    return total_loss, dice.item(), ce.item()

def train_epoch(seg_model, optimizer, train_loader, epoch, device):
    seg_model.train()
    total_loss = 0
    total_recon = 0
    total_kl_loc = 0
    
    beta_local = 0.01
    recon_weight = 0.99

    for batch in train_loader:
        flair, seg ,_,_= batch
        flair = flair.to(device)
        seg = seg.to(device)
        flair = flair.unsqueeze(1)
        seg = seg.unsqueeze(1)
        
        
        # gradient 초기화
        optimizer.zero_grad()
        
        # Forward pass
        output, mu_local, logvar_local = seg_model(flair)

        # Reconstruction loss (dice loss) - 튜플 반환값 처리
        dice_loss_combined, dice_value, ce_value = combined_binary_loss_sigmoid(output, seg)
        
        # KL loss
        kl_loss_local = 1 + logvar_local - torch.square(mu_local) - torch.exp(logvar_local)
        kl_loss_local = torch.sum(kl_loss_local, dim=-1)
        kl_loss_local *= -0.5

        # Calculate means
        loss_k_loc = torch.mean(kl_loss_local)
        loss_d = dice_loss_combined  # 이미 평균 계산된 값

        # Total loss - 각 배치별로 계산
        loss = recon_weight * dice_loss_combined + beta_local * torch.mean(kl_loss_local)

        # Accumulate losses
        total_kl_loc += loss_k_loc.item()
        total_recon += loss_d.item()
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    print('train_total_local_kl: ', total_kl_loc/len(train_loader))
    print('train_total_dice: ', total_recon/len(train_loader))
    
    return total_loss / len(train_loader)

def evaluate(seg_model, dev_loader, epoch, save_dir, scheduler, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_model.eval()
    total_loss = 0
    total_recon = 0
    total_kl_loc = 0
    beta_local = 0.01
    recon_weight = 0.99
    
    with torch.no_grad():
        for batch in dev_loader:
            flair, y, _, _ = batch
            flair = flair.to(device)
            flair = flair.unsqueeze(1)
            y = y.to(device)
            y = y.unsqueeze(1)
            
            # Evaluation
            output, mu_local, logvar_local = seg_model(flair)
     
            # 평가 시에도 binary loss 사용 (일관성 유지)
            dice_loss_combined, dice_value, ce_value = combined_binary_loss_sigmoid(output, y)
            
            kl_loss_local = 1 + logvar_local - torch.square(mu_local) - torch.exp(logvar_local)
            kl_loss_local = torch.sum(kl_loss_local, dim=-1)
            kl_loss_local *= -0.5
        
            loss_k_loc = torch.mean(kl_loss_local)
            loss_d = dice_loss_combined
            
            loss = recon_weight * dice_loss_combined + beta_local * torch.mean(kl_loss_local)

            total_kl_loc += loss_k_loc.item()
            total_recon += loss_d.item()
            total_loss += loss.item()

    avg_loss = total_loss / len(dev_loader)
    print('dev_total_local_kl: ', total_kl_loc/len(dev_loader))
    print('dev_total_dice: ', total_recon/len(dev_loader))

    scheduler.step(avg_loss)
    return avg_loss



