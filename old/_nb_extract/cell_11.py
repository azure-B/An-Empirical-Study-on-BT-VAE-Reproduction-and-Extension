import torch
import torch.optim as optim
from itertools import chain
import torch.nn as nn  # ← 추가

# 1) VAE 모델 초기화
vae = VAE(base=32).to(device)  # 그대로 유지

# 2) Encoder 가중치 로드
ckpt_path = "bt_pretrain.pth"
ckpt = torch.load(ckpt_path, map_location=device)

enc_state = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt

def _strip_module_prefix(state):
    return { (k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items() }

enc_state = _strip_module_prefix(enc_state)

# 로드는 기존 encoder 모듈에 먼저 진행
missing, unexpected = vae.encoder.load_state_dict(enc_state, strict=False)
print(f"[encoder load] missing={len(missing)} unexpected={len(unexpected)}")

# === (수정) 입력 어댑터: in_ch → exp_in 을 '필요할 때만' prepend ===
def _first_conv_in_channels(m: nn.Module) -> int:
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            return mod.in_channels
    raise RuntimeError("No Conv2d found in encoder")

# 인코더가 기대하는 입력 채널 (보통 1)
exp_in = _first_conv_in_channels(vae.encoder)

# 실제 데이터의 입력 채널 추정 (train_loader에서 한 배치)
try:
    _sample_batch = next(iter(train_loader))
    _flair = _sample_batch[0]  # (flair, seg, left, right) 가정
    if _flair.ndim == 3:          # [B,H,W]
        in_ch = 1
    elif _flair.ndim == 4:        # [B,C,H,W]
        in_ch = _flair.shape[1]
    else:
        raise ValueError(f"Unexpected flair shape: {tuple(_flair.shape)}")
except Exception as e:
    print(f"[warn] fail to infer in_ch from train_loader: {e}")
    in_ch = exp_in  # 추정 실패 시 어댑터 생략

# 필요할 때만 어댑터 추가
if in_ch != exp_in:
    input_adapter = nn.Conv2d(in_channels=in_ch, out_channels=exp_in, kernel_size=1, bias=False).to(device)
    nn.init.kaiming_normal_(input_adapter.weight, nonlinearity='relu')
    vae.encoder = nn.Sequential(input_adapter, vae.encoder).to(device)
    print(f"[adapter] prepend 1x1 Conv: {in_ch} -> {exp_in}")
else:
    print(f"[adapter] skipped (in_ch == exp_in == {exp_in})")

# 3) Optimizer / Scheduler (변경 없음, 단 어댑터 파라미터가 포함되도록 '붙인 후' 생성)
optimizers = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizers,
    mode='min',
    factor=0.95,
    patience=3,
    verbose=True,
    min_lr=1e-8
)
