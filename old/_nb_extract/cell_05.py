# === 실행 ===
device = "cuda:2" if torch.cuda.is_available() else "cpu"
encoder = Encoder(base=32).to(device).train()   # 이미 만들어둔 Encoder 인스턴스라면 이 줄만


pretrain_bt(
    encoder=encoder,
    train_loader=train_loader,
    val_loader=val_loader,
    
    epochs=200,            # 테스트로 1~3 epoch 먼저
    lr=3e-4,
    weight_decay=1e-4,
    amp=False,           # 처음엔 False 권장 (수치 이슈 확인 후 True)
    grad_clip=1.0,
    ckpt_path="bt_pretrain_bp.pth",
    print_every=1,       # 매 이터 로그
    augment_on_gpu=True # get_two_views가 CPU 텐서 기준이면 False
)