printed = False

for bidx, (flair, seg, left, right) in enumerate(train_loader):  # ← 반환되는 4개 다 받음
    flair = flair.to(device)
    
    seg   = seg.to(device)
    # left, right는 필요 없다면 그냥 무시하거나 아래처럼:
    # _, _, left, right = batch  로도 가능
    flair = flair.unsqueeze(1)

    print(flair.shape)

    # 1) 단계별로 쪼개서 실행
    x = flair  # [B,1,256,256]

    # initial
    x = vae.encoder.encoder_initial_flair(x)
    print("[after initial]", tuple(x.shape))  # 기대: (B, 32, 256, 256)  ← base=32 가정

    # features를 구성요소별로 추출
    features = vae.encoder.encoder_features_flair
    for i, m in enumerate(features):
        x = m(x)
        print(f"[after features[{i}] {m.__class__.__name__}]", tuple(x.shape))

    out, mu, logvar = vae(flair)

    if not printed:
        print(f"[DEBUG] out:    {tuple(out.shape)}")
        print(f"[DEBUG] mu:     {tuple(mu.shape)}")
        print(f"[DEBUG] logvar: {tuple(logvar.shape)}")
        printed = True
