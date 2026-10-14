# 학습 설정
total_epochs = 400
train_losses = []
dev_losses = []
best_loss = float('inf')

# 결과 저장 디렉토리 설정
results_dir = 'results/result_BP'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

try:
    for epoch in range(1, total_epochs + 1):
        start_time = timer()
        
        # 학습 단계
        train_loss = train_epoch(vae, optimizers, train_loader, epoch, device)
        train_losses.append(train_loss)
        
        # 평가 단계
        dev_loss = evaluate(vae, val_loader, epoch, results_dir, scheduler, device)
        dev_losses.append(dev_loss)
        
        scheduler.step(dev_loss)
        
        # 최고 성능 모델 저장
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(
                vae.state_dict(),
                join(results_dir, 'VAE_model_best_BP_30.pt')
            )
            print(f"새로운 최고 성능 모델 저장됨. Loss: {best_loss:.4f}")
        
        # 에폭 통계 출력
        end_time = timer()
        print(
            f"에폭 {epoch}: "
            f"학습 손실: {train_loss:.4f}, "
            f"검증 손실: {dev_loss:.4f}, "
            f"에폭 소요 시간 = {(end_time - start_time):.3f}초"
        )
    
    print('학습이 완료되었습니다.')

except KeyboardInterrupt:
    print('사용자에 의해 학습이 중단되었습니다.')