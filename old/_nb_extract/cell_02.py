import matplotlib.pyplot as plt
import torch

# DataLoader에서 한 배치 꺼내기
imgs, masks, lefts, rights = next(iter(train_loader))

# 첫 번째 배치의 첫 번째 샘플만 확인
i = 0
orig  = imgs[i]
left  = lefts[i]
right = rights[i]

# 텐서 → numpy
def to_np(x):
    return x.squeeze().detach().cpu().numpy()

plt.figure(figsize=(12,4))
for j, (im, title) in enumerate([(orig, "Main Aug"), (left, "Left Aug"), (right, "Right Aug")]):
    plt.subplot(1,3,j+1)
    plt.imshow(to_np(im), cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
plt.show()
