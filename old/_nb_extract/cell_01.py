batch_size = 64
from torch.utils.data import DataLoader

# 스플릿 별 루트 (각 폴더 바로 아래에 original/ 과 label/ 이 있어야 함)
train_root = "datasets/bp_data/train_split/train"
val_root   = "datasets/bp_data/train_split/val"
test_root  = "datasets/bp_data/train_split/test"



train_set = CovidDataset(train_root)
val_set   = CovidDataset(val_root)
test_set  = CovidDataset(test_root)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

print(len(train_set), len(val_set), len(test_set))
