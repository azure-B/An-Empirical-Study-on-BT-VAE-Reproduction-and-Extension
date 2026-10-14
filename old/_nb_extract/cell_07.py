
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
from torch import Tensor
import os
# For displaying and evaluating results.
import numpy as np
from matplotlib import pyplot as plt

# For monitoring resource-usage and progress.
from timeit import default_timer as timer
import psutil
from os.path import join

# GPU 번호 지정 (0번 GPU 사용)

# ===== Paths: 너의 데이터 구조에 맞게 수정 =====
# 데이터셋 루트 (train/val/test 및 각 original/label 이 안에 존재)
root_dir = 'datasets/bp_data/train_30_split'

# 기존 코드에서 raw_data_dir이 '원본 데이터 상위 폴더' 개념이라면, 
# 지금은 이미 train/val/test 구조가 준비되어 있으니 루트 자체를 가리키게 함
raw_data_dir = root_dir

# 전처리/로그/모델 저장 디렉토리들은 같은 루트 밑에 만들어 두자
prep_data_dir    = join(root_dir, 'prep_data')
losses_dir       = join(root_dir, 'losses')
models_dir       = join(root_dir, 'saved_models')
checkpoint_dir   = join(root_dir, 'checkpoints')

# (선택) DataLoader가 명시적인 이미지/마스크 경로를 요구한다면 아래도 활용 가능
train_img_dir = join(root_dir, 'train', 'original')
train_mask_dir = join(root_dir, 'train', 'label')
val_img_dir   = join(root_dir, 'val', 'original')
val_mask_dir  = join(root_dir, 'val', 'label')
test_img_dir  = join(root_dir, 'test', 'original')
test_mask_dir = join(root_dir, 'test', 'label')

# 디렉토리 보장
os.makedirs(prep_data_dir,  exist_ok=True)
os.makedirs(losses_dir,     exist_ok=True)
os.makedirs(models_dir,     exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
