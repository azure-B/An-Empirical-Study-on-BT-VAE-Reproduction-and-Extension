
# === PyTorch Core ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === Optimizers & LR scheduling ===
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
)
# Keras의 regularizers.l2 -> PyTorch는 보통 optimizer의 weight_decay로 처리
# 예: AdamW(params, lr=1e-3, weight_decay=1e-4)


import torchvision
from torchvision import transforms
from torchvision import models  
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, JaccardIndex
)
from torchmetrics.segmentation import DiceScore as Dice


import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.morphology import label
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime
import sys
from contextlib import nullcontext
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# from data_load_BT import CovidImageOnlyDataset, CovidDataset
from data_load_BT import  CovidDataset
from models1 import Encoder, Decoder, VAE
