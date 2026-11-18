"""
Lazy drop-in for BT-VAE_CODE/data_load_BT.CovidDataset.

원본은 __init__에서 전 슬라이스를 RAM에 올려 VRAM/RAM이 터진다.
여기서는 경로만 보관하고 __getitem__ 때 디스크에서 읽는다.
반환 형식은 동일: img[H,W], mask[H,W], left[1,H,W], right[1,H,W]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

# pipeline.datasets REGISTRY와 동일 키 (노트북 편의)
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "datasets"
REGISTRY = {
    "bp_train_split": "bp_data/train_split",
    "bp_train_30": "bp_data/train_30_split",
    "covid": "covid_8_1_1",
    "covid_90p": "covid_8_1_1_90p",
}


class CovidDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.img_size = tuple(img_size)
        self.path_original = self.data_dir / "original"
        self.path_label = self.data_dir / "label"

        if not self.path_original.is_dir():
            raise FileNotFoundError(f"original/ 없음: {self.path_original}")
        if not self.path_label.is_dir():
            raise FileNotFoundError(f"label/ 없음: {self.path_label}")

        names = sorted(
            p.name
            for p in self.path_original.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
        self.filenames = [n for n in names if (self.path_label / n).exists()]
        if not self.filenames:
            raise RuntimeError(f"이미지/마스크 쌍이 없음: {self.data_dir}")

        # 원본 API 호환용 빈 리스트 (eager 코드가 길이를 참조하는 경우 대비)
        self.image_list = self.filenames
        self.label_list = self.filenames
        self.image_list_left = self.filenames
        self.image_list_right = self.filenames

    def __len__(self):
        return len(self.filenames)

    def _load_gray(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L").resize(self.img_size, Image.BILINEAR)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return t / 255.0

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L").resize(self.img_size, Image.NEAREST)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return (t > 0).float()

    def __getitem__(self, idx):
        n = len(self.filenames)
        last_err = None
        for offset in range(n):
            j = (idx + offset) % n
            name = self.filenames[j]
            try:
                image = self._load_gray(self.path_original / name)
                label = self._load_mask(self.path_label / name)
            except (OSError, ValueError, SyntaxError) as e:
                last_err = e
                continue
            left = TF.rotate(image.unsqueeze(0), 90)
            right = TF.rotate(image.unsqueeze(0), 270)
            return image, label, left, right
        raise OSError(f"all images failed near idx={idx}: {last_err}")
