"""
BT-VAE dataset loader (lazy).

기존 CovidDataset API 유지:
  img, mask, view_left, view_right = dataset[i]
  - img/mask: [H, W]
  - view_left/right: [1, H, W]  (90° / 270° 회전, Barlow Twins용)

사용 예:
  from data_load_BT import datasets

  data = datasets.bp_train_split(batch_size=64)
  # data.train / data.val / data.test
  # data.train_loader / data.val_loader / data.test_loader
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"

# 노트북에서 쓰는 이름 → 상대 경로
REGISTRY = {
    "bp_train_split": "bp_data/train_split",
    "bp_train_30": "bp_data/train_30_split",
    "covid": "covid_8_1_1",
    "covid_90p": "covid_8_1_1_90p",
}


class CovidDataset(Dataset):
    """경로만 들고 있고, __getitem__ 때 디스크에서 읽음 (메모리 폭주 방지)."""

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

    def __len__(self):
        return len(self.filenames)

    def _load_gray(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L").resize(self.img_size, Image.BILINEAR)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return t / 255.0  # [H, W]

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("L").resize(self.img_size, Image.NEAREST)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        return (t > 0).float()  # [H, W]

    def __getitem__(self, idx):
        # Skip corrupt / truncated files (common on large BP dumps).
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


@dataclass
class DataBundle:
    name: str
    root: Path
    train: CovidDataset
    val: CovidDataset
    test: CovidDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    def __repr__(self) -> str:
        return (
            f"DataBundle(name={self.name!r}, root={str(self.root)!r}, "
            f"train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"
        )


def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def load_bundle(
    name: str,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    img_size=(256, 256),
) -> DataBundle:
    """REGISTRY 이름으로 train/val/test + DataLoader 묶음 반환."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}. available: {list(REGISTRY)}")

    root = DATA_ROOT / REGISTRY[name]
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train = CovidDataset(root / "train", img_size=img_size)
    val = CovidDataset(root / "val", img_size=img_size)
    test = CovidDataset(root / "test", img_size=img_size)

    return DataBundle(
        name=name,
        root=root,
        train=train,
        val=val,
        test=test,
        train_loader=_make_loader(
            train, batch_size, True, num_workers, pin_memory, drop_last=True
        ),
        val_loader=_make_loader(
            val, batch_size, False, num_workers, pin_memory, drop_last=False
        ),
        test_loader=_make_loader(
            test, batch_size, False, num_workers, pin_memory, drop_last=False
        ),
    )


class _DatasetsAPI:
    """일목요연한 접근: datasets.bp_train_split(), datasets.list() 등."""

    def list(self):
        rows = []
        for key, rel in REGISTRY.items():
            root = DATA_ROOT / rel
            ok = (root / "train" / "original").is_dir()
            rows.append((key, str(root), ok))
        return rows

    def __repr__(self) -> str:
        lines = ["BTDatasets:"]
        for key, path, ok in self.list():
            mark = "OK" if ok else "MISSING"
            lines.append(f"  [{mark}] {key:16s} -> {path}")
        return "\n".join(lines)

    def get(self, name: str, **kwargs) -> DataBundle:
        return load_bundle(name, **kwargs)

    def bp_train_split(self, **kwargs) -> DataBundle:
        return load_bundle("bp_train_split", **kwargs)

    def bp_train_30(self, **kwargs) -> DataBundle:
        return load_bundle("bp_train_30", **kwargs)

    def covid(self, **kwargs) -> DataBundle:
        return load_bundle("covid", **kwargs)

    def covid_90p(self, **kwargs) -> DataBundle:
        return load_bundle("covid_90p", **kwargs)


datasets = _DatasetsAPI()
