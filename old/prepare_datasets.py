"""
프로젝트 내 datasets/ 로 전처리·배치.
원문 로직은 그대로 두고, 노트북의 데이터 경로만 여기로 맞춘다.

  datasets/
    raw/                          # 다운로드 원본
    bp/  covid/                   # staging PNG
    bp_data/train_split/...
    bp_data/train_30_split/...    # train 30%
    covid_8_1_1/...
    covid_8_1_1_90p/...           # train 90%

  python prepare_datasets.py
  python prepare_datasets.py --layout-only
"""

from __future__ import annotations

import argparse
import io
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets" / "raw"
STAGING_BP = ROOT / "datasets" / "bp"
STAGING_COVID = ROOT / "datasets" / "covid"
DATA_ROOT = ROOT / "datasets"

SEED = 42
SPLIT_811 = (0.8, 0.1, 0.1)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_gray_png(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        amin, amax = float(np.min(arr)), float(np.max(arr))
        if amax > amin:
            arr = ((arr - amin) / (amax - amin) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)
    Image.fromarray(arr).convert("L").save(path)


def save_mask_png(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr)
    Image.fromarray(((arr > 0).astype(np.uint8) * 255)).convert("L").save(path)


def assign_splits(names: list[str], ratios=SPLIT_811, seed=SEED) -> dict[str, str]:
    rng = random.Random(seed)
    names = list(names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    mapping = {}
    for i, name in enumerate(names):
        if i < n_train:
            mapping[name] = "train"
        elif i < n_train + n_val:
            mapping[name] = "val"
        else:
            mapping[name] = "test"
    return mapping


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_split(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for split in ("train", "val", "test"):
        for kind in ("original", "label"):
            sdir = src_root / split / kind
            ddir = dst_root / split / kind
            ensure_dir(ddir)
            files = sorted(sdir.glob("*.png"))
            for f in files:
                link_or_copy(f, ddir / f.name)
            print(f"  {dst_root.relative_to(ROOT)}/{split}/{kind}: {len(files)}")


def make_label_fraction(
    full_split_root: Path,
    out_root: Path,
    train_frac: float,
    seed: int = SEED,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ("val", "test"):
        for kind in ("original", "label"):
            sdir = full_split_root / split / kind
            ddir = out_root / split / kind
            ensure_dir(ddir)
            for f in sorted(sdir.glob("*.png")):
                link_or_copy(f, ddir / f.name)

    train_orig = full_split_root / "train" / "original"
    names = sorted(p.stem for p in train_orig.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(names)
    keep = names[: max(1, int(len(names) * train_frac))]

    for kind in ("original", "label"):
        sdir = full_split_root / "train" / kind
        ddir = out_root / "train" / kind
        ensure_dir(ddir)
        for stem in keep:
            link_or_copy(sdir / f"{stem}.png", ddir / f"{stem}.png")

    print(
        f"  {out_root.relative_to(ROOT)}: train={len(keep)}/{len(names)} "
        f"({train_frac:.0%})"
    )


def prepare_bp_pngs() -> Path:
    zip_path = RAW / "Nerve-Ultrasound-Image-Segmentation.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    if (STAGING_BP / "train" / "original").exists() and any(
        (STAGING_BP / "train" / "original").glob("*.png")
    ):
        print(f"[BP] reuse {STAGING_BP}")
        return STAGING_BP

    if STAGING_BP.exists():
        shutil.rmtree(STAGING_BP)
    for split in ("train", "val", "test"):
        ensure_dir(STAGING_BP / split / "original")
        ensure_dir(STAGING_BP / split / "label")

    print(f"[BP] streaming from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        img_names = sorted(
            n
            for n in zf.namelist()
            if n.startswith("train/")
            and n.endswith(".tif")
            and "_mask" not in n
        )
        split_map = assign_splits([Path(n).stem for n in img_names])
        print(f"[BP] converting {len(img_names)} pairs ...")
        for i, name in enumerate(img_names, 1):
            stem = Path(name).stem
            split = split_map[stem]
            with zf.open(name) as f:
                img = np.array(Image.open(io.BytesIO(f.read())).convert("L"))
            with zf.open(f"train/{stem}_mask.tif") as f:
                msk = np.array(Image.open(io.BytesIO(f.read())).convert("L"))
            save_gray_png(img, STAGING_BP / split / "original" / f"{stem}.png")
            save_mask_png(msk, STAGING_BP / split / "label" / f"{stem}.png")
            if i % 500 == 0 or i == len(img_names):
                print(f"  {i}/{len(img_names)}")
    return STAGING_BP


def _load_nifti(path: Path):
    import nibabel as nib

    return nib.load(str(path)).get_fdata()


def _iter_slices(vol: np.ndarray, msk: np.ndarray):
    if vol.ndim != 3:
        raise ValueError(f"expected 3D, got {vol.shape}")
    if vol.shape[-1] <= max(vol.shape[0], vol.shape[1]):
        for i in range(vol.shape[-1]):
            yield i, vol[..., i], msk[..., i]
    else:
        for i in range(vol.shape[0]):
            yield i, vol[i], msk[i]


def prepare_covid_pngs() -> Path:
    try:
        import nibabel  # noqa: F401
    except ImportError:
        os.system("pip install nibabel -q")

    if (STAGING_COVID / "train" / "original").exists() and any(
        (STAGING_COVID / "train" / "original").glob("*.png")
    ):
        print(f"[COVID] reuse {STAGING_COVID}")
        return STAGING_COVID

    if STAGING_COVID.exists():
        shutil.rmtree(STAGING_COVID)
    for split in ("train", "val", "test"):
        ensure_dir(STAGING_COVID / split / "original")
        ensure_dir(STAGING_COVID / split / "label")

    items: list[tuple[str, np.ndarray, np.ndarray]] = []
    im1, mk1 = RAW / "medseg1_tr_im.nii.gz", RAW / "medseg1_tr_mask.nii.gz"
    if im1.exists() and mk1.exists():
        print("[COVID] MedSeg Dataset 1 ...")
        vol, msk = _load_nifti(im1), _load_nifti(mk1)
        for i, sl, ml in _iter_slices(vol, msk):
            items.append((f"medseg1_{i:03d}", sl, ml))

    im2_zip, mk2_zip = RAW / "medseg2_rp_im.zip", RAW / "medseg2_rp_msk.zip"
    if im2_zip.exists() and mk2_zip.exists():
        im2_dir, mk2_dir = RAW / "medseg2_im", RAW / "medseg2_msk"
        for d, zpath in ((im2_dir, im2_zip), (mk2_dir, mk2_zip)):
            if d.exists():
                shutil.rmtree(d)
            ensure_dir(d)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(d)
        print("[COVID] MedSeg Dataset 2 ...")
        for im_path in sorted(im2_dir.rglob("*.nii*")):
            candidates = list(mk2_dir.rglob(im_path.name))
            if not candidates:
                base = im_path.name.replace(".nii.gz", "").replace(".nii", "")
                candidates = [
                    p
                    for p in mk2_dir.rglob("*.nii*")
                    if base in p.name
                    or p.name.replace(".nii.gz", "").replace(".nii", "") in base
                ]
            if not candidates:
                print(f"  warn: no mask for {im_path.name}")
                continue
            vol, msk = _load_nifti(im_path), _load_nifti(candidates[0])
            case = im_path.name.replace(".nii.gz", "").replace(".nii", "")
            before = len(items)
            for i, sl, ml in _iter_slices(vol, msk):
                items.append((f"{case}_{i:03d}", sl, ml))
            print(f"  {case}: +{len(items) - before}")

    if not items:
        raise RuntimeError("No COVID slices")

    split_map = assign_splits([n for n, _, _ in items])
    print(f"[COVID] writing {len(items)} slices ...")
    for name, img, msk in items:
        split = split_map[name]
        save_gray_png(img, STAGING_COVID / split / "original" / f"{name}.png")
        save_mask_png(msk, STAGING_COVID / split / "label" / f"{name}.png")
    return STAGING_COVID


def layout_project(bp_src: Path, covid_src: Path) -> None:
    print(f"\n[LAYOUT] under {DATA_ROOT}")
    bp_full = DATA_ROOT / "bp_data" / "train_split"
    bp_30 = DATA_ROOT / "bp_data" / "train_30_split"
    covid_full = DATA_ROOT / "covid_8_1_1"
    covid_90 = DATA_ROOT / "covid_8_1_1_90p"

    print("[LAYOUT] BP train_split")
    mirror_split(bp_src, bp_full)
    print("[LAYOUT] BP train_30_split")
    make_label_fraction(bp_full, bp_30, 0.30)
    print("[LAYOUT] COVID covid_8_1_1")
    mirror_split(covid_src, covid_full)
    print("[LAYOUT] COVID covid_8_1_1_90p")
    make_label_fraction(covid_full, covid_90, 0.90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    ensure_dir(RAW)

    if args.layout_only:
        bp_src, covid_src = STAGING_BP, STAGING_COVID
        if not any((bp_src / "train" / "original").glob("*.png")):
            raise FileNotFoundError(bp_src)
        if not any((covid_src / "train" / "original").glob("*.png")):
            raise FileNotFoundError(covid_src)
        print("[skip] conversion")
    else:
        bp_src = prepare_bp_pngs()
        covid_src = prepare_covid_pngs()

    layout_project(bp_src, covid_src)
    print("=== DONE ===")
    print(f"  {DATA_ROOT / 'bp_data'}")
    print(f"  {DATA_ROOT / 'covid_8_1_1'}")


if __name__ == "__main__":
    main()
