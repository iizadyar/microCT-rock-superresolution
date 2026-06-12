from __future__ import annotations

from pathlib import Path
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import resolve_path
from .preprocessing import IMAGE_EXTENSIONS, list_images, load_grayscale, image_to_tensor, match_scale, bicubic_resize
from .augmentation import paired_augment


@dataclass
class ImagePair:
    lr_path: Path
    hr_path: Path
    stem: str


def find_pairs(lr_dir: str | Path, hr_dir: str | Path, extensions=IMAGE_EXTENSIONS) -> list[ImagePair]:
    lr_dir = resolve_path(lr_dir)
    hr_dir = resolve_path(hr_dir)
    lr_files = list_images(lr_dir, extensions)
    hr_files = list_images(hr_dir, extensions)
    hr_by_name = {p.name: p for p in hr_files}
    hr_by_stem = {p.stem: p for p in hr_files}
    pairs = []
    for lr in lr_files:
        hr = hr_by_name.get(lr.name) or hr_by_stem.get(lr.stem)
        if hr is not None:
            pairs.append(ImagePair(lr, hr, lr.stem))
    if not pairs:
        raise FileNotFoundError(f"No matching LR/HR image pairs found in {lr_dir} and {hr_dir}")
    return pairs


class PairedImageDataset(Dataset):
    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        scale: int,
        augment: bool = False,
        deterministic_augment: bool = False,
        pre_upscale: bool = False,
        crop_to_scale: bool = True,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rot90: bool = True,
    ):
        self.scale = int(scale)
        self.augment = bool(augment)
        self.deterministic_augment = bool(deterministic_augment)
        self.pre_upscale = bool(pre_upscale)
        self.crop_to_scale = bool(crop_to_scale)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rot90 = rot90
        self.pairs = find_pairs(lr_dir, hr_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        lr = image_to_tensor(load_grayscale(pair.lr_path))
        hr = image_to_tensor(load_grayscale(pair.hr_path))
        if self.crop_to_scale:
            lr, hr = match_scale(lr, hr, self.scale)
        if self.pre_upscale:
            lr = bicubic_resize(lr, size=hr.shape[-2:])
        if self.augment:
            if self.deterministic_augment:
                state = random.getstate()
                random.seed(hash((pair.stem, self.scale)) % (2**32))
                lr, hr = paired_augment(lr, hr, self.horizontal_flip, self.vertical_flip, self.rot90)
                random.setstate(state)
            else:
                lr, hr = paired_augment(lr, hr, self.horizontal_flip, self.vertical_flip, self.rot90)
        return lr, hr, pair.stem
