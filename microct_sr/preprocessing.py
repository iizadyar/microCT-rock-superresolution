from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(folder: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])
    if not files:
        raise FileNotFoundError(f"No image files found in: {folder}")
    return files


def load_grayscale(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3:
        t = t.squeeze(0)
    arr = (t.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def center_crop(t: torch.Tensor, height: int, width: int) -> torch.Tensor:
    _, h, w = t.shape
    top = max((h - height) // 2, 0)
    left = max((w - width) // 2, 0)
    return t[:, top:top + height, left:left + width]


def match_scale(lr: torch.Tensor, hr: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, lr_h, lr_w = lr.shape
    _, hr_h, hr_w = hr.shape
    use_lr_h = min(lr_h, hr_h // scale)
    use_lr_w = min(lr_w, hr_w // scale)
    lr = center_crop(lr, use_lr_h, use_lr_w)
    hr = center_crop(hr, use_lr_h * scale, use_lr_w * scale)
    return lr, hr


def bicubic_resize(lr: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    x = lr.unsqueeze(0) if lr.ndim == 3 else lr
    y = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    return y.squeeze(0) if lr.ndim == 3 else y
