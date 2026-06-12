from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def create_demo_dataset(output_dir: str | Path, scale: int = 4, n_images: int = 12, hr_size: int = 64):
    output_dir = Path(output_dir)
    hr_dir = output_dir / "hr"
    lr_dir = output_dir / "lr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:hr_size, :hr_size]
    for i in range(n_images):
        arr = 0.5 + 0.25 * np.sin((x + i) / 5.0) + 0.20 * np.cos((y - i) / 7.0)
        for _ in range(8):
            cx, cy = rng.integers(4, hr_size - 4, size=2)
            r = rng.integers(2, 7)
            mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
            arr[mask] -= rng.uniform(0.2, 0.5)
        arr = np.clip(arr + rng.normal(0, 0.02, arr.shape), 0, 1)
        hr = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        lr = hr.resize((hr_size // scale, hr_size // scale), resample=Image.Resampling.BICUBIC)
        name = f"sample_{i:03d}.png"
        hr.save(hr_dir / name)
        lr.save(lr_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/quick_test_data")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--n-images", type=int, default=12)
    args = parser.parse_args()
    create_demo_dataset(args.output, args.scale, args.n_images)
    print(f"Demo data written to {args.output}")


if __name__ == "__main__":
    main()
