from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microct_sr.config import load_config
from microct_sr.engine import evaluate_bicubic, train_models
from scripts.make_demo_data import create_demo_dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS])


def copy_subset(source_lr: Path, source_hr: Path, output_root: Path, max_images: int) -> bool:
    lr_files = {p.stem: p for p in image_files(source_lr)}
    hr_files = {p.stem: p for p in image_files(source_hr)}
    stems = sorted(set(lr_files).intersection(hr_files))
    if not stems:
        return False
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "lr").mkdir(parents=True, exist_ok=True)
    (output_root / "hr").mkdir(parents=True, exist_ok=True)
    for stem in stems[:max_images]:
        shutil.copy2(lr_files[stem], output_root / "lr" / lr_files[stem].name)
        shutil.copy2(hr_files[stem], output_root / "hr" / hr_files[stem].name)
    return True


def source_paths(source: str):
    options = {
        "coifpm4x": (ROOT / "data/COIFPM/4x/lr", ROOT / "data/COIFPM/4x/hr", 4),
        "coifpm8x": (ROOT / "data/COIFPM/8x/lr", ROOT / "data/COIFPM/8x/hr", 8),
        "coifpm16x": (ROOT / "data/COIFPM/16x/lr", ROOT / "data/COIFPM/16x/hr", 16),
        "drsrd1": (ROOT / "data/DRSRD1/4x/lr", ROOT / "data/DRSRD1/4x/hr", 4),
    }
    return options[source]


def main():
    parser = argparse.ArgumentParser(description="Run a short code test.")
    parser.add_argument("--source", default="coifpm4x", choices=["coifpm4x", "coifpm8x", "coifpm16x", "drsrd1", "synthetic"])
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--no-hpo", action="store_true")
    args = parser.parse_args()

    output_data = ROOT / "outputs" / "quick_test_data"
    if args.source == "synthetic":
        create_demo_dataset(output_data, scale=4, n_images=args.max_images, hr_size=64)
        scale = 4
        source_name = "synthetic"
    else:
        lr_dir, hr_dir, scale = source_paths(args.source)
        copied = copy_subset(lr_dir, hr_dir, output_data, args.max_images)
        if not copied:
            create_demo_dataset(output_data, scale=4, n_images=args.max_images, hr_size=64)
            scale = 4
            source_name = "synthetic"
        else:
            source_name = args.source

    cfg = load_config("configs/quick_test.yaml")
    cfg["dataset"]["scale"] = scale
    cfg["dataset"]["lr_dir"] = str(output_data / "lr")
    cfg["dataset"]["hr_dir"] = str(output_data / "hr")
    cfg["training"]["epochs"] = args.epochs
    cfg["hpo"]["enabled"] = not args.no_hpo

    print("Quick test started")
    print(f"Data source: {source_name}")
    print(f"Scale factor: {scale}x")
    print(f"Maximum image pairs: {args.max_images}")
    print(f"Epochs per model: {args.epochs}")
    if args.source == "drsrd1":
        models = ["srcnn", "fsrcnn", "edsr", "rcan"]
        cfg["models"] = {
            "srcnn": {},
            "fsrcnn": {},
            "edsr": {"num_features": 4, "num_blocks": 1, "res_scale": 0.1},
            "rcan": {"num_features": 4, "num_blocks": 1, "reduction": 4},
        }
        print("Models: bicubic, SRCNN, FSRCNN, EDSR, RCAN")
    else:
        models = ["simplified_edsr"]
        cfg["models"] = {
            "simplified_edsr": {"num_features": 4, "num_blocks": 1, "res_scale": 0.1}
        }
        print("Models: bicubic, simplified EDSR")
    print("Loss: MSE")

    evaluate_bicubic(cfg)
    train_models(cfg, models)
    print("Quick test completed")


if __name__ == "__main__":
    main()
