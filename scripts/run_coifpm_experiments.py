from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microct_sr.config import load_config
from microct_sr.engine import evaluate_bicubic, train_models


MODELS = ["simplified_edsr"]


def main():
    for config_path in ["configs/coifpm_4x.yaml", "configs/coifpm_8x.yaml", "configs/coifpm_16x.yaml"]:
        print(f"Running {config_path}")
        cfg = load_config(config_path)
        evaluate_bicubic(cfg)
        train_models(cfg, MODELS)


if __name__ == "__main__":
    main()
