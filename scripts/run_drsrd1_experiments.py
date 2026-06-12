from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microct_sr.config import load_config
from microct_sr.engine import evaluate_bicubic, train_models


MODELS = ["srcnn", "fsrcnn", "edsr", "rcan"]


def main():
    cfg = load_config("configs/drsrd1_4x.yaml")
    evaluate_bicubic(cfg)
    train_models(cfg, MODELS)


if __name__ == "__main__":
    main()
