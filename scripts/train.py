from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microct_sr.config import load_config
from microct_sr.engine import train_models

def main():
    parser = argparse.ArgumentParser(description="Train super-resolution models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_models(cfg, args.models)


if __name__ == "__main__":
    main()
