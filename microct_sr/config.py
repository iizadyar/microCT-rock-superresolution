from __future__ import annotations

from pathlib import Path
import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def load_config(config_path: str | Path) -> dict:
    config_path = resolve_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(config_path)
    cfg["project_root"] = str(project_root())
    return cfg


def output_root(cfg: dict) -> Path:
    p = Path(cfg.get("outputs", {}).get("root", "outputs"))
    if p.is_absolute():
        return p
    return project_root() / p
