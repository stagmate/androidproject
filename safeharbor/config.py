from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
