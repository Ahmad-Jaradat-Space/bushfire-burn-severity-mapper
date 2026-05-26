"""Tiny IO helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(obj, fh, indent=2, default=str, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
