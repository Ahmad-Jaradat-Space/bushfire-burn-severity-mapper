"""Config loader.

Experiment configs live in `configs/experiments/<name>.yaml` and may declare an
`extends: <relative-path>` key. The loader resolves `extends` chains, merges
parents into children (child wins), and returns an OmegaConf DictConfig.

CLI dot-overrides ("data.tile_size=128", "experiment.split_mode=event_wise") are
applied last via `OmegaConf.from_dotlist`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _resolve_extends(path: Path) -> list[Path]:
    """Walk `extends:` chain back-to-front, returning [root, ..., self]."""
    chain: list[Path] = []
    visited: set[Path] = set()
    cur: Path | None = path
    while cur is not None:
        cur = cur.resolve()
        if cur in visited:
            raise ValueError(f"Cyclic extends chain at {cur}")
        visited.add(cur)
        chain.append(cur)
        cfg = OmegaConf.load(cur)
        ext = cfg.get("extends") if isinstance(cfg, DictConfig) else None
        cur = (cur.parent / ext).resolve() if ext else None
    chain.reverse()
    return chain


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load and merge a config file, applying optional CLI dot-overrides."""
    target = Path(path)
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    chain = _resolve_extends(target)
    merged = OmegaConf.create({})
    for p in chain:
        cfg = OmegaConf.load(p)
        # Don't propagate the `extends` key into the merged result
        if isinstance(cfg, DictConfig) and "extends" in cfg:
            cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != "extends"})
        merged = OmegaConf.merge(merged, cfg)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))

    assert isinstance(merged, (DictConfig, ListConfig))
    return merged  # type: ignore[return-value]
