"""Reproducibility + device selection.

Note: `PYTORCH_ENABLE_MPS_FALLBACK=1` must be exported BEFORE `torch` is imported.
We set it in `scripts/*.sh` launchers, not here. This module assumes it's already set.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class DeviceInfo:
    name: str
    backend: str
    fallback_enabled: bool


def pick_device(prefer: str = "mps") -> DeviceInfo:
    """Pick first available device in order: prefer -> cuda -> mps -> cpu.

    Imports torch lazily so non-DL workflows don't pay the cost.
    """
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "cpu", False)

    candidates = [prefer, "cuda", "mps", "cpu"]
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c == "cuda" and torch.cuda.is_available():
            return DeviceInfo("cuda", "cuda", False)
        if c == "mps" and torch.backends.mps.is_available():
            return DeviceInfo(
                "mps",
                "mps",
                fallback_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            )
        if c == "cpu":
            return DeviceInfo("cpu", "cpu", False)
    return DeviceInfo("cpu", "cpu", False)
