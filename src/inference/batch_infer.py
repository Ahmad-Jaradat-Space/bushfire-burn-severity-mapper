"""Tiled batch inference + a throughput / memory / cost benchmark.

The platform question behind "operate reliably at scale" is concrete: how many
pixels per second, at what peak memory, and therefore what cloud cost to map a
whole state? This runs the event-wise U-Net over an event with the same
overlap-averaged sliding window the trainer uses, times it, samples peak RSS,
and projects the cost of a continental run.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import psutil
import rasterio

from src.features.stack_features import build_stack
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)

EVENT = "kangaroo_island_2019_2020"
MODEL_DIR = REPO_ROOT / "outputs" / "models" / "unet_eventwise"
# Indicative on-demand worker-hour price (GCP, USD) for the projection.
USD_PER_HOUR = 0.0386
STATE_HECTARES = 80_000_000  # ~ NSW, a realistic enterprise AOI
TARGET_RES_M = 0.5  # project at operational aerial/drone resolution, not 30 m


def project_cost(
    px_per_s: float,
    hectares: float = STATE_HECTARES,
    res_m: float = TARGET_RES_M,
    usd_per_hour: float = USD_PER_HOUR,
) -> dict:
    """Project measured throughput onto a large AOI at a target resolution."""
    px = hectares * 1e4 / (res_m**2)
    sec = px / px_per_s
    return {"pixels": px, "hours": sec / 3600.0, "usd": sec / 3600.0 * usd_per_hour}


def _load_model(device):
    import torch

    from src.models.unet_model import build_unet
    from src.utils.config import load_config

    cfg = load_config("configs/experiments/unet_eventwise.yaml")
    model = build_unet(
        in_channels=cfg.unet.in_channels,
        num_classes=cfg.unet.num_classes,
        encoder_name=cfg.unet.encoder_name,
        encoder_weights=None,
        decoder_channels=tuple(cfg.unet.decoder_channels),
        dropout=cfg.unet.dropout,
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    return model.to(device).eval()


def tiled_event_inference(event: str, device: torch.device, write: bool = False) -> dict:
    """Predict one event end-to-end; return timing + memory stats."""
    from src.models.train_segmenter import _sliding_window_predict, _write_uint8

    stats = json.loads((MODEL_DIR / "normalization.json").read_text())
    mean = np.array(stats["mean"], np.float32)
    std = np.array(stats["std"], np.float32)
    interim = REPO_ROOT / "data" / "interim" / event
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre = ds.read().astype(np.float32)
        transform = ds.transform
        crs = ds.crs
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post = ds.read().astype(np.float32)
    image = build_stack(pre, post)
    H, W = image.shape[1:]

    model = _load_model(device)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.time()
    pred = _sliding_window_predict(
        model, image, mean, std, tile=256, stride=128, device=device, num_classes=4
    )
    seconds = time.time() - t0
    peak_mb = (proc.memory_info().rss) / 1e6
    px = int(H * W)
    if write:
        out = REPO_ROOT / "outputs" / "predictions" / "batch_demo" / f"{event}.tif"
        _write_uint8(out, pred, transform, crs)

    return {
        "event": event,
        "device": device.type,
        "pixels": px,
        "seconds": round(seconds, 2),
        "px_per_s": int(px / seconds),
        "peak_rss_mb": round(peak_mb, 1),
        "rss_delta_mb": round((peak_mb * 1e6 - rss0) / 1e6, 1),
    }


def benchmark(events: list[str] | None = None, devices: list[str] | None = None) -> pd.DataFrame:
    import torch

    events = events or [EVENT]
    if devices is None:
        devices = ["mps"] if torch.backends.mps.is_available() else ["cpu"]
    rows = []
    for dev in devices:
        for ev in events:
            r = tiled_event_inference(ev, torch.device(dev))
            proj = project_cost(r["px_per_s"])  # state mapped at aerial resolution
            r["proj_hours_state"] = round(proj["hours"], 1)
            r["proj_usd_state"] = round(proj["usd"], 2)
            rows.append(r)
            log.info(
                "%-4s %-28s %d px in %.1fs = %d px/s, peak %.0f MB",
                dev,
                ev,
                r["pixels"],
                r["seconds"],
                r["px_per_s"],
                r["peak_rss_mb"],
            )
    return pd.DataFrame(rows)
