"""Shared training driver for U-Net and SegFormer-B0.

Uses MPS bf16 autocast on Apple Silicon, with NaN-safe fp32 fallback. Mirrors
both the configs/experiments/{unet,segformer}_multiclass.yaml schemas; the
model factory is chosen by `experiment.model`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import IGNORE_ID, summary
from src.models.datasets import TileDataset, compute_normalisation
from src.models.losses import combo_loss
from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest
from src.utils.seed import pick_device, set_seeds

log = get_logger(__name__)


def _index_paths(events: list[str]) -> list[Path]:
    return [REPO_ROOT / "data" / "processed" / f"tile_index_{e}.parquet" for e in events]


def _build_model(cfg):
    name = cfg.experiment.model
    if name == "unet":
        from src.models.unet_model import build_unet
        return build_unet(
            in_channels=cfg.unet.in_channels,
            num_classes=cfg.unet.num_classes,
            encoder_name=cfg.unet.encoder_name,
            encoder_weights=cfg.unet.encoder_weights,
            decoder_channels=tuple(cfg.unet.decoder_channels),
            dropout=cfg.unet.dropout,
        )
    if name == "segformer":
        from src.models.segformer_model import SegformerWrapper
        return SegformerWrapper(
            in_channels=cfg.segformer.in_channels,
            num_classes=cfg.segformer.num_classes,
            backbone=cfg.segformer.backbone,
        )
    raise ValueError(f"Unknown model: {name}")


def _split_events(cfg) -> tuple[list[str], list[str], list[str]]:
    if cfg.experiment.split_mode == "vertical_slice":
        e = cfg.experiment.event
        return [e], [e], [e]
    return list(cfg.events.train), list(cfg.events.val), list(cfg.events.test)


def _sliding_window_predict(model, image: np.ndarray, mean: np.ndarray, std: np.ndarray,
                            tile: int, stride: int, device: torch.device,
                            num_classes: int = 4) -> np.ndarray:
    """Predict an entire H×W image with overlap averaging."""
    C, H, W = image.shape
    image = (image - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    logits_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + tile, H)
                x_end = min(x + tile, W)
                y_start = max(0, y_end - tile)
                x_start = max(0, x_end - tile)
                patch = image[:, y_start:y_end, x_start:x_end]
                pt = torch.from_numpy(patch.astype(np.float32))[None].to(device)
                out = model(pt).float().cpu().numpy()[0]
                logits_sum[:, y_start:y_end, x_start:x_end] += out
                counts[y_start:y_end, x_start:x_end] += 1
    logits_sum /= np.maximum(counts[None], 1)
    pred = logits_sum.argmax(axis=0).astype(np.uint8)
    return pred


def _write_uint8(path: Path, arr: np.ndarray, transform, crs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
        "nodata": 255, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr[np.newaxis, ...])


def train(config_path: str, fast_mode: bool = False) -> dict:
    cfg = load_config(config_path)
    set_seeds(cfg.project.seed)

    train_events, val_events, test_events = _split_events(cfg)
    log.info("Train: %s | Val: %s | Test: %s", train_events, val_events, test_events)

    # 1. Compute normalisation from the train set only
    train_ds_for_stats = TileDataset(
        _index_paths(train_events), split="train", cfg=cfg, augment=False,
    )
    stats_dir = REPO_ROOT / cfg.outputs.model_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "normalization.json"
    if not stats_path.exists():
        stats = compute_normalisation(train_ds_for_stats, max_tiles=200)
        stats_path.write_text(json.dumps(stats, indent=2))
        log.info("Wrote train normalisation stats to %s", stats_path)

    epochs = cfg.fast_mode.epochs if fast_mode else cfg.train.epochs
    max_tr = cfg.fast_mode.max_tiles_train if fast_mode else None
    max_va = cfg.fast_mode.max_tiles_val if fast_mode else None

    train_ds = TileDataset(_index_paths(train_events), "train", cfg=cfg,
                           augment=True, stats_path=stats_path, max_tiles=max_tr)
    val_ds = TileDataset(_index_paths(val_events), "val", cfg=cfg,
                         augment=False, stats_path=stats_path, max_tiles=max_va)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    log.info("DataLoaders: train=%d val=%d (batch=%d)",
             len(train_ds), len(val_ds), cfg.train.batch_size)

    device_info = pick_device(cfg.device.prefer)
    device = torch.device(device_info.name)
    log.info("Device: %s (fallback_enabled=%s)", device_info.name, device_info.fallback_enabled)
    if device_info.name == "mps" and not device_info.fallback_enabled:
        log.warning("PYTORCH_ENABLE_MPS_FALLBACK is not set — some ops may error on MPS. "
                    "Source scripts/setup_env.sh before running.")

    model = _build_model(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )

    use_bf16 = device_info.name in ("mps", "cuda")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    best_macro_iou = -math.inf
    best_path = stats_dir / "best.pt"
    history: list[dict] = []
    no_improve = 0
    step_times: list[float] = []
    baseline_step = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        running_n = 0
        for step, (image, _mask, label) in enumerate(train_dl):
            image = image.to(device, non_blocking=False)
            label = label.to(device, non_blocking=False)
            t0 = time.time()
            with torch.autocast(device_type=device_info.name, dtype=autocast_dtype, enabled=use_bf16):
                logits = model(image)
            loss, components = combo_loss(
                logits, label,
                ce_weight=cfg.train.loss.ce_weight,
                dice_weight=cfg.train.loss.dice_weight,
                ignore_index=cfg.train.loss.ignore_index,
            )
            loss.backward()
            if (step + 1) % cfg.train.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
            running += components["total"] * image.size(0)
            running_n += image.size(0)
            dt = time.time() - t0
            if len(step_times) < 10:
                step_times.append(dt)
                if len(step_times) == 10:
                    baseline_step = float(np.median(step_times))
                    log.info("Baseline step time: %.2fs", baseline_step)
            elif baseline_step is not None and dt > 3 * baseline_step:
                log.warning("Step %d took %.2fs (>3x baseline). Likely MPS CPU fallback. "
                            "Consider reducing tile_size or batch.", step, dt)
            if step % cfg.train.log_every == 0:
                log.info("  epoch=%d step=%d/%d loss=%.4f (ce=%.4f dice=%.4f) dt=%.2fs",
                         epoch, step, len(train_dl), components["total"],
                         components["ce"], components["dice"], dt)

        train_loss = running / max(running_n, 1)

        # Validation
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for image, _mask, label in val_dl:
                image = image.to(device); label = label.to(device)
                with torch.autocast(device_type=device_info.name, dtype=autocast_dtype, enabled=use_bf16):
                    logits = model(image)
                p = logits.float().argmax(dim=1).cpu().numpy().astype(np.uint8)
                all_pred.append(p); all_true.append(label.cpu().numpy().astype(np.uint8))
        if all_pred:
            ps = np.concatenate([a.ravel() for a in all_pred])
            ts = np.concatenate([a.ravel() for a in all_true])
            s = summary(ps, ts, num_classes=4)
            log.info("Epoch %d | train_loss=%.4f | val macro-IoU=%.3f macro-F1=%.3f",
                     epoch, train_loss, s["macro_iou"], s["macro_f1"])
            history.append({"epoch": epoch, "train_loss": train_loss, "val": s})
            if s["macro_iou"] > best_macro_iou:
                best_macro_iou = s["macro_iou"]
                torch.save(model.state_dict(), best_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.train.early_stop_patience:
                    log.info("Early stop at epoch %d (no improvement for %d epochs).",
                             epoch, no_improve)
                    break
        else:
            log.info("Epoch %d | train_loss=%.4f | (no val tiles)", epoch, train_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val": None})

    # Save history + final config snapshot
    (stats_dir / "history.json").write_text(json.dumps(history, indent=2))
    from omegaconf import OmegaConf
    (stats_dir / "config_snapshot.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Inference on val + test events
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    stats = json.loads(stats_path.read_text())
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    metrics_all: dict = {"history": history, "model": cfg.experiment.model,
                         "train_events": train_events,
                         "val_events": val_events, "test_events": test_events}
    pred_dir = REPO_ROOT / cfg.outputs.prediction_dir
    metrics_dir = REPO_ROOT / cfg.outputs.metrics_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from src.features.stack_features import build_stack
    for label_key, events in (("val", val_events), ("test", test_events)):
        metrics_all[label_key] = {}
        for ev in events:
            interim = REPO_ROOT / "data" / "interim" / ev
            if not (interim / "pre_stack_10m.tif").exists():
                log.warning("Skipping inference for %s — interim composites missing.", ev)
                continue
            with rasterio.open(interim / "pre_stack_10m.tif") as ds:
                pre = ds.read().astype(np.float32); transform = ds.transform; crs = ds.crs
            with rasterio.open(interim / "post_stack_10m.tif") as ds:
                post = ds.read().astype(np.float32)
            image = build_stack(pre, post)
            tile = int(cfg.data.tile_size)
            stride = tile // 2
            pred = _sliding_window_predict(model, image, mean, std, tile, stride, device,
                                           num_classes=4)
            with rasterio.open(interim / "label_10m.tif") as ds:
                lab = ds.read(1)
            metrics_all[label_key][ev] = summary(pred, lab, num_classes=4)
            out_path = pred_dir / f"{ev}.tif"
            _write_uint8(out_path, pred, transform, crs)
            write_manifest(
                out_path, event_id=ev,
                pipeline_step=f"{cfg.experiment.model}.predict",
                inputs={"best_checkpoint": str(best_path.relative_to(REPO_ROOT)),
                        "normalization": str(stats_path.relative_to(REPO_ROOT))},
                crs=str(crs),
            )
            log.info("[%s/%s] macro-IoU=%.3f", label_key, ev,
                     metrics_all[label_key][ev]["macro_iou"])

    (metrics_dir / "metrics.json").write_text(json.dumps(metrics_all, indent=2))
    return metrics_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fast-mode", action="store_true",
                   help="Use cfg.fast_mode (5 epochs, tile subset, early stop).")
    args = p.parse_args()
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        log.warning("Set PYTORCH_ENABLE_MPS_FALLBACK=1 in-process — "
                    "for production runs, source scripts/setup_env.sh BEFORE python starts.")
    train(args.config, fast_mode=args.fast_mode)


if __name__ == "__main__":
    main()
