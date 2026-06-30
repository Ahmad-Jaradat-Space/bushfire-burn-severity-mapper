"""Self-contained Prithvi-EO-2.0 frozen-probe trainer + predictor.

Trains only the decoder (the 300M ViT stays frozen) on the event-wise tiles,
using the 6-band POST composite as Prithvi's single-frame input, then predicts
the held-out Kangaroo scene and scores it on the same macro-IoU as every other
method. Kept separate from the 18-channel trainer so the foundation-model
dependency (terratorch) never touches the core pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import summary
from src.models.losses import combo_loss
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest
from src.utils.seed import pick_device, set_seeds

log = get_logger(__name__)

TRAIN_EVENTS = ["currowan_2019_2020", "gospers_mountain_2019_2020"]
VAL_EVENT = "kangaroo_island_2019_2020"
MODEL_DIR = REPO_ROOT / "outputs" / "models" / "prithvi_eventwise"
PRED_DIR = REPO_ROOT / "outputs" / "predictions" / "prithvi_eventwise"


def _tile_paths(events: list[str], split: str) -> list[Path]:
    paths: list[Path] = []
    for ev in events:
        idx = REPO_ROOT / "data" / "processed" / f"tile_index_{ev}.parquet"
        if not idx.exists():
            continue
        df = pd.read_parquet(idx)
        df = df[df["split"] == split]
        paths += [REPO_ROOT / p for p in df["tile_path"]]
    return paths


def _norm_stats(paths: list[Path], max_tiles: int = 120) -> tuple[np.ndarray, np.ndarray]:
    cols = [
        np.nan_to_num(np.load(p)["post"].astype(np.float32)).reshape(6, -1)
        for p in paths[:max_tiles]
    ]
    arr = np.concatenate(cols, axis=1)
    return arr.mean(axis=1), arr.std(axis=1) + 1e-6


def _prep(post: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    post = np.nan_to_num(post.astype(np.float32))
    post = (post - mean[:, None, None]) / std[:, None, None]
    return np.clip(np.nan_to_num(post), -10.0, 10.0)


class PostTiles(Dataset):
    def __init__(self, paths, mean, std):
        self.paths, self.mean, self.std = paths, mean, std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        post = _prep(d["post"], self.mean, self.std)
        return torch.from_numpy(post), torch.from_numpy(d["label"].astype(np.int64))


def _predict_event(model, event, mean, std, device, tile=256, stride=256) -> np.ndarray:
    with rasterio.open(REPO_ROOT / "data" / "interim" / event / "post_stack_10m.tif") as ds:
        post = _prep(ds.read(), mean, std)
    _, H, W = post.shape
    logits_sum = np.zeros((4, H, W), np.float32)
    counts = np.zeros((H, W), np.float32)
    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                ye, xe = min(y + tile, H), min(x + tile, W)
                y0, x0 = max(0, ye - tile), max(0, xe - tile)
                patch = torch.from_numpy(post[:, y0:ye, x0:xe])[None].to(device)
                out = model(patch).float().cpu().numpy()[0]
                logits_sum[:, y0:ye, x0:xe] += out
                counts[y0:ye, x0:xe] += 1
    return (logits_sum / np.maximum(counts[None], 1)).argmax(0).astype(np.uint8)


def train(
    epochs: int = 20, batch_size: int = 4, lr: float = 1e-3, seed: int = 42, fast_mode: bool = False
) -> dict:
    set_seeds(seed)
    device = torch.device(pick_device("mps").name)
    train_paths = _tile_paths(TRAIN_EVENTS, "train")
    val_paths = _tile_paths([VAL_EVENT], "val")
    if fast_mode:
        train_paths, val_paths, epochs = train_paths[:16], val_paths[:4], 2
    if not train_paths:
        raise SystemExit("No event-wise train tiles found — run tiling first.")
    mean, std = _norm_stats(train_paths)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()})
    )
    log.info(
        "Prithvi probe: %d train / %d val tiles on %s",
        len(train_paths),
        len(val_paths),
        device.type,
    )

    from src.models.prithvi_model import build_prithvi

    model = build_prithvi(num_classes=4, frozen=True, dropout=0.1).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    train_dl = DataLoader(
        PostTiles(train_paths, mean, std), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dl = DataLoader(PostTiles(val_paths, mean, std), batch_size=batch_size)

    best_iou, best_path, history, no_improve = -math.inf, MODEL_DIR / "best.pt", [], 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for img, label in train_dl:
            img, label = img.to(device), label.to(device)
            loss, comp = combo_loss(
                model(img), label, ce_weight=0.5, dice_weight=0.5, ignore_index=255
            )
            if not torch.isfinite(loss):
                continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += comp["total"]
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for img, label in val_dl:
                p = model(img.to(device)).float().argmax(1).cpu().numpy().astype(np.uint8)
                preds.append(p)
                trues.append(label.numpy().astype(np.uint8))
        s = (
            summary(
                np.concatenate([p.ravel() for p in preds]),
                np.concatenate([t.ravel() for t in trues]),
                num_classes=4,
            )
            if preds
            else {"macro_iou": float("nan")}
        )
        log.info(
            "epoch %d | train_loss=%.4f | val macro-IoU=%.3f",
            epoch,
            running / max(len(train_dl), 1),
            s["macro_iou"],
        )
        history.append({"epoch": epoch, "val": s})
        if s["macro_iou"] > best_iou:
            best_iou = s["macro_iou"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= 6:
                log.info("early stop at epoch %d", epoch)
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    (MODEL_DIR / "history.json").write_text(json.dumps(history, indent=2))

    # Predict + score the held-out event.
    with rasterio.open(REPO_ROOT / "data" / "interim" / VAL_EVENT / "post_stack_10m.tif") as ds:
        transform, crs = ds.transform, ds.crs
    pred = _predict_event(model, VAL_EVENT, mean, std, device)
    with rasterio.open(REPO_ROOT / "data" / "interim" / VAL_EVENT / "label_10m.tif") as ds:
        lab = ds.read(1)
    final = summary(pred, lab, num_classes=4)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_tif = PRED_DIR / f"{VAL_EVENT}.tif"
    meta = {
        "driver": "GTiff",
        "height": pred.shape[0],
        "width": pred.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(pred[np.newaxis])
    write_manifest(
        out_tif,
        event_id=VAL_EVENT,
        pipeline_step="prithvi.predict",
        inputs={
            "backbone": "Prithvi-EO-2.0-300M (frozen)",
            "trainable": "decoder",
            "best_val_macro_iou": best_iou,
        },
        crs=str(crs),
    )
    log.info(
        "Prithvi event-wise macro-IoU on %s = %.3f (best val %.3f)",
        VAL_EVENT,
        final["macro_iou"],
        best_iou,
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"event_wise": final}, indent=2))
    return final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fast-mode", action="store_true")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, fast_mode=args.fast_mode)


if __name__ == "__main__":
    main()
