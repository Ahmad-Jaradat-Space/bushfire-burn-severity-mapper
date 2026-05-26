"""Per-output provenance manifest.

Every raster written by this repo ships with a sidecar `<output>.provenance.json`
containing source URLs, STAC item IDs, pipeline git SHA, CRS, resampling, class remap,
and a UTC timestamp. This lets a hiring manager (or future-you) reproduce any artefact.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

from src.utils.io import write_json


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(
    output_path: str | Path,
    *,
    event_id: str,
    pipeline_step: str,
    inputs: dict[str, Any],
    crs: str,
    resampling: str | None = None,
    class_remap: dict[str, int] | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `<output_path>.provenance.json` next to the raster.

    Parameters
    ----------
    output_path : path to the raster being described (existence not required at call time)
    event_id : stable event identifier
    pipeline_step : e.g. "fetch_sentinel", "preprocess", "train_rf"
    inputs : dict of input descriptors — e.g. {"stac_items": [...], "source_url": "..."}
    crs : EPSG string e.g. "EPSG:32754"
    resampling : "bilinear" | "nearest" | None
    class_remap : mapping (only relevant for label rasters)
    notes : free-form
    extra : any additional fields
    """
    output_path = Path(output_path)
    manifest = {
        "output_file": str(output_path.name),
        "event_id": event_id,
        "pipeline_step": pipeline_step,
        "git_sha": _git_sha(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "crs": crs,
        "resampling": resampling,
        "class_remap": class_remap,
        "inputs": inputs,
        "notes": notes,
    }
    if extra:
        manifest.update(extra)

    manifest_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    write_json(manifest_path, manifest)
    return manifest_path
