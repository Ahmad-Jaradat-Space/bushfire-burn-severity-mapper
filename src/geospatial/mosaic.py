"""Mosaic tiled inference outputs back into one seamless GeoTIFF.

Large-area processing tiles the imagery, predicts per tile, then has to stitch
the tiles back into a single continuous raster for delivery. This is the
re-assembly step (``rasterio.merge``), preserving the uint8 severity profile.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.merge import merge


def mosaic_tiles(tile_paths: list[str | Path], out_path: str | Path, nodata: int = 255) -> Path:
    """Merge adjacent/overlapping single-band tiles into one GeoTIFF."""
    out_path = Path(out_path)
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs, nodata=nodata)
        meta = srcs[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "nodata": nodata,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()
    return out_path
