"""Generate the M6.5 non-technical demo assets.

If real Kangaroo Island predictions exist in outputs/predictions/, those are
used; otherwise we generate a synthetic Kangaroo-shaped composite + prediction
so the demo pipeline + HTML render correctly even before live data lands.

Outputs (under docs/demo/):
  kangaroo_pre.png         Pre-fire true-colour composite
  kangaroo_post.png        Post-fire true-colour composite
  kangaroo_severity.png    Predicted severity map (with legend)
  kangaroo_animation.gif   3-frame loop: pre → post → severity
  kangaroo_slider.html     juxtapose.js before/after slider
  non_expert_panel.md      Plain-English explainer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.maps import (
    SEVERITY_COLORS,
    SEVERITY_NAMES,
    render_severity_map,
    render_truecolor,
)

log = get_logger(__name__)

DEMO_DIR = REPO_ROOT / "docs" / "demo"


def _synthetic_kangaroo(h=512, w=900, seed=0):
    """Synthetic stand-in: pre/post + severity that look like a fire over an island.

    Used when no real prediction exists yet so the demo always renders.
    """
    rng = np.random.default_rng(seed)
    # Land/sea mask: rough island shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    rx, ry = w * 0.40, h * 0.32
    land = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy) ** 2) / (ry ** 2) < 1.0
    sea = ~land

    # Pre-fire RGB (greens / khakis on land, blue on water)
    pre = np.zeros((3, h, w), dtype=np.float32)
    pre[0] = np.where(land, 0.18 + 0.08 * rng.random((h, w)),
                            0.05 + 0.03 * rng.random((h, w)))   # R
    pre[1] = np.where(land, 0.30 + 0.10 * rng.random((h, w)),
                            0.12 + 0.04 * rng.random((h, w)))   # G
    pre[2] = np.where(land, 0.14 + 0.05 * rng.random((h, w)),
                            0.35 + 0.06 * rng.random((h, w)))   # B

    # Burn footprint: west two-thirds of the island
    burn_centre = (cy - 30, int(cx - rx * 0.4))
    rb_x = w * 0.20
    rb_y = h * 0.22
    burn = land & (((xx - burn_centre[1]) ** 2) / (rb_x ** 2)
                   + ((yy - burn_centre[0]) ** 2) / (rb_y ** 2) < 1.0)

    # Severity gradient: outer=low, middle=high, inner=very_high
    radial = np.sqrt(((xx - burn_centre[1]) / rb_x) ** 2
                     + ((yy - burn_centre[0]) / rb_y) ** 2)
    severity = np.full((h, w), 255, dtype=np.uint8)
    severity[land] = 0
    severity[burn & (radial < 0.95)] = 1
    severity[burn & (radial < 0.65)] = 2
    severity[burn & (radial < 0.35)] = 3

    # Post-fire RGB: dark/red where burned
    post = pre.copy()
    post[0] = np.where(burn, 0.28 + 0.05 * rng.random((h, w)), pre[0])
    post[1] = np.where(burn, 0.14 + 0.04 * rng.random((h, w)), pre[1])
    post[2] = np.where(burn, 0.08 + 0.02 * rng.random((h, w)), pre[2])

    # Keep water blue + smooth
    for c, v in enumerate([0.06, 0.14, 0.36]):
        pre[c] = np.where(sea, v + 0.03 * rng.random((h, w)), pre[c])
        post[c] = np.where(sea, v + 0.03 * rng.random((h, w)), post[c])

    return pre, post, severity


def _kangaroo_real() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Try to load real Kangaroo Island assets if M3/M4 have run."""
    interim = REPO_ROOT / "data" / "interim" / "kangaroo_island_2019_2020"
    pred_candidates = [
        REPO_ROOT / "outputs" / "predictions" / "rf_multiclass" / "kangaroo_island_2019_2020.tif",
        REPO_ROOT / "outputs" / "predictions" / "baseline_dnbr" / "kangaroo_island_2019_2020" / "multiclass.tif",
    ]
    pred_path = next((p for p in pred_candidates if p.exists()), None)
    if pred_path is None or not (interim / "pre_stack_10m.tif").exists():
        return None
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre6 = ds.read().astype(np.float32)
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post6 = ds.read().astype(np.float32)
    # Bands ordered B02..B12. True-colour uses R=B04 (idx 2), G=B03 (1), B=B02 (0).
    pre_rgb = pre6[[2, 1, 0]]
    post_rgb = post6[[2, 1, 0]]
    with rasterio.open(pred_path) as ds:
        sev = ds.read(1)
    return pre_rgb, post_rgb, sev


def _animate_gif(pre_png: Path, post_png: Path, severity_png: Path, out: Path,
                 duration_ms: int = 1800) -> Path:
    frames = [Image.open(p).convert("RGB") for p in (pre_png, post_png, severity_png)]
    # Resize to same width to avoid the GIF picking the widest frame's canvas
    target_w = min(f.width for f in frames)
    frames = [f.resize((target_w, int(f.height * target_w / f.width))) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms,
                   loop=0, optimize=False)
    return out


SLIDER_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Kangaroo Island 2019–2020 — Pre vs Post burn severity</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css" />
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ font-size: 1.4em; }}
    .caveat {{ background: #fff3cd; border-left: 4px solid #f0ad4e; padding: 0.8em 1em;
              font-size: 0.92em; margin: 1em 0; }}
    .juxtapose {{ width: 100%; height: 600px; }}
    footer {{ margin-top: 2em; font-size: 0.85em; color: #666; }}
  </style>
</head>
<body>
  <h1>Kangaroo Island — before vs after the 2019–2020 bushfires</h1>
  <p>Slide the bar left/right to compare the pre-fire and post-fire satellite views. Both images are from
  Sentinel-2 (European Space Agency), accessed via the Microsoft Planetary Computer STAC API.</p>

  <div class="caveat">
    <strong>Research use only.</strong> The supervised models in this project learn from
    <a href="https://gis.environment.gov.au/gispubmap/rest/services/threats/AUS_GEEBAM_Fire_Severity/MapServer">AUS&nbsp;GEEBAM</a>
    — a public satellite-derived <em>proxy</em> for burn severity. It is not field-validated ground truth, and
    these maps are not suitable for emergency response or safety-of-life decisions.
  </div>

  <div class="juxtapose">
    <img src="{pre_src}" data-label="Pre-fire (Oct 2019)" />
    <img src="{post_src}" data-label="Post-fire (Jan 2020)" />
  </div>

  <h2>Predicted severity</h2>
  <p><img src="{severity_src}" alt="Predicted severity map" style="width:100%;max-width:900px;" /></p>

  <footer>
    Contains modified Copernicus Sentinel data [2019–2020] processed by ESA.
    AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0.
  </footer>

  <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
</body>
</html>
"""


NON_EXPERT_MD = f"""# How to read this map (for non-experts)

We're mapping how badly each patch of bushland was burned during the 2019–2020 Australian
"Black Summer" fires, using free satellite photographs taken before and after the fires.

## What the colours mean

| Colour | What it means |
|---|---|
| <span style="color:{SEVERITY_COLORS[0]}">■</span> {SEVERITY_NAMES[0]} | The vegetation was not visibly burned in the satellite image. |
| <span style="color:{SEVERITY_COLORS[1]}">■</span> {SEVERITY_NAMES[1]} | Some scorching — leaves/grass affected, but the trees likely survived. |
| <span style="color:{SEVERITY_COLORS[2]}">■</span> {SEVERITY_NAMES[2]} | Significant burning — canopy loss in many trees. |
| <span style="color:{SEVERITY_COLORS[3]}">■</span> {SEVERITY_NAMES[3]} | The fire was hot enough that almost no green vegetation survived. |

## How a satellite "sees" fire

Live vegetation reflects a lot of near-infrared light (invisible to us, but the satellite
detects it). Burned vegetation reflects much less near-infrared and more short-wave infrared.
Comparing the two views before and after the fire makes burn scars stand out clearly.

## Where this map can be wrong

- **Low and moderate are merged.** The label source (AUS GEEBAM) does not separate them.
- **Steep south-facing slopes can look "burned"** even when they aren't — shadows look dark in satellite images.
- **Clouds and smoke** hide the surface. Where we couldn't see clearly in either image, we mark the pixel as "no data" (transparent).
- **This is not a real-time map.** All comparisons use images taken weeks or months apart, never live.

## What this map should not be used for

- Emergency response
- Insurance assessment
- Safety-of-life decisions

It is published as a research and learning artefact only.
"""


def make_demo(force_synthetic: bool = False) -> dict[str, Path]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    real = None if force_synthetic else _kangaroo_real()
    if real is None:
        log.warning("No real Kangaroo data found yet — rendering SYNTHETIC demo "
                    "so the slider/animation infrastructure is reviewable.")
        pre_rgb, post_rgb, severity = _synthetic_kangaroo()
    else:
        log.info("Using real Kangaroo Island composites + prediction.")
        pre_rgb, post_rgb, severity = real

    out_pre = DEMO_DIR / "kangaroo_pre.png"
    out_post = DEMO_DIR / "kangaroo_post.png"
    out_sev = DEMO_DIR / "kangaroo_severity.png"
    render_truecolor(pre_rgb, out_pre, title="Pre-fire (October 2019)")
    render_truecolor(post_rgb, out_post, title="Post-fire (January 2020)")
    render_severity_map(severity, out_sev, title="Predicted burn severity")

    out_gif = DEMO_DIR / "kangaroo_animation.gif"
    _animate_gif(out_pre, out_post, out_sev, out_gif)

    out_html = DEMO_DIR / "kangaroo_slider.html"
    out_html.write_text(SLIDER_HTML_TEMPLATE.format(
        pre_src=out_pre.name, post_src=out_post.name, severity_src=out_sev.name,
    ))

    out_md = DEMO_DIR / "non_expert_panel.md"
    out_md.write_text(NON_EXPERT_MD)

    paths = {"pre": out_pre, "post": out_post, "severity": out_sev,
             "gif": out_gif, "html": out_html, "md": out_md}
    for k, p in paths.items():
        log.info("  %s -> %s", k, p.relative_to(REPO_ROOT))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M6.5 non-technical demo assets.")
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Skip the check for real Kangaroo predictions.")
    args = parser.parse_args()
    make_demo(force_synthetic=args.force_synthetic)


if __name__ == "__main__":
    main()
