"""Multi-filter site dossiers — one panel per human-review site.

For each site in the final queue, gather the free multi-sensor views that a
person would otherwise assemble by hand (built after the rank-123 alteration
zone was diagnosed exactly this way):

    NAIP false-colour IR  (NIR-R-G, ~0.6 m)   vegetation vs bare/stained rock
    Sentinel-2 true colour (10 m)              context colour
    Sentinel-2 SWIR geology (B12-B11-B04)      lithology / alteration contrast
    iron-oxide ratio (B04/B02)                 ferric staining (gossan filter)
    clay ratio (B11/B12)                       hydroxyl/argillic alteration
    GLO-30 hillshade                           landform context

Ratio maps at 10-20 m carry shadow/topography contamination in steep terrain
— they are QUALITATIVE overlays for a human, not measurements. Tiles fail
independently: a missing source (e.g. NAIP outside CONUS) renders as a
labelled "unavailable" tile, never an error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from interesting_intel.pipeline import CONUS, _read_bands, chip_grid

logger = logging.getLogger(__name__)

MPC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# (key, human label) in panel order
TILE_ORDER = [
    ("naip_false_ir", "NAIP false-colour IR (NIR-R-G) - vegetation RED"),
    ("s2_truecolor", "Sentinel-2 true colour"),
    ("s2_swir_geology", "Sentinel-2 SWIR geology (B12-B11-B04)"),
    ("iron_oxide", "iron-oxide ratio B04/B02 (bright = ferric staining)"),
    ("clay_ratio", "clay ratio B11/B12 (bright = hydroxyl; shadow-affected)"),
    ("hillshade", "GLO-30 hillshade"),
]


def stretch(a: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
    lo, hi = np.nanpercentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def to_rgb(bands: Sequence[np.ndarray]) -> np.ndarray:
    """Stack three bands into a stretched HxWx3 float [0,1] image."""
    return np.dstack([np.nan_to_num(stretch(b), nan=0.0) for b in bands])


def ratio_to_rgb(ratio: np.ndarray) -> np.ndarray:
    """Colormapped (inferno) view of a stretched band ratio."""
    import matplotlib.cm as cm

    r = np.nan_to_num(stretch(ratio), nan=0.0)
    return cm.inferno(r)[:, :, :3]


# ------------------------------------------------------------- data gathers

def naip_false_ir(lat: float, lon: float, half_m: float = 250, px: int = 800
                  ) -> Optional[np.ndarray]:
    """NIR-R-G false colour from NAIP's 4th band. CONUS only."""
    from archaeo_intel.data_access import stac_search

    w, s, e, n = CONUS
    if not (w <= lon <= e and s <= lat <= n):
        return None
    grid = chip_grid(lat, lon, half_m)
    feats = stac_search("naip", list(grid),
                        datetime="2018-01-01T00:00:00Z/2026-12-31T23:59:59Z",
                        limit=6, endpoint=MPC)
    acc = np.full((3, px, px), np.nan, "float32")
    for f in feats:
        try:
            g = _read_bands(f["assets"]["image"]["href"], grid, px, px,
                            bands=(4, 1, 2), signed=True)
        except Exception:
            continue
        acc = np.where(np.isfinite(acc), acc, g)
        if np.isfinite(acc[0]).mean() > 0.95:
            break
    if np.isfinite(acc[0]).mean() < 0.5:
        return None
    return to_rgb([acc[0], acc[1], acc[2]])


def s2_bands(lat: float, lon: float, half_m: float = 1000, px: int = 200,
             years=(2024, 2025, 2026)) -> Optional[Dict[str, np.ndarray]]:
    """Six bands of one recent low-cloud, mostly-clear Sentinel-2 scene."""
    from archaeo_intel.composite import find_scenes, scl_mask
    from archaeo_intel.data_access import read_grid

    grid = chip_grid(lat, lon, half_m)
    need = ("red", "green", "blue", "nir", "swir16", "swir22")
    for f in find_scenes(list(grid), season=("01-01", "12-31"), years=years,
                         cloud_lt=10, max_scenes=20):
        a = f.get("assets", {})
        if not all(b in a for b in need) or "scl" not in a:
            continue
        try:
            scl = read_grid(a["scl"]["href"], grid, px, px)
            if scl_mask(scl).mean() < 0.9:
                continue
            out = {b: read_grid(a[b]["href"], grid, px, px) for b in need}
            out["scene_id"] = f["id"]
            return out
        except Exception:
            continue
    return None


def hillshade(lat: float, lon: float, half_m: float = 1000, px: int = 200
              ) -> Optional[np.ndarray]:
    from matplotlib.colors import LightSource

    from archaeo_intel.data_access import read_grid, stac_search

    grid = chip_grid(lat, lon, half_m)
    try:
        f = stac_search("cop-dem-glo-30", list(grid), limit=2)[0]
        dem = read_grid(f["assets"]["data"]["href"], grid, px, px)
    except Exception:
        return None
    if not np.isfinite(dem).any():
        return None
    m_per_px = 2 * half_m / px
    ls = LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(np.nan_to_num(dem, nan=float(np.nanmedian(dem))),
                      dx=m_per_px, dy=m_per_px)
    return np.dstack([hs, hs, hs])


def gather_views(lat: float, lon: float, *, half_ir: float = 250,
                 half_s2: float = 1000) -> Dict[str, Optional[np.ndarray]]:
    """All six tiles for a site; each is HxWx3 float [0,1] or None."""
    tiles: Dict[str, Optional[np.ndarray]] = {}
    tiles["naip_false_ir"] = naip_false_ir(lat, lon, half_m=half_ir)
    s2 = s2_bands(lat, lon, half_m=half_s2)
    if s2 is None:
        tiles["s2_truecolor"] = tiles["s2_swir_geology"] = None
        tiles["iron_oxide"] = tiles["clay_ratio"] = None
    else:
        tiles["s2_truecolor"] = to_rgb([s2["red"], s2["green"], s2["blue"]])
        tiles["s2_swir_geology"] = to_rgb([s2["swir22"], s2["swir16"],
                                           s2["red"]])
        tiles["iron_oxide"] = ratio_to_rgb(s2["red"] / (s2["blue"] + 1e-6))
        tiles["clay_ratio"] = ratio_to_rgb(s2["swir16"] / (s2["swir22"] + 1e-6))
    tiles["hillshade"] = hillshade(lat, lon, half_m=half_s2)
    return tiles


# ------------------------------------------------------------- composition

def compose_panel(tiles: Sequence[Tuple[str, Optional[np.ndarray]]],
                  tile_px: int = 500, cols: int = 3, title: str = ""):
    """Labelled grid of tiles -> PIL Image. Missing tiles render as
    'unavailable' rather than failing the panel. Pure composition — no
    network — so it is unit-testable offline."""
    import math

    from PIL import Image, ImageDraw

    n = len(tiles)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cap = 20
    top = 26 if title else 0
    W, H = cols * tile_px, top + rows * (tile_px + cap)
    sheet = Image.new("RGB", (W, H), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((6, 6), title, fill=(255, 255, 255))
    for i, (label, arr) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = c * tile_px, top + r * (tile_px + cap)
        if arr is None:
            draw.rectangle([x, y, x + tile_px, y + tile_px], fill=(40, 40, 40))
            draw.text((x + 8, y + tile_px // 2), f"{label}: unavailable",
                      fill=(160, 160, 160))
        else:
            a = (np.clip(np.nan_to_num(arr, nan=0.0), 0, 1) * 255
                 ).astype("uint8")
            im = Image.fromarray(a).resize((tile_px, tile_px), Image.LANCZOS)
            # centre crosshair marks the site
            d = ImageDraw.Draw(im)
            m = tile_px // 2
            d.line([(m - 16, m), (m - 6, m)], fill=(255, 60, 60), width=2)
            d.line([(m + 6, m), (m + 16, m)], fill=(255, 60, 60), width=2)
            d.line([(m, m - 16), (m, m - 6)], fill=(255, 60, 60), width=2)
            d.line([(m, m + 6), (m, m + 16)], fill=(255, 60, 60), width=2)
            sheet.paste(im, (x, y))
        draw.text((x + 4, y + tile_px + 3), label[:70], fill=(230, 230, 230))
    return sheet


def filter_sheet(lat: float, lon: float, out_path, *, title: str = "",
                 tile_px: int = 500) -> Optional[Path]:
    """Fetch all views and write one dossier panel. Returns None only if
    EVERY tile failed (no point writing an all-grey panel)."""
    out_path = Path(out_path)
    views = gather_views(lat, lon)
    if all(v is None for v in views.values()):
        logger.info("filter_sheet: all tiles unavailable at %.4f,%.4f",
                    lat, lon)
        return None
    tiles = [(label, views[key]) for key, label in TILE_ORDER]
    sheet = compose_panel(tiles, tile_px=tile_px,
                          title=title or f"{lat:.4f}, {lon:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
