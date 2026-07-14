"""Multi-temporal compositing of Sentinel-2 (Orengo & Petrie style).

Per-pixel temporal median over many same-season scenes cancels transient
conditions (this year's crop, moisture, plough state) and leaves persistent
soil/moisture signatures.

Measured honest performance (Upper Khabur, Menze-Ur ground truth):
- amplifies anthrosol contrast at LARGE tells (2.3x at Tell Brak)
- does NOT make BSI a general small-site detector (population AUC ~0.55)
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .data_access import read_grid, stac_search

S2_BANDS = ("red", "green", "blue", "nir", "swir16")

# SCL codes to KEEP (bare soil / vegetation / water / unclassified);
# masks clouds (8,9,10), shadow (3), snow (11), saturated (1), dark (2), nodata (0)
SCL_GOOD = (4, 5, 6, 7)


def find_scenes(bbox, season=("07-01", "10-31"),
                years=(2019, 2020, 2021, 2022, 2023, 2024),
                cloud_lt=15, max_scenes=30):
    """Low-cloud Sentinel-2 L2A scenes in a season window across years.
    STAC datetime is a single interval, so query per-year and concatenate."""
    feats = []
    for y in years:
        feats.extend(stac_search(
            "sentinel-2-l2a", bbox,
            datetime=f"{y}-{season[0]}T00:00:00Z/{y}-{season[1]}T23:59:59Z",
            query={"eo:cloud_cover": {"lt": cloud_lt}}, limit=50,
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]))
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    return feats[:max_scenes]


def scl_mask(scl: np.ndarray) -> np.ndarray:
    """Boolean keep-mask from a resampled SCL band."""
    return np.isin(np.nan_to_num(scl, nan=0).astype(int), SCL_GOOD)


def temporal_median(feats, grid, width, height, bands=S2_BANDS,
                    min_good_frac=0.2, max_used=30, log=print):
    """Per-pixel temporal median of each band over cloud-masked scenes.

    Returns (dict band->2D array, n_scenes_used); (None, n) if fewer than 3
    usable scenes.
    """
    from rasterio.enums import Resampling
    stacks: dict = {b: [] for b in bands}
    used = 0
    for f in feats:
        assets = f.get("assets", {})
        if not all(b in assets for b in bands) or "scl" not in assets:
            continue
        try:
            scl = read_grid(assets["scl"]["href"], grid, width, height,
                            resampling=Resampling.nearest)
            good = scl_mask(scl)
            if good.mean() < min_good_frac:
                continue
            for b in bands:
                g = read_grid(assets[b]["href"], grid, width, height)
                g[~good] = np.nan
                stacks[b].append(g.astype("float32"))
            used += 1
        except Exception as exc:
            log(f"  ! skip {f.get('id')}: {type(exc).__name__}")
        if used >= max_used:
            break
    if used < 3:
        return None, used
    return {b: np.nanmedian(np.stack(stacks[b]), axis=0) for b in bands}, used


def bsi(red, green, blue, nir, swir):
    """Bare-soil index — anthrosol (mudbrick-derived) soils read high."""
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + 1e-6)


def ndmi(nir, swir):
    """Moisture index — hollow-ways and ditches retain moisture."""
    return (nir - swir) / (nir + swir + 1e-6)


def local_anomaly(index_img: np.ndarray, background_sigma_px: float = 25) -> np.ndarray:
    """Index minus its large-scale background — isolates local features."""
    filled = np.nan_to_num(index_img, nan=float(np.nanmean(index_img)))
    return index_img - gaussian_filter(filled, background_sigma_px)
