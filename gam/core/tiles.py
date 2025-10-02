"""Global 10°×10° tiling utilities and 0.1° grid spec (EPSG:4326).

Tile IDs follow: t_{LAT}_{LON}
- LAT bands: S90, S80, ..., S10, N00, N10, ..., N80
- LON bands: W180, W170, ..., W10, E000, E010, ..., E170

Examples:
- t_N20_E120 covers lon [120,130), lat [20,30)
- t_S10_W010 covers lon [-10,0), lat [-10,0)

All math is exact on degree boundaries and aligns with a 0.1° pixel grid.

Public API:
- tiles_10x10_ids() -> list[str]
- tile_bounds_10x10(tile_id: str) -> (minx, miny, maxx, maxy)
- grid_spec_0p1() -> dict
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import re

from rasterio.transform import from_origin
from affine import Affine

# Constants
TILE_SIZE_DEG: float = 10.0
RESOLUTION_0P1: float = 0.1
PIXELS_PER_TILE: int = int(round(TILE_SIZE_DEG / RESOLUTION_0P1))  # 100
WORLD_BOUNDS: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)

_TILE_ID_RE = re.compile(r"^t_([NS])(\d{2})_([EW])(\d{3})$")


def _lat_label_from_min(min_lat: float) -> str:
    """Format latitude band label from band minimum latitude in degrees."""
    if min_lat % 10 != 0:
        raise ValueError("Latitude band minimum must be multiple of 10 degrees")
    if min_lat < 0:
        return f"S{int(abs(min_lat)):02d}"
    else:
        return f"N{int(min_lat):02d}"


def _lon_label_from_min(min_lon: float) -> str:
    """Format longitude band label from band minimum longitude in degrees."""
    if min_lon % 10 != 0:
        raise ValueError("Longitude band minimum must be multiple of 10 degrees")
    if min_lon < 0:
        return f"W{int(abs(min_lon)):03d}"
    else:
        return f"E{int(min_lon):03d}"


def _lat_bands_10deg() -> List[Tuple[str, float, float]]:
    """Return list of (label, miny, maxy) for 10° latitude bands covering [-90,90)."""
    bands: List[Tuple[str, float, float]] = []
    # Latitude bands: [-90,-80), ..., [-10,0), [0,10), ..., [80,90)
    for miny in range(-90, 90, 10):
        maxy = miny + 10
        if maxy > 90:
            break
        label = _lat_label_from_min(float(miny))
        bands.append((label, float(miny), float(maxy)))
    return bands


def _lon_bands_10deg() -> List[Tuple[str, float, float]]:
    """Return list of (label, minx, maxx) for 10° longitude bands covering [-180,180)."""
    bands: List[Tuple[str, float, float]] = []
    # Longitude bands: [-180,-170), ..., [-10,0), [0,10), ..., [170,180)
    for minx in range(-180, 180, 10):
        maxx = minx + 10
        if maxx > 180:
            break
        label = _lon_label_from_min(float(minx))
        bands.append((label, float(minx), float(maxx)))
    return bands


def tiles_10x10_ids() -> List[str]:
    """Enumerate all 10°×10° tile IDs in deterministic order.

    Order:
    - Latitude from S90 up to N80 (increasing min latitude)
    - Longitude from W180 up to E170 (increasing min longitude)
    """
    ids: List[str] = []
    for lat_label, _, _ in _lat_bands_10deg():
        for lon_label, _, _ in _lon_bands_10deg():
            ids.append(f"t_{lat_label}_{lon_label}")
    return ids


def _parse_tile_id(tile_id: str) -> Tuple[str, int, str, int]:
    """Validate and parse a tile id into components."""
    m = _TILE_ID_RE.match(tile_id)
    if not m:
        raise ValueError(
            "Malformed tile_id. Expected pattern 't_[NS]\\d{2}_[EW]\\d{3}', "
            "e.g., 't_N20_E120'"
        )
    lat_hemi, lat_deg_s, lon_hemi, lon_deg_s = m.groups()
    lat_deg = int(lat_deg_s)
    lon_deg = int(lon_deg_s)

    # Validate ranges and multiples of 10
    if lat_deg % 10 != 0 or lon_deg % 10 != 0:
        raise ValueError("Degrees in tile_id must be multiples of 10")
    if lat_deg > 90 or lon_deg > 180:
        raise ValueError("Latitude degrees must be ≤ 90, longitude degrees must be ≤ 180")
    # Additional validity: min latitude must be within [-90, 80]; min longitude within [-180, 170]
    if lat_hemi == "N" and lat_deg > 80:
        raise ValueError("N latitude band must be ≤ N80 for 10° tiles")
    if lat_hemi == "S" and lat_deg > 90:
        raise ValueError("S latitude band must be ≤ S90 for 10° tiles")
    if lon_hemi == "E" and lon_deg > 170:
        raise ValueError("E longitude band must be ≤ E170 for 10° tiles")
    if lon_hemi == "W" and lon_deg > 180:
        raise ValueError("W longitude band must be ≤ W180 for 10° tiles")

    return lat_hemi, lat_deg, lon_hemi, lon_deg


def tile_bounds_10x10(tile_id: str) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) for a 10°×10° tile id.

    Example:
    - t_N00_E000 -> (0, 0, 10, 10)
    - t_S10_W010 -> (-10, -10, 0, 0)
    - t_N20_E120 -> (120, 20, 130, 30)
    """
    lat_hemi, lat_deg, lon_hemi, lon_deg = _parse_tile_id(tile_id)

    miny = float(lat_deg if lat_hemi == "N" else -lat_deg)
    maxy = miny + TILE_SIZE_DEG

    minx = float(lon_deg if lon_hemi == "E" else -lon_deg)
    maxx = minx + TILE_SIZE_DEG

    # Final bounds check within world
    wminx, wminy, wmaxx, wmaxy = WORLD_BOUNDS
    if not (wminx <= minx < maxx <= wmaxx) or not (wminy <= miny < maxy <= wmaxy):
        raise ValueError(f"Tile {tile_id} bounds outside world extents")

    return (minx, miny, maxx, maxy)


def grid_spec_0p1() -> Dict:
    """Return 0.1° global grid spec for EPSG:4326.

    Includes:
    - transform (Affine): origin at (-180, 90), pixel size 0.1° (north-up)
    - pixel_size: (0.1, 0.1)
    - bounds: (-180, -90, 180, 90)
    - width, height: (3600, 1800)
    - crs: "EPSG:4326"
    """
    minx, miny, maxx, maxy = WORLD_BOUNDS
    pixel_size = (RESOLUTION_0P1, RESOLUTION_0P1)
    transform: Affine = from_origin(minx, maxy, RESOLUTION_0P1, RESOLUTION_0P1)
    width = int(round((maxx - minx) / RESOLUTION_0P1))
    height = int(round((maxy - miny) / RESOLUTION_0P1))
    return {
        "transform": transform,
        "pixel_size": pixel_size,
        "bounds": WORLD_BOUNDS,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }


__all__ = [
    "tiles_10x10_ids",
    "tile_bounds_10x10",
    "grid_spec_0p1",
    "TILE_SIZE_DEG",
    "RESOLUTION_0P1",
    "PIXELS_PER_TILE",
    "WORLD_BOUNDS",
]