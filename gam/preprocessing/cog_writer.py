"""Per-tile warping/cropping and COG writing utilities.

- warp_crop_to_tile: Warp and crop a source raster to a 10°×10° tile at 0.1° resolution
  in EPSG:4326, returning a float32 2D array with NaN nodata, exactly 100×100 pixels.

- write_cog: Write a float32 NaN-nodata array to a Cloud-Optimized GeoTIFF (COG),
  using rio-cogeo if available, else a GTiff with appropriate tiling, compression,
  and overviews.

All paths handled via pathlib for Windows safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from affine import Affine

from gam.core.tiles import PIXELS_PER_TILE


def _resampling_from_str(name: str) -> Resampling:
    name_l = (name or "bilinear").strip().lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }
    return mapping.get(name_l, Resampling.bilinear)


def warp_crop_to_tile(
    input_path: str,
    tile_bounds: Tuple[float, float, float, float],
    dst_res: float = 0.1,
    dst_crs: str = "EPSG:4326",
    resampling: str = "bilinear",
) -> np.ndarray:
    """Warp + crop source raster to a fixed 10°×10° tile grid.

    Parameters:
    - input_path: path to source raster
    - tile_bounds: (minx, miny, maxx, maxy) of the 10° tile in EPSG:4326
    - dst_res: target pixel size in degrees (default 0.1°)
    - dst_crs: target CRS (default EPSG:4326)
    - resampling: resampling method name for continuous fields ('bilinear' default)

    Returns:
    - 2D numpy array of shape (100, 100), dtype float32, with NaN nodata
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input raster not found: {in_path}")

    minx, miny, maxx, maxy = tile_bounds
    rows = int(round((maxy - miny) / dst_res))
    cols = int(round((maxx - minx) / dst_res))
    if rows != PIXELS_PER_TILE or cols != PIXELS_PER_TILE:
        raise ValueError(
            f"Tile size mismatch: expected {PIXELS_PER_TILE}x{PIXELS_PER_TILE}, got {rows}x{cols}"
        )

    dst_transform: Affine = from_origin(minx, maxy, dst_res, dst_res)
    rs = _resampling_from_str(resampling)

    # Open and create a WarpedVRT aligned exactly to the tile grid and shape
    with rasterio.open(in_path) as src:
        vrt_opts = {
            "crs": dst_crs,
            "transform": dst_transform,
            "height": rows,
            "width": cols,
            "resampling": rs,
            "src_nodata": src.nodata,
            "dst_nodata": np.float32(np.nan),
        }
        with WarpedVRT(src, **vrt_opts) as vrt:
            data = vrt.read(1, out_dtype="float32", masked=True, resampling=rs)

    # Convert masked to NaN-filled float32
    arr = np.asarray(data.filled(np.nan), dtype="float32")
    return arr


def write_cog(
    output_path: str,
    array: np.ndarray,
    tile_bounds: Tuple[float, float, float, float],
    dst_res: float = 0.1,
    tags: Optional[Dict] = None,
) -> None:
    """Write a float32 NaN-nodata array as a COG, using rio-cogeo when available.

    Parameters:
    - output_path: destination file path (COG)
    - array: 2D float32 array (NaN nodata), expected 100×100
    - tile_bounds: (minx, miny, maxx, maxy) of the tile
    - dst_res: pixel size (0.1°)
    - tags: optional dict of tags/metadata to write
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim != 2:
        raise ValueError("Array must be 2D (rows, cols)")
    if array.shape[0] != PIXELS_PER_TILE or array.shape[1] != PIXELS_PER_TILE:
        raise ValueError(
            f"Array shape must be {PIXELS_PER_TILE}x{PIXELS_PER_TILE}, got {array.shape}"
        )

    minx, miny, maxx, maxy = tile_bounds
    transform = from_origin(minx, maxy, dst_res, dst_res)
    height, width = array.shape

    # Try rio-cogeo path first
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        from rasterio.io import MemoryFile

        profile = {
            **cog_profiles.get("deflate"),
            # Ensure requested tiling and compression profile
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "DEFLATE",
            "predictor": 2,
            "zlevel": 9,
            "bigtiff": "IF_SAFER",
        }

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=np.float32(np.nan),
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=2,
                zlevel=9,
            ) as mem_ds:
                mem_ds.write(array, 1)
                if tags:
                    mem_ds.update_tags(**tags)

            # Translate to a COG with internal overviews
            cog_translate(
                memfile,
                str(out_path),
                profile,
                nodata=np.float32(np.nan),
                overview_levels=[2, 4, 8, 16],
                overview_resampling="average",
                web_optimized=False,
                add_mask=False,
                quiet=True,
            )
        return
    except Exception:
        # Fallback to plain GTiff + overviews (near-COG). Silent import/runtime failure fallback.
        pass

    # Fallback writer using rasterio GTiff with requested creation options and overviews
    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.float32(np.nan),
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        predictor=2,
        zlevel=9,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="ALL_CPUS",
    ) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**tags)
        # Build power-of-two overviews for COG-style pyramids
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        # Copy overview resampling method hint
        dst.update_tags(OVERVIEW_RESAMPLING="AVERAGE")