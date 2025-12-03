#!/usr/bin/env python3
"""
Shared raster processing utilities for GeoAnomalyMapper.

This module contains common functions for clipping, reprojecting, and resampling
raster data to ensure consistent handling across different data types (gravity,
magnetic, DEM, InSAR, lithology).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

logger = logging.getLogger(__name__)


def clip_and_reproject_raster(
    input_path: Path,
    output_path: Path,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    target_crs: str = "EPSG:4326",
    is_categorical: bool = False,
) -> bool:
    """
    Clip input_path to bounds and resample it onto a common grid.

    Supports both continuous (bilinear) and categorical (nearest neighbor)
    data types.

    Args:
        input_path: Path to input raster file.
        output_path: Path for output raster file.
        bounds: Tuple of (minx, miny, maxx, maxy) bounding box.
        resolution: Output grid resolution in degrees.
        target_crs: Target coordinate reference system (default: EPSG:4326).
        is_categorical: If True, use nearest neighbor resampling for categorical data
                        (e.g., lithology classes). Otherwise, use bilinear for continuous
                        data (default: False).

    Returns:
        True if processing succeeded, False otherwise.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")

    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return False

    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bounds: {bounds}")

    width = max(1, int(round((maxx - minx) / resolution)))
    height = max(1, int(round((maxy - miny) / resolution)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    logger.info("Processing %s → %s", input_path.name, output_path.name)

    resampling_method = Resampling.nearest if is_categorical else Resampling.bilinear

    with rasterio.open(input_path) as src:
        dst_array = np.full((height, width), src.nodata or np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=resampling_method,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dst_array.dtype,
        crs=target_crs,
        transform=transform,
        nodata=np.nan,
        compress='DEFLATE',
        BIGTIFF='YES',
    ) as dst:
        dst.write(dst_array, 1)

    logger.info("Saved %s", output_path)
    return True