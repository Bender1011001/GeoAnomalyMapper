#!/usr/bin/env python3
"""
Shared raster processing utilities for GeoAnomalyMapper.

This module contains common functions for clipping, reprojecting, and resampling
raster data to ensure consistent handling across different data types (gravity,
magnetic, DEM, InSAR, lithology).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.merge import merge

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

    logger.info("Processing %s -> %s", input_path.name, output_path.name)


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

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

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


def parse_tile_bounds(tile_id: str) -> Tuple[float, float, float, float]:
    """
    Parse 1x1 degree tile ID (e.g., 'N48W090') to bounding box (min_lon, min_lat, max_lon, max_lat).
    
    Tiles are named by upper-left integer degree corner and span 1° south and east.
    """
    if len(tile_id) != 7 or not tile_id[0].isalpha() or tile_id[3] not in 'EW':
        raise ValueError(f"Invalid tile_id format: {tile_id}")
    
    ns = tile_id[0]
    ew = tile_id[3]
    
    try:
        lat_deg = int(tile_id[1:3])
        lon_deg = int(tile_id[4:7])
    except ValueError as e:
        raise ValueError(f"Invalid degrees in tile_id '{tile_id}': {e}")
    
    lat_ul = lat_deg if ns == 'N' else -lat_deg
    lon_ul = lon_deg if ew == 'E' else -lon_deg
    
    # Tile spans 1° south (lat-1 to lat) and east (lon-1 to lon)
    return (lon_ul - 1.0, lat_ul - 1.0, lon_ul, lat_ul)


def boxes_intersect(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> bool:
    """
    Check if two axis-aligned bounding boxes intersect.
    
    Boxes as (minx, miny, maxx, maxy).
    """
    left1, bottom1, right1, top1 = box1
    left2, bottom2, right2, top2 = box2
    return not (right1 < left2 or right2 < left1 or top1 < bottom2 or top2 < bottom1)


def mosaic_and_clip_seasonal(
    seasonal_dir: Path,
    season: str,
    region: Tuple[float, float, float, float],
    resolution: float,
    output_path: Path,
    metric: str = "COH12",
    pol: str = "vv"
) -> bool:
    """
    Find, mosaic (if needed), and clip seasonal coherence tiles intersecting the region.
    
    Supports single or multiple tiles; outputs clipped raster on target grid.
    
    Args:
        seasonal_dir: Base dir with tile subdirs (e.g., data/raw/insar/seasonal_usa).
        season: One of ['winter', 'spring', 'summer', 'fall'].
        region: (min_lon, min_lat, max_lon, max_lat).
        resolution: Target grid resolution (degrees).
        output_path: Output clipped GeoTIFF path.
        metric: Coherence metric ('COH12').
        pol: Polarization ('vv').
    
    Returns:
        True if successful (file written), False if no relevant tiles.
    """
    if resolution <= 0:
        logger.warning("Invalid resolution: %s", resolution)
        return False
    
    tile_dirs = [d for d in seasonal_dir.iterdir() if d.is_dir()]
    relevant_tiles: List[Path] = []
    
    minx, miny, maxx, maxy = region
    
    for tile_dir in tile_dirs:
        tile_id = tile_dir.name
        try:
            tile_box = parse_tile_bounds(tile_id)
        except ValueError:
            logger.debug("Skipping invalid tile: %s", tile_id)
            continue
        
        if boxes_intersect(tile_box, region):
            tile_file = tile_dir / f"{tile_id}_{season}_{pol}_{metric}.tif"
            if tile_file.exists():
                relevant_tiles.append(tile_file)
    
    if not relevant_tiles:
        logger.debug("No seasonal tiles found for %s/%s intersecting region", seasonal_dir.name, season)
        return False
    
    logger.info("Found %d seasonal tiles for %s: %s", len(relevant_tiles), season, [t.name for t in relevant_tiles])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(relevant_tiles) == 1:
        return clip_and_reproject_raster(relevant_tiles[0], output_path, region, resolution)
    
    # Mosaic multiple tiles
    sources = [rasterio.open(t) for t in relevant_tiles]
    try:
        src_crs = sources[0].crs
        mosaic_array, mosaic_transform = merge.merge(sources)
        mosaic_data = mosaic_array[0]  # Single band
        
        # Clip/reproject mosaic to target grid (reuse clip logic)
        width = max(1, int(round((maxx - minx) / resolution)))
        height = max(1, int(round((maxy - miny) / resolution)))
        dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
        dst_crs = "EPSG:4326"
        
        dst_array = np.full((height, width), np.nan, dtype=np.float32)
        
        reproject(
            source=(mosaic_data, mosaic_transform),
            destination=dst_array,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )
        
        # Write with optimized profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': dst_crs,
            'transform': dst_transform,
            'nodata': np.nan,
            'compress': 'DEFLATE',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        }
        if height * width * 4 > 4e9:
            profile['BIGTIFF'] = 'YES'
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_array, 1)
        
        logger.info("Mosaicked and clipped seasonal data: %s", output_path)
        return True
        
    finally:
        for src in sources:
            src.close()