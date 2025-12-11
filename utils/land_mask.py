#!/usr/bin/env python3
"""
Land Masking Utility for GeoAnomalyMapper
==========================================

Provides functionality to create land/ocean masks from Natural Earth data.
This module is designed to be used throughout the pipeline to exclude ocean
areas from processing, improving computational efficiency and reducing false
positives in mineral exploration applications.

Key Features:
- Loads Natural Earth land boundaries
- Rasterizes to match any input raster grid
- Handles CRS reprojection automatically
- Provides both boolean masks and masked data arrays

Usage:
    from utils.land_mask import create_land_mask, apply_land_mask
    
    with rasterio.open("input.tif") as src:
        mask = create_land_mask(src)
        masked_data = apply_land_mask(src.read(1), mask)
"""

import logging
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def create_land_mask(
    src: rasterio.DatasetReader,
    buffer_deg: float = 0.0,
    cache_key: Optional[str] = None
) -> np.ndarray:
    """
    Create a boolean land mask where Land=True, Ocean=False.
    
    Uses Natural Earth low-resolution land boundaries (1:110m scale).
    Automatically reprojects to match the input raster's CRS.
    
    Args:
        src: Open rasterio dataset reader
        buffer_deg: Buffer distance in degrees (positive=expand land, negative=erode)
        cache_key: Optional cache identifier (future enhancement)
        
    Returns:
        Boolean numpy array of shape (height, width) where True=land, False=ocean
        
    Raises:
        RuntimeError: If Natural Earth data cannot be loaded
        
    Example:
        >>> with rasterio.open("gravity.tif") as src:
        ...     land_mask = create_land_mask(src)
        ...     land_pixels = np.sum(land_mask)
        ...     logger.info(f"Land coverage: {100*land_pixels/land_mask.size:.1f}%")
    """
    logger.info("Creating land mask from Natural Earth boundaries...")
    
    # Load Natural Earth land boundaries
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except (AttributeError, FileNotFoundError):
        # Fallback: Download from Natural Earth CDN
        logger.warning("Local Natural Earth data not found, downloading from CDN...")
        try:
            world = gpd.read_file("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip")
        except Exception as e:
            raise RuntimeError(
                "Failed to load Natural Earth land boundaries. "
                "Install geopandas with: pip install geopandas\n"
                f"Error: {e}"
            ) from e
    
    # Get raster bounds and create bounding box
    bounds = src.bounds
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Reproject world geometries to match raster CRS if needed
    if str(src.crs) != str(world.crs):
        logger.info(f"Reprojecting land boundaries from {world.crs} to {src.crs}...")
        world = world.to_crs(src.crs)
    
    # Clip land polygons to raster bounds for efficiency
    land = world.clip(bbox)
    
    # Check if any land exists in this region
    if land.empty:
        logger.warning(
            f"No land found within raster bounds {bounds}. "
            "This region may be entirely ocean. Returning all-False mask."
        )
        return np.zeros((src.height, src.width), dtype=bool)
    
    # Apply buffer if requested (useful for coastal handling)
    if buffer_deg != 0.0:
        logger.info(f"Applying {buffer_deg:.4f}° buffer to land boundaries...")
        land.geometry = land.geometry.buffer(buffer_deg)
    
    # Rasterize land polygons to match raster grid
    logger.info(f"Rasterizing land mask to {src.width}×{src.height} grid...")
    mask = features.rasterize(
        shapes=land.geometry,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,           # Background (ocean) = 0
        default_value=1,  # Foreground (land) = 1
        dtype=rasterio.uint8
    )
    
    land_pixels = np.sum(mask)
    land_fraction = 100.0 * land_pixels / mask.size
    logger.info(f"Land mask created: {land_pixels:,} land pixels ({land_fraction:.1f}% coverage)")
    
    return mask.astype(bool)


def apply_land_mask(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = np.nan,
    inplace: bool = False
) -> np.ndarray:
    """
    Apply land mask to data array, setting ocean pixels to fill_value.
    
    Args:
        data: Input data array (2D or 3D with bands as first dimension)
        mask: Boolean mask where True=land, False=ocean
        fill_value: Value to assign to ocean pixels (default: NaN)
        inplace: If True, modifies data array directly; otherwise creates copy
        
    Returns:
        Masked data array with ocean pixels set to fill_value
        
    Example:
        >>> gravity_data = src.read(1)
        >>> land_mask = create_land_mask(src)
        >>> masked_gravity = apply_land_mask(gravity_data, land_mask, fill_value=0.0)
    """
    if not inplace:
        data = data.copy()
    
    # Handle multi-band data
    if data.ndim == 3:
        # Broadcast mask across all bands
        for i in range(data.shape[0]):
            data[i, ~mask] = fill_value
    elif data.ndim == 2:
        data[~mask] = fill_value
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")
    
    ocean_pixels = np.sum(~mask)
    logger.debug(f"Masked {ocean_pixels:,} ocean pixels with value {fill_value}")
    
    return data


def get_land_fraction(mask: np.ndarray) -> float:
    """
    Calculate the fraction of land pixels in a mask.
    
    Args:
        mask: Boolean land mask
        
    Returns:
        Fraction of True pixels (0.0 to 1.0)
    """
    return np.sum(mask) / mask.size


def mask_raster_file(
    input_path: str,
    output_path: str,
    fill_value: float = 0.0,
    buffer_deg: float = 0.0
) -> Tuple[int, int]:
    """
    Convenience function to mask an entire raster file.
    
    Reads input raster, creates land mask, applies it, and saves result.
    
    Args:
        input_path: Path to input raster
        output_path: Path to output masked raster
        fill_value: Value for ocean pixels
        buffer_deg: Buffer distance in degrees
        
    Returns:
        Tuple of (land_pixels, ocean_pixels)
        
    Example:
        >>> mask_raster_file("gravity.tif", "gravity_masked.tif", fill_value=np.nan)
    """
    with rasterio.open(input_path) as src:
        logger.info(f"Masking {input_path}...")
        
        # Read data and metadata
        data = src.read(1)
        profile = src.profile.copy()
        
        # Create and apply mask
        land_mask = create_land_mask(src, buffer_deg=buffer_deg)
        masked_data = apply_land_mask(data, land_mask, fill_value=fill_value)
        
        # Save masked raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(masked_data, 1)
        
        land_pixels = np.sum(land_mask)
        ocean_pixels = np.sum(~land_mask)
        logger.info(f"Saved masked raster to {output_path}")
        
        return land_pixels, ocean_pixels
