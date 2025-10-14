#!/usr/bin/env python3
"""
Process NASADEM Elevation Data for Multi-Modal Anomaly Detection.

This script processes NASADEM tiles for a target region, extracts elevation from .hgt files,
creates a seamless mosaic, resamples to a uniform grid (0.0025° ~250m resolution to match
high-res gravity data), and outputs a GeoTIFF. Handles ZIP extraction if tiles are archived.
Nodata value: -32768 (voids in NASADEM).

Target region: Carlsbad Caverns, NM (lon: -105.0 to -104.0, lat: 32.0 to 33.0).
Output: data/processed/elevation/nasadem_processed.tif (EPSG:4326, meters ASL).

Usage:
    python process_nasadem_elevation.py

Dependencies:
    - rasterio (for raster I/O and warping)
    - numpy (for grid computations)
    - Install: pip install rasterio numpy

Author: GeoAnomalyMapper Pipeline
Version: 1.0
"""
import logging
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds, Affine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/outputs/processing.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
NASUDEM_DIR = Path('data/raw/elevation/nasadem')
PROCESSED_DIR = Path('data/processed/elevation')
OUTPUT_PATH = PROCESSED_DIR / 'nasadem_processed.tif'
TARGET_BBOX = (-123.0, 32.0, -114.0, 42.0)  # (lon_min, lat_min, lon_max, lat_max) - covering available California tiles
GRID_RES = 0.0025  # degrees (~250m at equator)
NODATA_VALUE = -32768
EXPECTED_ELEV_RANGE = (800, 1500)  # Approximate for Carlsbad Caverns (m ASL)

# Tile naming: NASADEM follows SRTM convention, e.g., 'n32w105.hgt' for N32°W105° tile
def generate_tile_names(bbox: Tuple[float, float, float, float]) -> List[str]:
    """
    Generate expected NASADEM tile names for the bounding box.

    Tiles are 1°x1°. For bbox spanning multiple tiles, return all covering ones.
    Format: 'n{lat:02d}w{lon:03d}' (lat N positive, lon W positive).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    for lat in range(int(lat_min), int(lat_max) + 1):
        for lon in range(int(-lon_max), int(-lon_min) + 1):  # W longitudes positive
            tile_name = f'n{lat:02d}w{abs(lon):03d}'
            tiles.append(tile_name)
    return tiles

def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> Optional[Path]:
    """
    Extract ZIP file containing .hgt if not already extracted.

    Returns path to .hgt file if successful, None otherwise.
    """
    if not zip_path.exists():
        return None

    hgt_name = zip_path.stem + '.hgt'  # e.g., NASADEM_HGT_n32w105.hgt
    hgt_path = extract_dir / hgt_name

    if hgt_path.exists():
        logger.info(f".hgt already extracted: {hgt_path}")
        return hgt_path

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        if hgt_path.exists():
            logger.info(f"Extracted .hgt from {zip_path} to {hgt_path}")
            return hgt_path
        else:
            logger.warning(f"No .hgt found in {zip_path}")
            return None
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP: {zip_path}")
        return None

def get_hgt_path(tile_name: str) -> Optional[Path]:
    """
    Find .hgt file for a tile: check extracted dir first, then ZIP.

    Priority: extracted > ZIP (extract if needed).
    """
    tile_dir = NASUDEM_DIR / tile_name
    hgt_in_dir = tile_dir / f'{tile_name}.hgt'

    if hgt_in_dir.exists():
        return hgt_in_dir

    # Check ZIP
    zip_name = f'NASADEM_HGT_{tile_name.upper()}.zip'
    zip_path = NASUDEM_DIR / zip_name

    if zip_path.exists():
        return extract_zip_if_needed(zip_path, tile_dir)

    logger.warning(f"No data found for tile: {tile_name} (ZIP or extracted)")
    return None

def read_hgt_as_dataset(hgt_path: Path) -> Optional[rasterio.DatasetReader]:
    """
    Read .hgt file as rasterio dataset with proper georeferencing.
    
    NASADEM .hgt: 3601x3601 (1 arcsec), int16, big-endian, upper-left origin.
    Transform: from tile bounds (lon_min to lon_max, lat_max to lat_min).
    Uses numpy to read binary if GDAL driver fails.
    """
    if not hgt_path.exists():
        return None

    tile_name = hgt_path.stem.lower()  # e.g., 'n32w105'
    lat_str, lon_str = tile_name[1:].split('w')
    lat = int(lat_str)
    lon = -int(lon_str)  # W negative

    # Tile bounds: 1°x1°, pixel size 1/3600 deg
    west, south = lon, lat - 1
    east, north = lon + 1, lat
    transform = from_bounds(west, south, east, north, 3601, 3601)
    width, height = 3601, 3601  # 1 arcsec global

    try:
        # First try GDAL HGT driver
        with rasterio.open(
            hgt_path,
            driver='HGT',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.int16,
            crs='EPSG:4326',
            transform=transform,
            nodata=NODATA_VALUE
        ) as src:
            # Verify data integrity
            data = src.read(1)
            if np.all(data == NODATA_VALUE):
                logger.warning(f"All nodata in {hgt_path}")
            return src
    except Exception as e:
        logger.warning(f"GDAL HGT driver failed for {hgt_path}: {e}. Trying numpy binary read.")

    # Fallback: Read as binary with numpy
    try:
        # HGT is big-endian int16, row-major, upper-left origin
        with open(hgt_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype='>i2', count=height * width)
            if len(raw_data) != height * width:
                raise ValueError(f"Incomplete file: expected {height*width}, got {len(raw_data)} values")
        
        data = raw_data.reshape((height, width)).astype(np.float32)
        data[data == NODATA_VALUE] = np.nan

        # Create in-memory rasterio dataset
        from rasterio.io import MemoryFile
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform,
                nodata=np.nan
            ) as mem:
                mem.write(data, 1)
                # Verify
                mem_data = mem.read(1)
                if np.all(np.isnan(mem_data)):
                    logger.warning(f"All nodata in {hgt_path} (numpy read)")
                return mem  # Return the memory dataset
    except Exception as e:
        logger.error(f"Numpy binary read failed for {hgt_path}: {e}")
        return None

def create_target_profile(bbox: Tuple[float, float, float, float], res: float) -> dict:
    """
    Create rasterio profile for target grid.

    Uniform grid over bbox at given resolution.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    width = int((lon_max - lon_min) / res) + 1
    height = int((lat_max - lat_min) / res) + 1
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    return {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,  # Output as float for resampling
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': np.nan,  # Use NaN for output nodata
        'compress': 'lzw',  # Compression for efficiency
        'tiled': True
    }

def process_elevation(bbox: Tuple[float, float, float, float], res: float) -> bool:
    """
    Main processing: mosaic tiles, clip, resample, validate.

    Returns True if successful (even if nodata-only).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tile_names = generate_tile_names(bbox)
    logger.info(f"Processing {len(tile_names)} potential tiles for bbox {bbox}")

    datasets: List[rasterio.DatasetReader] = []
    used_tiles = []

    for tile_name in tile_names:
        hgt_path = get_hgt_path(tile_name)
        if hgt_path:
            ds = read_hgt_as_dataset(hgt_path)
            if ds:
                # Clip to bbox early to reduce memory
                window = rasterio.windows.from_bounds(*bbox, transform=ds.transform)
                clipped = ds.read(1, window=window, masked=True)
                if not np.all(clipped.mask):
                    datasets.append(ds)
                    used_tiles.append(tile_name)
                    logger.info(f"Added tile {tile_name} (coverage: {(~clipped.mask).sum()} pixels)")
                else:
                    ds.close()
            if ds:
                ds.close()  # Ensure close even if clipped

    if not datasets:
        logger.warning("No valid tiles found for target region. Creating nodata template.")
        # Create empty raster
        profile = create_target_profile(bbox, res)
        with rasterio.open(OUTPUT_PATH, 'w', **profile) as dst:
            dst.write(np.full((profile['height'], profile['width']), np.nan, dtype=np.float32), 1)
        _validate_output(OUTPUT_PATH, bbox, res, used_tiles)
        return True

    # Mosaic
    try:
        mosaic, out_transform = merge(datasets, method='first', nodata=NODATA_VALUE)
        mosaic = np.where(mosaic == NODATA_VALUE, np.nan, mosaic.astype(np.float32))
        logger.info(f"Mosaicked {len(used_tiles)} tiles: shape {mosaic.shape}")
    finally:
        for ds in datasets:
            ds.close()

    # Resample to target grid
    profile = create_target_profile(bbox, res)
    dst_crs = rasterio.CRS.from_epsg(4326)
    src_transform = Affine.from_gdal(*out_transform)  # Assuming from rasterio.transform

    with rasterio.open(OUTPUT_PATH, 'w', **profile) as dst:
        warp.reproject(
            source=mosaic,
            destination=dst.read(1),
            src_transform=out_transform,
            src_crs=dst_crs,
            dst_transform=profile['transform'],
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

    _validate_output(OUTPUT_PATH, bbox, res, used_tiles)
    return True

def _validate_output(out_path: Path, bbox: Tuple[float, float, float, float], res: float, used_tiles: List[str]):
    """
    Validate output: bounds, resolution, stats, coverage.
    """
    with rasterio.open(out_path, 'r+') as src:
        data = src.read(1, masked=True)
        valid_pixels = (~data.mask).sum()
        total_pixels = data.size
        coverage_pct = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Check bounds and resolution
        actual_bounds = rasterio.transform.array_bounds(src.height, src.width, src.transform)
        expected_width = (bbox[2] - bbox[0]) / res
        actual_res_x = (actual_bounds[2] - actual_bounds[0]) / src.width
        logger.info(f"Output validated: bounds {actual_bounds}, res ~{actual_res_x:.4f}°")
        if abs(actual_res_x - res) > 1e-4:
            logger.warning(f"Resolution mismatch: expected {res}, got {actual_res_x}")

        # Elevation stats (ignore nodata)
        valid_data = data[~data.mask]
        if len(valid_data) > 0:
            min_elev, max_elev, mean_elev = valid_data.min(), valid_data.max(), valid_data.mean()
            logger.info(f"Elevation stats: min={min_elev:.1f}m, max={max_elev:.1f}m, mean={mean_elev:.1f}m")
            if not (EXPECTED_ELEV_RANGE[0] <= mean_elev <= EXPECTED_ELEV_RANGE[1]):
                logger.warning(f"Mean elevation {mean_elev:.1f}m outside expected range {EXPECTED_ELEV_RANGE}")
        else:
            logger.info("Output is entirely nodata (no coverage for region)")

        logger.info(f"Coverage: {coverage_pct:.1f}% ({valid_pixels}/{total_pixels} pixels)")
        logger.info(f"Used tiles: {used_tiles}")

        # Add metadata
        try:
            src.update_tags(
                title='NASADEM Processed Elevation Mosaic',
                description=f'Processed {datetime.now().isoformat()}. Tiles: {", ".join(used_tiles)}',
                bbox=str(bbox),
                resolution=f'{res} degrees',
                units='meters above sea level',
                nodata=str(np.nan)
            )
            logger.info("Metadata updated successfully")
        except Exception as e:
            logger.warning(f"Failed to update metadata: {e}")

if __name__ == '__main__':
    success = process_elevation(TARGET_BBOX, GRID_RES)
    if success:
        logger.info(f"Elevation processing complete. Output: {OUTPUT_PATH}")
    else:
        logger.error("Elevation processing failed.")
        sys.exit(1)