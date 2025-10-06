#!/usr/bin/env python3
"""
Global Anomaly Fusion Pipeline.

Processes global magnetic and gravity datasets into tiled, normalized, and fused anomaly maps.
Outputs Cloud-Optimized GeoTIFFs per tile and a final PMTiles file for web visualization.

Usage: python process_global_map.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gam.core.tiles import tiles_10x10_ids, tile_bounds_10x10
from gam.preprocessing.cog_writer import warp_crop_to_tile, write_cog
from gam.modeling.fuse_simple import robust_z, fuse_layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Data paths (update if needed)
MAGNETIC_PATH = Path('data/raw/EMAG2_V3_Sea_Level.tif')
GRAVITY_PATH = Path('data/raw/EGM2008_Free_Air_Anomaly.tif')

# Output directories
OUTPUT_BASE = Path('data/outputs/cog')
MAG_OUTPUT_DIR = OUTPUT_BASE / 'mag'
GRAV_OUTPUT_DIR = OUTPUT_BASE / 'grav'
FUSED_OUTPUT_DIR = OUTPUT_BASE / 'fused'

# Ensure output directories exist
MAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FUSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_tile(tile_id: str) -> None:
    """Process a single 10°×10° tile: warp, normalize, fuse, and write outputs."""
    logger.info(f"Processing tile {tile_id}")

    # Get tile bounds
    minx, miny, maxx, maxy = tile_bounds_10x10(tile_id)

    # Warp magnetic data
    try:
        mag_array = warp_crop_to_tile(str(MAGNETIC_PATH), (minx, miny, maxx, maxy))
        logger.debug(f"Magnetic array shape: {mag_array.shape}")
    except Exception as e:
        logger.error(f"Failed to warp magnetic data for {tile_id}: {e}")
        return

    # Warp gravity data
    try:
        grav_array = warp_crop_to_tile(str(GRAVITY_PATH), (minx, miny, maxx, maxy))
        logger.debug(f"Gravity array shape: {grav_array.shape}")
    except Exception as e:
        logger.error(f"Failed to warp gravity data for {tile_id}: {e}")
        return

    # Normalize magnetic
    mag_norm = robust_z(mag_array)

    # Normalize gravity
    grav_norm = robust_z(grav_array)

    # Fuse layers
    fused = fuse_layers(mag_norm, grav_norm)

    # Write outputs
    tile_bounds = (minx, miny, maxx, maxy)

    mag_out_path = MAG_OUTPUT_DIR / f"{tile_id}.tif"
    write_cog(str(mag_out_path), mag_norm, tile_bounds)

    grav_out_path = GRAV_OUTPUT_DIR / f"{tile_id}.tif"
    write_cog(str(grav_out_path), grav_norm, tile_bounds)

    fused_out_path = FUSED_OUTPUT_DIR / f"{tile_id}.tif"
    write_cog(str(fused_out_path), fused, tile_bounds)

    logger.info(f"Completed tile {tile_id}")


def main() -> None:
    """Main processing loop."""
    logger.info("Starting global anomaly fusion pipeline")

    # Get all tile IDs
    tile_ids = tiles_10x10_ids()
    logger.info(f"Processing {len(tile_ids)} tiles")

    # Process each tile with progress bar
    for tile_id in tqdm(tile_ids, desc="Processing tiles"):
        process_tile(tile_id)

    logger.info("Tile processing complete. Generating final PMTiles...")

    # Create VRT from fused tiles
    vrt_path = Path('fused_anomaly.vrt')
    fused_tiles = list(FUSED_OUTPUT_DIR.glob('*.tif'))
    if not fused_tiles:
        logger.error("No fused tiles found!")
        return

    # Use gdalbuildvrt (assumes GDAL is installed)
    import subprocess
    cmd_vrt = ['gdalbuildvrt', str(vrt_path)] + [str(p) for p in fused_tiles]
    try:
        subprocess.run(cmd_vrt, check=True)
        logger.info(f"Created VRT: {vrt_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create VRT: {e}")
        return

    # Convert to PMTiles (assumes pmtiles is installed)
    pmtiles_path = Path('fused_anomaly.pmtiles')
    cmd_pmtiles = ['pmtiles', 'convert', str(vrt_path), str(pmtiles_path)]
    try:
        subprocess.run(cmd_pmtiles, check=True)
        logger.info(f"Created PMTiles: {pmtiles_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create PMTiles: {e}")
        return

    logger.info("Pipeline complete!")


if __name__ == '__main__':
    main()