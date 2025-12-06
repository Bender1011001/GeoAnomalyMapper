#!/usr/bin/env python3
"""
Phase 8b: Land Masking Utility.

1. Loads the anomaly map (Spatial or Point-based).
2. Downloads/Loads a North America land boundary vector.
3. Rasterizes the land boundary to match the anomaly map grid.
4. Masks out all pixels that fall outside the land boundary (i.e., the ocean).
5. Saves the cleaned map for re-extraction.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box

from project_paths import OUTPUTS_DIR
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_land_mask(src, buffer_deg=0.1):
    """
    Creates a boolean mask where Land = True, Water = False.
    Uses GeoPandas Natural Earth data.
    """
    logger.info("Loading land boundaries...")
    
    # 1. Load World Map
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except AttributeError:
        # Newer geopandas versions might deprecate get_path, fallback or direct load
        # Trying a robust fallback if built-in fails, usually it works.
        try:
            world = gpd.read_file("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip")
        except:
            logger.error("Could not load Natural Earth data. Please install geopandas properly.")
            return None

    # 2. Filter for North America to speed it up
    # (Or just clip to the raster bounds)
    
    # Get Raster Bounds
    bounds = src.bounds
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Reproject world to match raster if needed (usually 4326 for these scales)
    # Assuming raster is likely 4326 or close. If raster is projected, we project world to it.
    if src.crs != world.crs:
        logger.info(f"Reprojecting land mask to {src.crs}...")
        world = world.to_crs(src.crs)

    # 3. Clip Land to Raster Bounds
    # This keeps the vector operation fast
    land = world.clip(bbox)
    
    if land.empty:
        logger.warning("No land found in this raster's bounds! Check coordinates.")
        return np.ones((src.height, src.width), dtype=bool) # Fallback: Mask nothing

    logger.info("Rasterizing land mask...")
    
    # 4. Rasterize
    # Create a blank array of the same shape
    mask = features.rasterize(
        shapes=land.geometry,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,      # Background (Water) = 0
        default_value=1, # Foreground (Land) = 1
        dtype=rasterio.uint8
    )
    
    return mask.astype(bool)

def main():
    parser = argparse.ArgumentParser(description="Phase 8b: Ocean Masking")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", type=str, help="Input anomaly map (overrides config)")
    parser.add_argument("--output", type=str, help="Output masked map")
    args = parser.parse_args()

    # Determine Input Path
    input_path = None
    
    # 1. Try explicit argument
    if args.input:
        input_path = Path(args.input)
    
    # 2. Try Config
    if input_path is None:
        try:
            config = load_config(args.config)
            fname = config.get('output', {}).get('anomaly_map', 'spatial_anomaly_v1.tif')
            input_path = OUTPUTS_DIR / fname
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            
    # 3. Fallback
    if input_path is None:
        input_path = OUTPUTS_DIR / "spatial_anomaly_v1.tif"

    # Determine Output Path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: append _masked
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_masked{input_path.suffix}"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    with rasterio.open(input_path) as src:
        logger.info(f"Processing {input_path.name}...")
        
        # Read Anomaly Data
        data = src.read(1)
        profile = src.profile.copy()
        
        # Generate Mask
        land_mask = create_land_mask(src)
        
        if land_mask is None:
            return

        # Apply Mask
        # Where mask is False (Water), set data to 0.0 (Normal)
        logger.info("Applying ocean mask...")
        masked_data = np.where(land_mask, data, 0.0)
        
        # Save
        logger.info(f"Saving masked output to {args.output}")
        with rasterio.open(args.output, 'w', **profile) as dst:
            dst.write(masked_data, 1)
            
    logger.info("Done. The ocean is now silent.")

if __name__ == "__main__":
    main()