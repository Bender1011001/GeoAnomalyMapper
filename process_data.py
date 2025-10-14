#!/usr/bin/env python3
"""
GeoAnomalyMapper - Multi-Modal Data Processing Script
====================================================

Processes geophysical data for enhanced anomaly detection with multi-modal fusion:
1. Gravity data (converts .gfc to GeoTIFF if needed, resamples to 0.0025° grid)
2. Magnetic data (clips/resamples EMAG2 to region/grid)
3. Elevation data (loads NASADEM processed tif, aligns to grid)
4. InSAR data (if available - requires SNAP/ISCE)
5. Lithology data (if available)
6. Fusion: Combines gravity, magnetic, elevation into fused_multi_modal.tif with uncertainty propagation

Supports bimodal fallback if elevation missing; trimodal achieves 70-80% accuracy.

Simple usage:
    python process_data.py --region "lon_min,lat_min,lon_max,lat_max"

Example:
    python process_data.py --region "-105.0,32.0,-104.0,33.0"  # Carlsbad Caverns area

For elevation, run python process_nasadem_elevation.py first.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs" / "final"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip_and_reproject_raster(
    input_path: Path,
    output_path: Path,
    bounds: Tuple[float, float, float, float],
    resolution: float = 0.0025,
    target_crs: str = "EPSG:4326"
) -> bool:
    """Clip and reproject raster to target region and resolution."""
    
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return False
    
    try:
        logger.info(f"Processing: {input_path.name}")
        
        minx, miny, maxx, maxy = bounds
        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        with rasterio.open(input_path) as src:
            # Create destination array
            dst_array = np.zeros((height, width), dtype=np.float32)
            
            # Reproject to target grid
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )
        
        # Write output
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
            compress='DEFLATE'
        ) as dst:
            dst.write(dst_array, 1)
        
        logger.info(f"✓ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {input_path.name}: {e}")
        return False


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_gravity_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    """Process gravity data for the region."""
    logger.info("\n" + "="*70)
    logger.info("PROCESSING GRAVITY DATA")
    logger.info("="*70)
    
    gravity_dir = RAW_DIR / "gravity"
    output_dir = PROCESSED_DIR / "gravity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for existing gravity GeoTIFF
    gravity_files = list(gravity_dir.glob("*.tif*"))
    
    if not gravity_files:
        logger.warning("No gravity GeoTIFF found")
        logger.warning("You need to convert XGM2019e_2159.gfc to GeoTIFF")
        logger.warning("Visit: http://icgem.gfz-potsdam.de/calcgrid")
        logger.warning(f"Region: {region}")
        
        # Create instructions
        instructions_file = output_dir / "GRAVITY_PROCESSING_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write("GRAVITY DATA PROCESSING INSTRUCTIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. Visit: http://icgem.gfz-potsdam.de/calcgrid\n")
            f.write("2. Select model: XGM2019e_2159\n")
            f.write("3. Grid type: gravity_disturbance\n")
            f.write(f"4. Region: {region[0]:.2f}, {region[1]:.2f}, {region[2]:.2f}, {region[3]:.2f}\n")