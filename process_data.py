#!/usr/bin/env python3
"""
GeoAnomalyMapper - Unified Data Processing Script
===================================================

Processes all downloaded geophysical data into a unified format ready for void detection:
1. Gravity data (converts .gfc to GeoTIFF if needed)
2. Magnetic data (clips to region)
3. Elevation data (auto-unzips, merges tiles, clips to region)
4. InSAR data (if available - requires SNAP/ISCE)
5. Lithology data (if available)

Simple usage:
    python process_data.py --region "lon_min,lat_min,lon_max,lat_max"
    
Example:
    python process_data.py --region "-105.0,32.0,-104.0,33.0"  # Carlsbad Caverns area
"""

import argparse
import logging
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import json

# --- Gdal (osgeo) is needed for VRT merging ---
# It's a dependency of rasterio, so it should be installed
try:
    from osgeo import gdal
except ImportError:
    print("Error: GDAL Python bindings not found.")
    print("Please ensure 'rasterio' and its dependencies are installed correctly:")
    print("pip install rasterio")
    sys.exit(1)
# ----------------------------------------------


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR
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
    resolution: float = 0.001,
    target_crs: str = "EPSG:4326"
) -> bool:
    """Clip and reproject raster to target region and resolution."""
    
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return False
    
    try:
        logger.info(f"Processing: {input_path.name}")
        
        minx, miny, maxx, maxy = bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
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

def process_gravity_data(region: Tuple[float, float, float, float]) -> bool:
    """Process gravity data for the region."""
    logger.info("\n" + "="*70)
    logger.info("PROCESSING GRAVITY DATA")
    logger.info("="*70)
    
    gravity_dir = RAW_DIR / "gravity"
    output_dir = PROCESSED_DIR / "gravity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for existing gravity GeoTIFF (e.g., EGM2008)
    gravity_files = list(gravity_dir.glob("*.tif*"))
    
    if not gravity_files:
        logger.warning("No gravity GeoTIFF found.")
        # Check for GFC (XGM2019e)
        if list(gravity_dir.glob("*.gfc")):
            logger.warning("Found .gfc file, but this requires manual conversion.")
            logger.warning("Visit: http://icgem.gfz-potsdam.de/calcgrid")
            logger.warning(f"Region: {region}")
        
        # Create instructions
        instructions_file = output_dir / "GRAVITY_PROCESSING_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write("GRAVITY DATA PROCESSING INSTRUCTIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. No usable GeoTIFF found in 'data/raw/gravity/'\n")
            f.write("2. If you have a .gfc file, convert it manually:\n")
            f.write("   - Visit: http://icgem.gfz-potsdam.de/calcgrid\n")
            f.write("   - Select model (e.g., XGM2019e_2159)\n")
            f.write("   - Select grid type (e.g., gravity_disturbance)\n")
            f.write(f"  - Set region: {region[0]:.2f}, {region[1]:.2f}, {region[2]:.2f}, {region[3]:.2f}\n")
            f.write("   - Set grid step (e.g., 0.01) and output GeoTIFF\n")
            f.write("3. Save the file to: data/raw/gravity/\n")
        
        logger.info(f"✓ Instructions saved: {instructions_file}")
        return False
    
    # Process existing gravity file
    gravity_file = gravity_files[0]
    output_file = output_dir / "gravity_processed.tif"
    
    return clip_and_reproject_raster(gravity_file, output_file, region)


def process_magnetic_data(region: Tuple[float, float, float, float]) -> bool:
    """Process magnetic anomaly data for the region."""
    logger.info("\n" + "="*70)
    logger.info("PROCESSING MAGNETIC DATA")
    logger.info("="*70)
    
    # Look in both 'magnetic' and 'emag2' folders, now recursively
    magnetic_dir = RAW_DIR / "magnetic"
    emag2_dir = RAW_DIR / "emag2"
    output_dir = PROCESSED_DIR / "magnetic"
    
    # Use rglob to find .tif files recursively in both directories
    magnetic_files = list(magnetic_dir.rglob("*.tif*")) + list(emag2_dir.rglob("*.tif*"))
    
    if not magnetic_files:
        logger.warning("No magnetic GeoTIFF found (e.g., EMAG2).")
        logger.warning("Please download 'EMAG2_V3_SeaLevel_DataTiff.tif'")
        logger.warning("from NOAA/NCEI and place it in 'data/raw/magnetic/'.")
        return False

    magnetic_file = magnetic_files[0]
    logger.info(f"Found magnetic data: {magnetic_file}")
    output_file = output_dir / "magnetic_processed.tif"
    
    return clip_and_reproject_raster(magnetic_file, output_file, region)


def process_dem_data(region: Tuple[float, float, float, float]) -> bool:
    """
    *** UPGRADED FUNCTION ***
    Auto-unzip, merge, and process elevation data for the region.
    """
    logger.info("\n" + "="*70)
    logger.info("PROCESSING ELEVATION DATA (UPGRADED)")
    logger.info("="*70)
    
    dem_dir = RAW_DIR / "elevation" / "nasadem" # Specific to your data
    output_dir = PROCESSED_DIR / "dem"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Unzip all .zip files ---
    zip_files = list(dem_dir.glob("*.zip"))
    if zip_files:
        logger.info(f"Found {len(zip_files)} DEM .zip files. Unzipping...")
        for zf_path in zip_files:
            try:
                with zipfile.ZipFile(zf_path, 'r') as zf:
                    # Extract to a subdirectory named after the zip
                    extract_path = dem_dir / zf_path.stem
                    extract_path.mkdir(exist_ok=True)
                    zf.extractall(extract_path)
                logger.info(f"  Unzipped: {zf_path.name}")
                # Keep the zip file for archive, don't unlink
                # zf_path.unlink() 
            except Exception as e:
                logger.warning(f"  Failed to unzip {zf_path.name}: {e}")
        logger.info("Unzipping complete.")
    
    # --- Step 2: Find all .hgt files ---
    hgt_files = list(dem_dir.glob("**/*.hgt"))
    
    if not hgt_files:
        # Fallback to check for TIFs in a generic 'dem' folder
        generic_dem_dir = RAW_DIR / "dem"
        tif_files = list(generic_dem_dir.glob("*.tif"))
        if tif_files:
            logger.info("Found standard DEM .tif file. Using that.")
            dem_file = tif_files[0]
            output_file = output_dir / "dem_processed.tif"
            return clip_and_reproject_raster(dem_file, output_file, region, resolution=0.0001)
        
        logger.warning("No DEM files found (.hgt or .tif)")
        logger.warning("Download Copernicus DEM 30m or NASADEM tiles for your region")
        return False

    logger.info(f"Found {len(hgt_files)} .hgt tiles.")

    # --- Step 3: Build a Virtual Raster (VRT) to merge them ---
    vrt_path = output_dir / "nasadem_merged.vrt"
    try:
        logger.info("Building virtual raster (VRT) to merge tiles...")
        gdal.BuildVRT(
            str(vrt_path),
            [str(f) for f in hgt_files],
            options=gdal.BuildVRTOptions(
                srcNodata=-32768,  # Standard SRTM void value
                VRTNodata=-32768
            )
        )
        logger.info(f"✓ VRT created: {vrt_path}")
    except Exception as e:
        logger.error(f"✗ Failed to build VRT: {e}")
        return False
    
    # --- Step 4: Process the merged VRT ---
    output_file = output_dir / "dem_processed.tif"
    return clip_and_reproject_raster(vrt_path, output_file, region, resolution=0.0001)


def process_insar_data(region: Tuple[float, float, float, float]) -> bool:
    """Process InSAR data if available."""
    logger.info("\n" + "="*70)
    logger.info("PROCESSING INSAR DATA")
    logger.info("="*70)
    
    insar_dir = RAW_DIR / "insar" / "sentinel1"
    output_dir = PROCESSED_DIR / "insar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if user has *manually* processed InSAR data
    processed_files = list(output_dir.glob("*.tif"))
    if processed_files:
        logger.info("Found pre-processed InSAR GeoTIFF. Clipping...")
        output_file = output_dir / "insar_processed.tif"
        return clip_and_reproject_raster(processed_files[0], output_file, region)

    # Check for raw data
    if not insar_dir.exists() or not any(insar_dir.glob("*.SAFE")):
        logger.warning("No raw InSAR data found (optional)")
        logger.warning("InSAR provides subsidence data for improved void detection")
        return False
    
    # Raw data exists, but not processed
    logger.warning("Found raw Sentinel-1 .SAFE directories.")
    logger.warning("This requires complex manual processing with SNAP or ISCE.")
    logger.warning("See 'process_insar_data.py' to generate a guide.")
    
    # Create guide
    guide_file = output_dir / "INSAR_PROCESSING_GUIDE.md"
    with open(guide_file, 'w') as f:
        f.write("# InSAR Processing Guide\n\n")
        f.write("## Option 1: COMET LiCSAR (Recommended)\n")
        f.write("1. Visit: https://comet.nerc.ac.uk/COMET-LiCS-portal/\n")
        f.write(f"2. Search for region: {region}\n")
        f.write("3. Download interferograms (already processed!)\n")
        f.write("4. Place in data/processed/insar/\n\n")
        f.write("## Option 2: Process Manually with SNAP\n")
        f.write("1. You have raw .SAFE data.\n")
        f.write("2. Run 'python process_insar_data.py' to get a SNAP XML graph.\n")
        f.write("3. Use the SNAP 'gpt' tool with the graph to process the data.\n")
        f.write("4. This is a complex, multi-hour step.\n")
    
    logger.info(f"✓ Guide saved: {guide_file}")
    return False


def process_lithology_data(region: Tuple[float, float, float, float]) -> bool:
    """Process lithology/geology data if available."""
    logger.info("\n" + "="*70)
    logger.info("PROCESSING LITHOLOGY DATA")
    logger.info("="*70)
    
    lithology_dir = RAW_DIR / "lithology"
    output_dir = PROCESSED_DIR / "lithology"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for lithology data
    if not lithology_dir.exists():
        logger.warning("No lithology data found (optional)")
        logger.warning("Lithology helps identify karst-prone rock types")
        return False
    
    # Look for compatible formats
    litho_files = list(lithology_dir.glob("*.tif"))
    litho_files.extend(lithology_dir.glob("*.shp"))
    
    if not litho_files:
        logger.warning("No processable lithology data found")
        logger.warning("Download from USGS or state geological surveys")
        return False
    
    # Process first available file
    logger.info(f"Found lithology data: {litho_files[0].name}")
    logger.warning("Lithology processing requires geological knowledge to map rock types")
    logger.warning("This step is not fully automated. Skipping.")
    return False


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_all_data(region: Tuple[float, float, float, float]):
    """Process all available data for void detection."""
    
    logger.info("\n" + "="*70)
    logger.info("GEOANOMALYMAPPER - DATA PROCESSING")
    logger.info("="*70)
    logger.info(f"Region: {region}")
    logger.info(f"Output directory: {PROCESSED_DIR}")
    logger.info("")
    
    results = {}
    
    # Process each dataset
    results['gravity'] = process_gravity_data(region)
    results['magnetic'] = process_magnetic_data(region)
    results['dem'] = process_dem_data(region) # Use upgraded function
    results['insar'] = process_insar_data(region)
    results['lithology'] = process_lithology_data(region)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*70)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "⚠ SKIPPED/MISSING"
        logger.info(f"{dataset.upper():15} {status}")
    
    # Check if we have minimum required data
    if results['gravity'] or results['magnetic']:
        logger.info("\n✓ Minimum data available for void detection")
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS")
        logger.info("="*70)
        logger.info("Run: python detect_voids.py --region <your_region>")
        logger.info("(This is handled by 'run_pipeline.py')")
        logger.info("="*70 + "\n")
    else:
        logger.warning("\n✗ Insufficient data for void detection")
        logger.warning("Need at least gravity OR magnetic data")
        logger.warning("Complete manual processing steps above")
    
    # Save processing log
    log_file = PROCESSED_DIR / "processing_log.json"
    log_data = {
        'timestamp': str(np.datetime64('now')),
        'region': region,
        'results': results
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Processing log saved: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process all geophysical data for void detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Carlsbad Caverns area (New Mexico)
  python process_data.py --region "-105.0,32.0,-104.0,33.0"
  
  # San Andreas Fault area (California)
  python process_data.py --region "-122.0,36.0,-121.0,37.0"
"""
    )
    
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.001,
        help='Output resolution in degrees (default: 0.001 = ~100m)'
    )
    
    args = parser.parse_args()
    
    # Parse region
    try:
        coords = list(map(float, args.region.split(',')))
        if len(coords) != 4:
            raise ValueError("Need 4 coordinates")
        region = tuple(coords)
    except Exception as e:
        logger.error(f"Invalid region format: {e}")
        logger.error("Use format: lon_min,lat_min,lon_max,lat_max")
        logger.error("Example: -105.0,32.0,-104.0,33.0")
        sys.exit(1)
    
    # Run processing
    process_all_data(region)


if __name__ == "__main__":
    main()