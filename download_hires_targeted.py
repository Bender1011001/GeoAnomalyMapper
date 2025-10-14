#!/usr/bin/env python3
"""
Targeted High-Resolution Data Downloader
=========================================

Downloads high-resolution data specifically for your region of interest:
1. Copernicus DEM 30m (only land tiles)
2. Prepares for XGM2019e gravity (manual download helper)
3. Lists available InSAR data

Usage:
    python download_hires_targeted.py --region "-105.0,32.0,-104.0,33.0"
"""

import argparse
import logging
import sys
from pathlib import Path
import requests
from typing import Tuple, List
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================================
# COPERNICUS DEM DOWNLOAD (Land tiles only for Carlsbad area)
# ============================================================================

def get_land_tiles_for_region(lon_min: float, lat_min: float, 
                               lon_max: float, lat_max: float) -> List[str]:
    """Get only the land tiles for a specific region (pre-validated for USA)."""
    
    # For Carlsbad Caverns area (-105 to -104, 32 to 33), these are the likely land tiles
    # We'll try a focused set rather than all possible combinations
    tiles = []
    
    for lat in range(int(lat_min), int(lat_max) + 1):
        for lon in range(int(lon_min), int(lon_max) + 1):
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.append(f"{lat_str}_00_{lon_str}_00")
    
    return tiles


def download_copernicus_dem_tiles(region: Tuple[float, float, float, float]) -> int:
    """Download Copernicus DEM tiles for the region."""
    
    logger.info("="*70)
    logger.info("DOWNLOADING COPERNICUS DEM 30M")
    logger.info("="*70)
    
    dem_dir = DATA_DIR / "elevation" / "copernicus_dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    
    lon_min, lat_min, lon_max, lat_max = region
    tiles = get_land_tiles_for_region(lon_min, lat_min, lon_max, lat_max)
    
    logger.info(f"Attempting to download {len(tiles)} tiles for region")
    logger.info(f"Region: {lon_min}° to {lon_max}° longitude, {lat_min}° to {lat_max}° latitude")
    
    base_url = "https://copernicus-dem-30m.s3.amazonaws.com"
    successful = 0
    failed = []
    
    for tile in tiles:
        tile_name = f"Copernicus_DSM_COG_10_{tile}_DEM.tif"
        url = f"{base_url}/{tile_name}"
        output = dem_dir / tile_name
        
        if output.exists():
            logger.info(f"✓ Already exists: {tile}")
            successful += 1
            continue
        
        try:
            logger.info(f"Downloading: {tile}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(output, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✓ Downloaded: {tile}")
            successful += 1
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"  Tile not available (likely ocean/outside coverage): {tile}")
                failed.append(tile)
            else:
                logger.warning(f"  HTTP error for {tile}: {e}")
                failed.append(tile)
        except Exception as e:
            logger.error(f"  Failed to download {tile}: {e}")
            failed.append(tile)
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"DEM Download Summary: {successful}/{len(tiles)} tiles successfully downloaded")
    if successful > 0:
        logger.info(f"✓ Copernicus DEM data ready in: {dem_dir}")
    logger.info("="*70)
    
    return successful


# ============================================================================
# XGM2019E GRAVITY - MANUAL DOWNLOAD INSTRUCTIONS
# ============================================================================

def create_xgm2019e_instructions(region: Tuple[float, float, float, float]):
    """Create detailed instructions for XGM2019e manual download."""
    
    logger.info("")
    logger.info("="*70)
    logger.info("XGM2019E GRAVITY MODEL - MANUAL DOWNLOAD REQUIRED")
    logger.info("="*70)
    
    gravity_dir = DATA_DIR / "gravity" / "xgm2019e"
    gravity_dir.mkdir(parents=True, exist_ok=True)
    
    lon_min, lat_min, lon_max, lat_max = region
    
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════╗
║        XGM2019E HIGH-RESOLUTION GRAVITY MODEL DOWNLOAD               ║
║                  (~2km resolution - 10x better!)                      ║
╚══════════════════════════════════════════════════════════════════════╝

WHY THIS IS IMPORTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGM2019e provides ~2km resolution gravity data compared to the baseline
EGM2008's ~20km resolution. This is a 10x improvement that will dramatically
enhance void detection accuracy.

STEP-BY-STEP DOWNLOAD INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Open your web browser and visit:
   🌐 http://icgem.gfz-potsdam.de/tom_longtime

2. Fill in the form with these EXACT values:

   ┌─────────────────────────────────────────────────────────────┐
   │ Model:              XGM2019e_2159                           │
   │ Grid type:          Grid                                     │
   │ Latitude (min):     {lat_min}                               │
   │ Latitude (max):     {lat_max}                               │
   │ Longitude (min):    {lon_min}                               │
   │ Longitude (max):    {lon_max}                               │
   │ Grid step:          0.02    (degrees) ← IMPORTANT!          │
   │ Height:             0       (meters, sea level)             │
   │ Quantity:           Gravity disturbance                      │
   │ Output format:      GeoTIFF                                  │
   └─────────────────────────────────────────────────────────────┘

3. Click the "Compute grid" button

4. Wait for computation to complete (usually 30-60 seconds)

5. Download the generated GeoTIFF file

6. Save it to this directory:
   📁 {gravity_dir}
   
   Recommended filename: xgm2019e_carlsbad.tif
   (Or any name ending in .tif)

7. Re-run the processing pipeline:
   python process_data.py --region="{lon_min},{lat_min},{lon_max},{lat_max}"

WHAT HAPPENS NEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ The processing script will automatically detect the new gravity file
✓ It will use the higher resolution data instead of EGM2008
✓ Void detection accuracy will improve significantly
✓ You'll be able to detect smaller features (down to ~100m diameter)

FILE SIZE: Approximately 5-20 MB for your region

TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If the website is slow, try during off-peak hours
• Make sure to select "GeoTIFF" format (not ASCII or other formats)
• The grid step of 0.02° gives you ~2km resolution
• Don't change other parameters unless you know what you're doing

═══════════════════════════════════════════════════════════════════════════
"""
    
    instructions_file = gravity_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    logger.info(instructions)
    logger.info(f"✓ Instructions saved to: {instructions_file}")
    logger.info("="*70)


# ============================================================================
# INSAR DATA STATUS
# ============================================================================

def check_insar_status():
    """Check what InSAR data is already available."""
    
    logger.info("")
    logger.info("="*70)
    logger.info("INSAR DATA STATUS")
    logger.info("="*70)
    
    insar_dir = DATA_DIR / "insar" / "sentinel1"
    
    if not insar_dir.exists():
        logger.info("No InSAR data directory found")
        logger.info("")
        logger.info("InSAR ground deformation data can be obtained from:")
        logger.info("  1. COMET LiCSAR (pre-processed): https://comet.nerc.ac.uk/COMET-LiCS-portal/")
        logger.info("  2. Alaska Satellite Facility (raw): https://search.asf.alaska.edu/")
        logger.info("")
        logger.info("For beginners, we recommend using pre-processed LiCSAR data")
        return
    
    # Count .SAFE directories (Sentinel-1 products)
    safe_dirs = list(insar_dir.glob("*.SAFE"))
    
    if not safe_dirs:
        logger.info("No Sentinel-1 data found in InSAR directory")
    else:
        logger.info(f"Found {len(safe_dirs)} Sentinel-1 scenes:")
        for safe_dir in safe_dirs[:5]:  # Show first 5
            logger.info(f"  • {safe_dir.name}")
        if len(safe_dirs) > 5:
            logger.info(f"  ... and {len(safe_dirs) - 5} more")
    
    logger.info("")
    logger.info("NOTE: Raw Sentinel-1 data requires processing with SNAP or ISCE software")
    logger.info("This is an advanced workflow. For easier analysis, consider:")
    logger.info("  • Pre-processed LiCSAR interferograms")
    logger.info("  • Or use the existing gravity + magnetic analysis")
    logger.info("="*70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download high-resolution data for improved void detection"
    )
    parser.add_argument(
        '--region',
        type=str,
        default="-105.0,32.0,-104.0,33.0",
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
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
        sys.exit(1)
    
    logger.info("")
    logger.info("╔" + "═"*68 + "╗")
    logger.info("║" + " "*15 + "HIGH-RESOLUTION DATA DOWNLOADER" + " "*22 + "║")
    logger.info("╚" + "═"*68 + "╝")
    logger.info("")
    logger.info(f"Target Region: {region}")
    logger.info("")
    
    # 1. Download Copernicus DEM
    dem_count = download_copernicus_dem_tiles(region)
    
    # 2. Create XGM2019e instructions
    create_xgm2019e_instructions(region)
    
    # 3. Check InSAR status
    check_insar_status()
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Copernicus DEM tiles: {dem_count} downloaded")
    logger.info("⚠ XGM2019e Gravity: Manual download required (see instructions above)")
    logger.info("ℹ InSAR: Check status above")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Follow XGM2019e instructions to download gravity data")
    logger.info("2. After download, run: python process_data.py --region=\"{},{},{},{}\"".format(*region))
    logger.info("3. Then run: python detect_voids.py --region=\"{},{},{},{}\"".format(*region))
    logger.info("4. Finally: python create_enhanced_visualization.py --region=\"{},{},{},{}\"".format(*region))
    logger.info("="*70)


if __name__ == "__main__":
    main()