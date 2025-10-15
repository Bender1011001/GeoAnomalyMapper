#!/usr/bin/env python3
"""
Convert XGM2019e Spherical Harmonics to Gravity Grid
====================================================

Converts the XGM2019e_2159.gfc spherical harmonic coefficient file to a 
high-resolution gravity disturbance grid for the Carlsbad region.

This uses the ICGEM calculation service since converting spherical harmonics
directly requires complex mathematical operations that are better handled by
specialized tools.

Usage:
    python convert_xgm2019e_to_grid.py --region="-105.0,32.0,-104.0,33.0"
"""

import argparse
import logging
import sys
from pathlib import Path
import requests
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "gravity" / "xgm2019e"


def download_xgm2019e_grid(lon_min: float, lat_min: float, 
                           lon_max: float, lat_max: float,
                           output_file: Path) -> bool:
    """
    Download XGM2019e gravity grid using the ICGEM calculation service.
    
    This makes a programmatic request to the ICGEM service to compute
    the gravity grid for the specified region.
    """
    
    logger.info("="*70)
    logger.info("DOWNLOADING XGM2019E GRAVITY GRID FROM ICGEM")
    logger.info("="*70)
    logger.info(f"Region: {lon_min}° to {lon_max}°E, {lat_min}° to {lat_max}°N")
    logger.info(f"Resolution: 0.02° (~2 km)")
    logger.info("")
    
    # ICGEM calculation service URL
    base_url = "http://icgem.gfz-potsdam.de/calcgrid"
    
    # Parameters for the request
    params = {
        'model': 'XGM2019e_2159',
        'gridtype': 'grid',
        'latlim_north': lat_max,
        'latlim_south': lat_min,
        'longlim_west': lon_min,
        'longlim_east': lon_max,
        'gridstep': 0.02,  # ~2km resolution
        'height': 0,  # Sea level
        'functional': 'gravity_disturbance',  # mGal
        'outputformat': 'geotiff'
    }
    
    logger.info("Submitting calculation request to ICGEM...")
    logger.info("This may take 30-120 seconds depending on region size...")
    
    try:
        # Submit the calculation request
        response = requests.post(base_url, data=params, timeout=180)
        response.raise_for_status()
        
        # Check if we got a GeoTIFF back
        if response.headers.get('content-type', '').startswith('image/tiff'):
            # Save the GeoTIFF
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info("")
            logger.info("✓ SUCCESS!")
            logger.info(f"✓ Downloaded gravity grid: {file_size_mb:.2f} MB")
            logger.info(f"✓ Saved to: {output_file}")
            logger.info("="*70)
            return True
        else:
            # Sometimes ICGEM returns HTML with a job ID
            # In this case, we'd need to poll for results
            logger.warning("Server returned non-TIFF response")
            logger.warning("The ICGEM service may require manual download")
            logger.warning("Please use the manual instructions")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out. The ICGEM service may be slow.")
        logger.error("Please try the manual download method instead.")
        return False
    except Exception as e:
        logger.error(f"Failed to download grid: {e}")
        logger.error("Please use the manual download method.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert XGM2019e coefficients to gravity grid"
    )
    parser.add_argument(
        '--region',
        type=str,
        default="-105.0,32.0,-104.0,33.0",
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output GeoTIFF file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Parse region
    try:
        coords = list(map(float, args.region.split(',')))
        if len(coords) != 4:
            raise ValueError("Need 4 coordinates")
        lon_min, lat_min, lon_max, lat_max = coords
    except Exception as e:
        logger.error(f"Invalid region format: {e}")
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DATA_DIR / "xgm2019e_carlsbad.tif"
    
    logger.info("")
    logger.info("╔" + "═"*68 + "╗")
    logger.info("║" + " "*12 + "XGM2019E GRAVITY GRID CONVERTER" + " "*23 + "║")
    logger.info("╚" + "═"*68 + "╝")
    logger.info("")
    
    # Try programmatic download
    success = download_xgm2019e_grid(lon_min, lat_min, lon_max, lat_max, output_file)
    
    if success:
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Run: python GeoAnomalyMapper/process_data.py")
        logger.info("2. Then: python GeoAnomalyMapper/detect_voids.py")
        logger.info("3. Finally: python GeoAnomalyMapper/create_enhanced_visualization.py")
    else:
        logger.info("")
        logger.info("ALTERNATIVE: MANUAL DOWNLOAD")
        logger.info("Since programmatic download failed, please:")
        logger.info("1. Visit: http://icgem.gfz-potsdam.de/tom_longtime")
        logger.info("2. Use the parameters shown in DOWNLOAD_INSTRUCTIONS.txt")
        logger.info(f"3. Save the downloaded file as: {output_file}")


if __name__ == "__main__":
    main()