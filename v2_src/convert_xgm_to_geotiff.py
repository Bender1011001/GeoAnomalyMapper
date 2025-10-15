#!/usr/bin/env python3
"""
Convert XGM2019e to High-Resolution GeoTIFF Gravity Grid via ICGEM
=================================================================

This script downloads high-resolution gravity disturbance grid from the ICGEM
calculation service for the XGM2019e model. Supports custom grid resolution
down to 0.0025° (~250m) for detailed anomaly detection in small regions.

The script submits a calculation request to ICGEM and saves the resulting GeoTIFF.

Requirements:
- requests: For HTTP requests to ICGEM service

Install with: pip install requests
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
PROJECT_ROOT = BASE_DIR
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "gravity"

def download_xgm2019e_grid(lon_min: float, lat_min: float, 
                           lon_max: float, lat_max: float,
                           gridstep: float, output_file: Path) -> bool:
    """
    Download XGM2019e gravity grid using the ICGEM calculation service.
    
    Submits a programmatic request to compute the gravity grid for the specified
    region and resolution. For very fine resolutions (e.g., 0.0025°), the service
    may take longer or require manual intervention if it times out.
    
    Args:
        lon_min, lat_min, lon_max, lat_max: Region bounds (degrees)
        gridstep: Grid resolution (degrees, e.g., 0.0025 for ~250m)
        output_file: Output GeoTIFF path
    
    Returns:
        bool: True if successful
    """
    logger.info("="*70)
    logger.info("DOWNLOADING XGM2019e HIGH-RESOLUTION GRAVITY GRID FROM ICGEM")
    logger.info("="*70)
    logger.info(f"Region: {lon_min}° to {lon_max}°E, {lat_min}° to {lat_max}°N")
    logger.info(f"Resolution: {gridstep}° (~{int(111 * gridstep * 1000)}m at equator)")
    logger.info(f"Output: {output_file}")
    logger.info("")
    
    # ICGEM calculation service URL (note: uses GET for calcgrid in practice, but POST works for form data)
    base_url = "http://icgem.gfz-potsdam.de/calcgrid"
    
    # Parameters for the request
    params = {
        'model': 'XGM2019e_2159',
        'gridtype': 'grid',
        'latlim_north': lat_max,
        'latlim_south': lat_min,
        'longlim_west': lon_min,
        'longlim_east': lon_max,
        'gridstep': gridstep,  # Custom resolution
        'height': 0,  # Sea level
        'functional': 'gravity_disturbance',  # mGal
        'outputformat': 'geotiff'
    }
    
    logger.info("Submitting calculation request to ICGEM...")
    logger.info(f"This may take 1-5 minutes for fine resolution ({gridstep}°)...")
    logger.info("If it times out, use manual download (see below).")
    
    try:
        # Submit the calculation request (use POST for form data)
        response = requests.post(base_url, data=params, timeout=300)  # Increased timeout for fine grid
        response.raise_for_status()
        
        # Check if we got a GeoTIFF back directly
        content_type = response.headers.get('content-type', '')
        if 'image/tiff' in content_type or output_file.suffix.lower() == '.tif':
            # Save the GeoTIFF
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info("")
            logger.info("✓ SUCCESS!")
            logger.info(f"✓ Downloaded high-resolution gravity grid: {file_size_mb:.2f} MB")
            logger.info(f"✓ Saved to: {output_file}")
            logger.info(f"✓ Resolution: {gridstep}°")
            logger.info("="*70)
            return True
        else:
            # ICGEM may return HTML with a download link or job ID
            # For now, log and suggest manual
            logger.warning("ICGEM returned non-TIFF response (likely HTML job page)")
            logger.warning(f"Response preview: {response.text[:200]}...")
            logger.warning("For fine resolutions, manual download may be required.")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out. ICGEM service may be slow for fine resolutions.")
        logger.error("Try coarser gridstep (e.g., 0.01°) or manual download.")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def print_manual_instructions(lon_min, lat_min, lon_max, lat_max, gridstep):
    """Print detailed manual download instructions for ICGEM."""
    logger.info("")
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS (if automated failed):")
    logger.info("="*50)
    logger.info("1. Visit: http://icgem.gfz-potsdam.de/home")
    logger.info("2. Click 'Online Computation' -> 'Grid Calculation'")
    logger.info("3. Select:")
    logger.info(f"   - Model: XGM2019e_2159")
    logger.info(f"   - Functional: gravity_disturbance")
    logger.info(f"   - Height: 0 km (sea level)")
    logger.info(f"   - Grid type: Grid")
    logger.info(f"   - Latitude range: {lat_min} to {lat_max}")
    logger.info(f"   - Longitude range: {lon_min} to {lon_max}")
    logger.info(f"   - Grid step: {gridstep} degrees")
    logger.info("   - Output format: GeoTIFF")
    logger.info("4. Submit and download the resulting .tif file")
    logger.info(f"5. Save as: {DATA_DIR / 'xgm2019e_high_resolution.tif'}")
    logger.info("")

def main():
    parser = argparse.ArgumentParser(
        description="Download XGM2019e high-resolution gravity GeoTIFF from ICGEM"
    )
    parser.add_argument(
        '--input', type=str, default="data/raw/gravity/XGM2019e_2159.gfc",
        help='Path to XGM2019e_2159.gfc (not used directly, for compatibility)'
    )
    parser.add_argument(
        '--output', type=str, default="data/raw/gravity/xgm2019e_high_resolution.tif",
        help='Output GeoTIFF file path'
    )
    parser.add_argument(
        '--lat-min', type=float, default=32.0, help='Minimum latitude'
    )
    parser.add_argument(
        '--lat-max', type=float, default=33.0, help='Maximum latitude'
    )
    parser.add_argument(
        '--lon-min', type=float, default=-105.0, help='Minimum longitude'
    )
    parser.add_argument(
        '--lon-max', type=float, default=-104.0, help='Maximum longitude'
    )
    parser.add_argument(
        '--gridstep', type=float, default=0.0025, 
        help='Grid resolution in degrees (0.0025 ~250m)'
    )
    
    args = parser.parse_args()
    
    # Parse bounds
    lat_min, lat_max = args.lat_min, args.lat_max
    lon_min, lon_max = args.lon_min, args.lon_max
    gridstep = args.gridstep
    output_file = Path(args.output)
    
    # Validate bounds
    if lat_max <= lat_min or lon_max <= lon_min:
        logger.error("Invalid bounds: lat_max > lat_min and lon_max > lon_min required")
        sys.exit(1)
    
    logger.info("")
    logger.info("╔" + "═"*68 + "╗")
    logger.info("║" + " "*20 + "XGM2019e HIGH-RES GRAVITY DOWNLOADER" + " "*14 + "║")
    logger.info("╚" + "═"*68 + "╝")
    logger.info("")
    
    # Try automated download
    success = download_xgm2019e_grid(lon_min, lat_min, lon_max, lat_max, gridstep, output_file)
    
    if success:
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Verify the GeoTIFF: gdalinfo data/raw/gravity/xgm2019e_high_resolution.tif")
        logger.info("2. Run: python GeoAnomalyMapper/process_nasadem_elevation.py")
        logger.info("3. Run: python GeoAnomalyMapper/process_data.py --region \"-105.0,32.0,-104.0,33.0\"")
    else:
        print_manual_instructions(lon_min, lat_min, lon_max, lat_max, gridstep)
        logger.info("After manual download, proceed with step 2 above.")
        sys.exit(1)

if __name__ == "__main__":
    main()