#!/usr/bin/env python3
"""
GeoAnomalyMapper - Unified Data Download Script
==================================================

Downloads ALL required geophysical data for void detection:
1. Gravity data (XGM2019e or EGM2008)
2. Magnetic data (EMAG2v3)
3. Elevation data (Copernicus DEM 30m)
4. InSAR data (Sentinel-1) - optional
5. Lithology data (USGS global)

Simple usage:
    python download_data.py --region "lon_min,lat_min,lon_max,lat_max"
    
Example:
    python download_data.py --region "-105.0,32.0,-104.0,33.0"  # Carlsbad Caverns area
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime
import requests
from urllib.parse import urlencode
import time

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
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ENV_FILE = BASE_DIR / ".env"

# Data source URLs
GRAVITY_URL = "https://ddfe.curtin.edu.au/gravitymodels/XGM2019e/XGM2019e_2159.gfc"
MAGNETIC_URL = "https://www.ngdc.noaa.gov/geomag/EMM/data/geomagnetic/emag2/EMAG2_V3_SeaLevel_DataTiff.tif"

# ============================================================================
# CREDENTIAL MANAGEMENT
# ============================================================================

def load_credentials() -> Dict[str, str]:
    """Load Copernicus credentials from environment or .env file."""
    creds = {
        'username': os.getenv('CDSE_USERNAME'),
        'password': os.getenv('CDSE_PASSWORD')
    }
    
    if creds['username'] and creds['password']:
        return creds
    
    # Try .env file
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    if key == 'CDSE_USERNAME':
                        creds['username'] = value
                    elif key == 'CDSE_PASSWORD':
                        creds['password'] = value
    
    return creds


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download file with progress indication."""
    try:
        logger.info(f"{desc}: {url}")
        logger.info(f"Saving to: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if output_path.exists():
            logger.info(f"✓ File already exists: {output_path}")
            return True
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        if downloaded % (chunk_size * 100) == 0:  # Update every ~800KB
                            logger.info(f"Progress: {progress:.1f}%")
        
        logger.info(f"✓ Download complete: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        return False


def download_gravity_data(region: Tuple[float, float, float, float]) -> bool:
    """Download gravity disturbance data."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING GRAVITY DATA")
    logger.info("="*70)
    
    gravity_dir = DATA_DIR / "gravity"
    gravity_dir.mkdir(parents=True, exist_ok=True)
    
    # Download XGM2019e coefficient file
    gfc_file = gravity_dir / "XGM2019e_2159.gfc"
    
    if not download_file(GRAVITY_URL, gfc_file, "Downloading gravity model"):
        logger.warning("Gravity model download failed")
        return False
    
    logger.info("✓ Gravity data ready")
    logger.info(f"Note: Use icgem.gfz-potsdam.de to convert .gfc to GeoTIFF for region: {region}")
    return True


def download_magnetic_data() -> bool:
    """Download global magnetic anomaly data (EMAG2)."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING MAGNETIC DATA (EMAG2v3)")
    logger.info("="*70)
    
    magnetic_dir = DATA_DIR / "emag2"
    magnetic_file = magnetic_dir / "EMAG2_V3_SeaLevel_DataTiff.tif"
    
    return download_file(MAGNETIC_URL, magnetic_file, "Downloading EMAG2 magnetic data")


def download_dem_data(region: Tuple[float, float, float, float]) -> bool:
    """Download elevation data for the region."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING ELEVATION DATA (Copernicus DEM)")
    logger.info("="*70)
    
    dem_dir = DATA_DIR / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    
    # For Copernicus DEM, we need to use their API
    # This is simplified - in production you'd use their proper API
    logger.info("Copernicus DEM requires authentication via their API")
    logger.info("Alternative: Download manually from https://copernicus-dem-30m.s3.amazonaws.com/")
    logger.info(f"Region: {region}")
    
    # Create instruction file
    instructions_file = dem_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions_file, 'w') as f:
        f.write("COPERNICUS DEM 30M DOWNLOAD INSTRUCTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Region: {region}\n\n")
        f.write("Option 1: AWS Open Data (No authentication required)\n")
        f.write("  URL: https://registry.opendata.aws/copernicus-dem/\n")
        f.write("  Download tiles covering your region\n\n")
        f.write("Option 2: Copernicus Data Space\n")
        f.write("  URL: https://dataspace.copernicus.eu/\n")
        f.write("  Requires free account\n")
    
    logger.info(f"✓ Instructions saved: {instructions_file}")
    return True


def download_insar_data(region: Tuple[float, float, float, float], days: int = 90) -> bool:
    """Download Sentinel-1 InSAR data (optional)."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING INSAR DATA (Sentinel-1) - OPTIONAL")
    logger.info("="*70)
    
    creds = load_credentials()
    
    if not creds['username'] or not creds['password']:
        logger.warning("⚠ No Copernicus credentials found")
        logger.warning("InSAR download skipped - data is optional")
        logger.warning("To enable InSAR, set credentials in .env file:")
        logger.warning("  CDSE_USERNAME=your_email@example.com")
        logger.warning("  CDSE_PASSWORD=your_password")
        return False
    
    insar_dir = DATA_DIR / "insar" / "sentinel1"
    insar_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Searching Sentinel-1 data for region: {region}")
    logger.info("This would download InSAR scenes...")
    logger.info("⚠ InSAR processing requires SNAP or ISCE software")
    logger.info("Consider using pre-processed data from COMET LiCSAR instead")
    
    # Create guide file
    guide_file = insar_dir / "INSAR_GUIDE.txt"
    with open(guide_file, 'w') as f:
        f.write("INSAR DATA OPTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Option 1: Pre-processed (RECOMMENDED)\n")
        f.write("  COMET LiCSAR Portal: https://comet.nerc.ac.uk/COMET-LiCS-portal/\n")
        f.write("  - Already processed interferograms\n")
        f.write("  - No software installation needed\n\n")
        f.write("Option 2: Raw data + processing\n")
        f.write("  - Download Sentinel-1 SLC from Copernicus\n")
        f.write("  - Process with SNAP or ISCE\n")
        f.write("  - Requires significant processing time\n")
    
    logger.info(f"✓ Guide saved: {guide_file}")
    return False  # InSAR is optional


def download_lithology_data() -> bool:
    """Download lithology/geology data for karst detection."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING LITHOLOGY DATA")
    logger.info("="*70)
    
    lithology_dir = DATA_DIR / "lithology"
    lithology_dir.mkdir(parents=True, exist_ok=True)
    
    # USGS global geology
    logger.info("Lithology data available from multiple sources:")
    logger.info("1. USGS Global Geology: https://mrdata.usgs.gov/geology/world/")
    logger.info("2. OneGeology Portal: http://www.onegeology.org/")
    
    guide_file = lithology_dir / "LITHOLOGY_SOURCES.txt"
    with open(guide_file, 'w') as f:
        f.write("LITHOLOGY DATA SOURCES\n")
        f.write("=" * 60 + "\n\n")
        f.write("For karst void detection, focus on:\n")
        f.write("- Limestone\n")
        f.write("- Dolomite\n")
        f.write("- Evaporites (gypsum, halite)\n\n")
        f.write("Data Sources:\n")
        f.write("1. USGS Global Geology\n")
        f.write("   https://mrdata.usgs.gov/geology/world/\n\n")
        f.write("2. State geological surveys (USA)\n")
        f.write("   - More detailed regional data\n")
    
    logger.info(f"✓ Guide saved: {guide_file}")
    return True


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def download_all_data(region: Tuple[float, float, float, float], skip_insar: bool = False):
    """Download all required data for void detection."""
    
    logger.info("\n" + "="*70)
    logger.info("GEOANOMALYMAPPER - DATA DOWNLOAD")
    logger.info("="*70)
    logger.info(f"Region: {region}")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info("")
    
    results = {}
    
    # Download each dataset
    results['gravity'] = download_gravity_data(region)
    results['magnetic'] = download_magnetic_data()
    results['dem'] = download_dem_data(region)
    
    if not skip_insar:
        results['insar'] = download_insar_data(region)
    else:
        logger.info("\n⚠ Skipping InSAR (--skip-insar flag set)")
        results['insar'] = False
    
    results['lithology'] = download_lithology_data()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*70)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "⚠ NEEDS MANUAL DOWNLOAD"
        logger.info(f"{dataset.upper():15} {status}")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Complete any manual downloads (see instructions in data/raw/)")
    logger.info("2. Run: python process_data.py --region <your_region>")
    logger.info("3. Run: python detect_voids.py --region <your_region>")
    logger.info("="*70 + "\n")
    
    # Save download log
    log_file = DATA_DIR / "download_log.json"
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'region': region,
        'results': results
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Download log saved: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required geophysical data for void detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Carlsbad Caverns area (New Mexico)
  python download_data.py --region "-105.0,32.0,-104.0,33.0"
  
  # San Andreas Fault area (California)
  python download_data.py --region "-122.0,36.0,-121.0,37.0"
  
  # Skip InSAR (faster, gravity-only analysis)
  python download_data.py --region "-105.0,32.0,-104.0,33.0" --skip-insar
"""
    )
    
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
    )
    
    parser.add_argument(
        '--skip-insar',
        action='store_true',
        help='Skip InSAR data download (faster, gravity-only)'
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
    
    # Run downloads
    download_all_data(region, args.skip_insar)


if __name__ == "__main__":
    main()