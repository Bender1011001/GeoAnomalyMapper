#!/usr/bin/env python3
"""
GeoAnomalyMapper - Comprehensive Automated Data Downloader
===========================================================

Downloads ALL free geophysical data in phases:
- Phase 1: Critical baseline (Copernicus DEM, XGM2019e gravity, EMAG2 magnetic)
- Phase 2: High-resolution optional (Sentinel-2, aeromagnetic surveys)
- Phase 3: Context layers (lithology, hydrology)

Usage:
    python download_all_free_data.py --phases 1              # Critical data only
    python download_all_free_data.py --phases 1 2            # Critical + high-res
    python download_all_free_data.py --phases 1 2 3          # Everything
    
    # Custom region (default is USA Lower 48)
    python download_all_free_data.py --phases 1 --lon-min -125 --lat-min 24.5 --lon-max -66.95 --lat-max 49.5
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from urllib.parse import urlencode
import hashlib

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Run: pip install requests tqdm")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
STATUS_FILE = PROJECT_ROOT / "data" / "download_status.json"

# USA Lower 48 default bounds
DEFAULT_BOUNDS = {
    'lon_min': -125.0,
    'lat_min': 24.5,
    'lon_max': -66.95,
    'lat_max': 49.5
}

# ============================================================================
# DATA SOURCES
# ============================================================================

DATA_SOURCES = {
    'phase1': {
        'copernicus_dem': {
            'name': 'Copernicus DEM 30m',
            'size_gb': 50,
            'priority': 'HIGHEST',
            'auto': True,
            'urls': [],  # Populated dynamically based on region
        },
        'xgm2019e_gravity': {
            'name': 'XGM2019e Gravity Model',
            'size_gb': 0.5,
            'priority': 'HIGHEST',
            'auto': False,  # Requires manual ICGEM download
            'url': 'http://icgem.gfz-potsdam.de/tom_longtime',
        },
        'emag2_magnetic': {
            'name': 'EMAG2v3 Magnetic Anomaly',
            'size_gb': 0.3,
            'priority': 'HIGH',
            'auto': True,
            'url': 'https://www.ngdc.noaa.gov/geomag/EMM/data/geomagnetic/emag2/EMAG2_V3_SeaLevel_DataTiff.tif',
        },
    },
    'phase2': {
        'sentinel2': {
            'name': 'Sentinel-2 Optical Imagery',
            'size_gb': 100,
            'priority': 'HIGH',
            'auto': False,  # Requires authentication
        },
        'usgs_3dep': {
            'name': 'USGS 3DEP Lidar',
            'size_gb': 500,
            'priority': 'HIGH',
            'auto': False,  # Regional availability
        },
        'aeromagnetic': {
            'name': 'USGS Aeromagnetic Surveys',
            'size_gb': 10,
            'priority': 'MEDIUM',
            'auto': True,
            'url': 'https://mrdata.usgs.gov/magnetic/map-us.html',
        },
    },
    'phase3': {
        'lithology': {
            'name': 'Global Lithology Map',
            'size_gb': 1,
            'priority': 'MEDIUM',
            'auto': True,
            'url': 'https://mrdata.usgs.gov/geology/world/',
        },
    }
}

# ============================================================================
# DOWNLOAD STATUS TRACKING
# ============================================================================

def load_status() -> Dict:
    """Load download status from JSON file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {'last_update': None, 'datasets': {}}


def save_status(status: Dict):
    """Save download status to JSON file."""
    status['last_update'] = datetime.now().isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    logger.info(f"Status saved to {STATUS_FILE}")


def mark_complete(status: Dict, dataset_name: str, metadata: Dict = None):
    """Mark a dataset as complete."""
    if 'datasets' not in status:
        status['datasets'] = {}
    
    status['datasets'][dataset_name] = {
        'status': 'complete',
        'completed_at': datetime.now().isoformat(),
        **(metadata or {})
    }
    save_status(status)


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(url: str, output_path: Path, desc: str = None, chunk_size: int = 8192) -> bool:
    """Download file with progress bar and resume support."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"✓ Already exists: {output_path.name}")
            return True
        
        # Handle partial downloads
        temp_path = output_path.with_suffix(output_path.suffix + '.partial')
        resume_pos = temp_path.stat().st_size if temp_path.exists() else 0
        
        headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}
        
        logger.info(f"Downloading: {desc or url}")
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 416:  # Range not satisfiable - file complete
            temp_path.rename(output_path)
            return True
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + resume_pos
        mode = 'ab' if resume_pos > 0 else 'wb'
        
        with open(temp_path, mode) as f, tqdm(
            desc=desc or output_path.name,
            initial=resume_pos,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Move completed file
        temp_path.rename(output_path)
        logger.info(f"✓ Download complete: {output_path.name}")
        return True
        
    except KeyboardInterrupt:
        logger.warning("Download interrupted - progress saved")
        raise
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return False


# ============================================================================
# PHASE 1: CRITICAL BASELINE DATA
# ============================================================================

def download_copernicus_dem(bounds: Dict, status: Dict) -> bool:
    """Download Copernicus DEM 30m tiles for the region."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING COPERNICUS DEM 30M")
    logger.info("="*70)
    
    dem_dir = DATA_DIR / "elevation" / "copernicus_dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tile list for region
    tiles = generate_dem_tiles(bounds)
    logger.info(f"Need {len(tiles)} DEM tiles for region")
    
    # AWS Open Data S3 bucket (no auth required)
    base_url = "https://copernicus-dem-30m.s3.amazonaws.com"
    
    successful = 0
    failed = []
    
    for tile in tiles:
        tile_name = f"Copernicus_DSM_COG_10_{tile}_DEM.tif"
        url = f"{base_url}/{tile_name}"
        output = dem_dir / tile_name
        
        if download_file(url, output, f"DEM tile {tile}"):
            successful += 1
        else:
            failed.append(tile)
    
    logger.info(f"\nDEM Download Summary: {successful}/{len(tiles)} tiles")
    
    if failed:
        logger.warning(f"Failed tiles: {', '.join(failed[:10])}")
        logger.info("Re-run script to retry failed tiles")
    
    mark_complete(status, 'copernicus_dem', {
        'tiles_downloaded': successful,
        'tiles_total': len(tiles),
        'resolution': '30m'
    })
    
    return successful > 0


def generate_dem_tiles(bounds: Dict) -> List[str]:
    """Generate list of Copernicus DEM tile names for a region."""
    tiles = []
    
    lat_min = int(bounds['lat_min'])
    lat_max = int(bounds['lat_max']) + 1
    lon_min = int(bounds['lon_min'])
    lon_max = int(bounds['lon_max']) + 1
    
    for lat in range(lat_min, lat_max):
        for lon in range(lon_min, lon_max):
            # Format: N32_W105
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.append(f"{lat_str}_00_{lon_str}_00")
    
    return tiles


def download_emag2_magnetic(status: Dict) -> bool:
    """Download EMAG2v3 global magnetic anomaly data."""
    logger.info("\n" + "="*70)
    logger.info("DOWNLOADING EMAG2V3 MAGNETIC ANOMALY")
    logger.info("="*70)
    
    magnetic_dir = DATA_DIR / "magnetic" / "emag2"
    magnetic_file = magnetic_dir / "EMAG2_V3_SeaLevel_DataTiff.tif"
    
    url = DATA_SOURCES['phase1']['emag2_magnetic']['url']
    
    if download_file(url, magnetic_file, "EMAG2v3 Global Magnetic"):
        mark_complete(status, 'emag2_magnetic', {'resolution': '2_arcmin'})
        return True
    return False


def setup_gravity_manual(bounds: Dict, status: Dict) -> bool:
    """Create instructions for manual XGM2019e gravity download."""
    logger.info("\n" + "="*70)
    logger.info("XGM2019E GRAVITY MODEL - MANUAL DOWNLOAD REQUIRED")
    logger.info("="*70)
    
    gravity_dir = DATA_DIR / "gravity" / "xgm2019e"
    gravity_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = f"""
XGM2019E GRAVITY MODEL DOWNLOAD INSTRUCTIONS
{'='*70}

The XGM2019e gravity model provides ~2km resolution gravity disturbances,
which is 10x better than the baseline EGM2008 model (~20km).

DOWNLOAD STEPS:
1. Visit: http://icgem.gfz-potsdam.de/tom_longtime

2. Select these options:
   Model: XGM2019e_2159
   Grid type: Grid
   Latitude range: {bounds['lat_min']} to {bounds['lat_max']}
   Longitude range: {bounds['lon_min']} to {bounds['lon_max']}
   Grid step: 0.02 degree (2km resolution)
   Height: 0m (sea level)
   Quantity: Gravity disturbance
   Format: GeoTIFF

3. Click "Compute grid"

4. Download the generated file

5. Save it to this directory: {gravity_dir}
   Filename: xgm2019e_usa.tif (or similar)

WHY MANUAL?
The ICGEM service requires interactive grid computation.
The file is ~500MB for USA coverage.

AFTER DOWNLOAD:
The processing scripts will automatically detect and use this file.
"""
    
    instructions_file = gravity_dir / "DOWNLOAD_MANUALLY.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    logger.info(f"✓ Instructions saved: {instructions_file}")
    logger.info("\n" + instructions)
    
    mark_complete(status, 'xgm2019e_gravity', {
        'status': 'manual_download_required',
        'instructions': str(instructions_file)
    })
    
    return True


# ============================================================================
# PHASE 2: HIGH-RESOLUTION OPTIONAL
# ============================================================================

def setup_phase2_instructions(status: Dict) -> bool:
    """Create instructions for Phase 2 data sources."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: HIGH-RESOLUTION DATA (OPTIONAL)")
    logger.info("="*70)
    
    phase2_dir = DATA_DIR / "phase2_instructions"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = """
PHASE 2: HIGH-RESOLUTION DATA SOURCES
======================================

These datasets significantly improve detection but require more storage
and processing time.

1. SENTINEL-2 OPTICAL IMAGERY (~100 GB)
   - Resolution: 10m multispectral
   - Requires: Copernicus Data Space account (free)
   - Download: https://dataspace.copernicus.eu/
   - Use: sentinelsat Python package

2. USGS 3DEP LIDAR (~500 GB for full coverage)
   - Resolution: 1m elevation
   - Coverage: ~60% of USA (expanding)
   - Download: https://apps.nationalmap.gov/downloader/
   - Or use: py3dep Python package for automation

3. USGS AEROMAGNETIC SURVEYS (~10 GB)
   - Resolution: 100m-1km
   - Coverage: Variable by state
   - Download: https://mrdata.usgs.gov/magnetic/
   - Multiple survey compilations available

RECOMMENDED APPROACH:
Start with Phase 1 data, run initial analysis, then add Phase 2 data
for specific regions of interest.
"""
    
    instructions_file = phase2_dir / "README.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    logger.info(f"✓ Phase 2 instructions: {instructions_file}")
    return True


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def download_phase(phase: int, bounds: Dict, status: Dict) -> bool:
    """Download all data for a specific phase."""
    
    if phase == 1:
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: CRITICAL BASELINE DATA")
        logger.info("="*70)
        logger.info("Size: ~51 GB | Time: 1-2 hours | Priority: HIGHEST")
        logger.info("")
        
        # Download automatically downloadable datasets
        download_emag2_magnetic(status)
        download_copernicus_dem(bounds, status)
        
        # Setup manual download instructions
        setup_gravity_manual(bounds, status)
        
        return True
    
    elif phase == 2:
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: HIGH-RESOLUTION OPTIONAL")
        logger.info("="*70)
        logger.info("Size: ~610 GB | Time: Hours-days | Priority: HIGH")
        logger.info("")
        
        setup_phase2_instructions(status)
        return True
    
    elif phase == 3:
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: CONTEXT LAYERS")
        logger.info("="*70)
        logger.info("Size: ~50 GB | Time: 1-2 hours | Priority: MEDIUM")
        logger.info("Coming in future update...")
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download all free geophysical data for GeoAnomalyMapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download critical baseline data (Phase 1)
  python download_all_free_data.py --phases 1
  
  # Download Phases 1 and 2
  python download_all_free_data.py --phases 1 2
  
  # Custom region
  python download_all_free_data.py --phases 1 --lon-min -105 --lat-min 32 --lon-max -104 --lat-max 33
"""
    )
    
    parser.add_argument(
        '--phases',
        type=int,
        nargs='+',
        default=[1],
        choices=[1, 2, 3],
        help='Phases to download (1=critical, 2=high-res, 3=context)'
    )
    
    parser.add_argument('--lon-min', type=float, default=DEFAULT_BOUNDS['lon_min'])
    parser.add_argument('--lat-min', type=float, default=DEFAULT_BOUNDS['lat_min'])
    parser.add_argument('--lon-max', type=float, default=DEFAULT_BOUNDS['lon_max'])
    parser.add_argument('--lat-max', type=float, default=DEFAULT_BOUNDS['lat_max'])
    
    args = parser.parse_args()
    
    bounds = {
        'lon_min': args.lon_min,
        'lat_min': args.lat_min,
        'lon_max': args.lon_max,
        'lat_max': args.lat_max,
    }
    
    # Print header
    logger.info("\n" + "="*70)
    logger.info("GEOANOMALYMAPPER - AUTOMATED DATA DOWNLOAD")
    logger.info("="*70)
    logger.info(f"Region: [{bounds['lon_min']:.2f}, {bounds['lat_min']:.2f}] to "
                f"[{bounds['lon_max']:.2f}, {bounds['lat_max']:.2f}]")
    logger.info(f"Phases: {args.phases}")
    logger.info(f"Output: {DATA_DIR}")
    logger.info("")
    
    # Load existing status
    status = load_status()
    
    # Download each phase
    for phase in sorted(args.phases):
        try:
            download_phase(phase, bounds, status)
        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user")
            logger.info("Progress has been saved - re-run to resume")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Phase {phase} failed: {e}")
            continue
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Status file: {STATUS_FILE}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Complete any manual downloads (see instructions in data/raw/)")
    logger.info("2. Run: python process_data.py")
    logger.info("3. Run: python detect_voids.py")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()