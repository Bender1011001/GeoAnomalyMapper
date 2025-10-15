#!/usr/bin/env python3
"""
Unified Data Agent for GeoAnomalyMapper.
Handles automated acquisition of geophysical datasets (gravity, magnetic, InSAR, DEM)
with CLI interface, dry-run support, and integration with RobustDownloader and data_status.json.
Respects GAM_USE_V2_CONFIG for backward compatibility with config/paths shims.
Does not write to docs/ directory.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Core imports for integration
from utils import paths_shim  # Backward-compatible path access
from utils.error_handling import RobustDownloader  # Resilient downloads

# Setup logging consistent with codebase
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(paths_shim.get_data_dir() / 'data_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Dataset configurations (free/public sources; expand as needed)
DATASETS = {
    'emag2_magnetic': {
        'url': 'https://www.ncei.noaa.gov/pub/geodetics/geomag/EMAG2_10/EMAG2_V3_SeaLevel_DataTiff.tif',
        'checksum': None,  # Add SHA256 if available
        'filename': 'EMAG2_V3_SeaLevel_DataTiff.tif',
        'size_mb': 84.5,
        'resolution': '~2 km',
        'coverage': 'Global',
        'bbox_support': False,  # Global, no bbox filter
        'dir': 'emag2'
    },
    'egm2008_gravity': {
        'url': 'https://earth-info.nga.mil/portals/165/products-services/geospatial-products-and-services/gravity/geoid-models/egm2008/egm2008_geoid.zip',  # Example; adjust to direct TIFF if available
        'checksum': None,
        'filename': 'egm2008_geoid.tiff',  # Assume extracted/converted
        'size_mb': 0.5,
        'resolution': '~20 km',
        'coverage': 'Global',
        'bbox_support': False,
        'dir': 'gravity'
    },
    'xgm2019e_gravity': {
        'url': 'https://icgem.gfz-potsdam.de/home',  # Coefficients; requires manual download or API; placeholder for grid
        'checksum': None,
        'filename': 'XGM2019e_2159.gfc',
        'size_mb': 10.0,  # Approx
        'resolution': '~2 km',
        'coverage': 'Global (requires conversion)',
        'bbox_support': False,
        'dir': 'gravity',
        'note': 'Download coefficients; use convert_xgm2019e_to_grid.py for grid'
    },
    'srtm_dem': {  # Fallback for Copernicus
        'url': 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTM_20m_Digital_Elevation/SRTM_20m_001_20200313T000000_20200313T000000.hgt.zip',  # Example tile; bbox-based
        'checksum': None,
        'filename': 'srtm_carlsbad_30m.tif',  # Post-clip
        'size_mb': 5.0,
        'resolution': '30 m',
        'coverage': 'Global land',
        'bbox_support': True,
        'dir': 'elevation/srtm'
    },
    'insar_sentinel1': {
        'url': 'https://scihub.copernicus.eu/dhus/search?q=platformname:"Sentinel-1" AND producttype:"GRD" AND footprint:"Intersects(BBOX({bbox}))"',  # API query
        'checksum': None,
        'filename': 'sentinel1_scene.zip',  # Multiple
        'size_mb': 500.0,  # Per scene
        'resolution': '~20 m',
        'coverage': 'Global (requires auth/processing)',
        'bbox_support': True,
        'dir': 'insar/sentinel1',
        'auth_service': 'copernicus'  # If tokens configured
    }
}

def load_status_json(status_path: Path) -> Dict[str, Any]:
    """Load data_status.json with backward compatibility."""
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
        logger.info(f"Loaded existing status from {status_path}")
        return status
    else:
        logger.info(f"No existing status file; starting fresh")
        return {}

def save_status_json(status: Dict[str, Any], status_path: Path) -> None:
    """Save updated status to data_status.json."""
    # Ensure backward compatibility: preserve existing keys not touched
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)
    logger.info(f"Updated status saved to {status_path}")

def update_dataset_status(status: Dict[str, Any], dataset_key: str, output_path: Path, success: bool, downloader_size: int = 0) -> None:
    """Update status for a dataset post-download."""
    size_mb = downloader_size / (1024 * 1024) if downloader_size else 0
    now = datetime.now().isoformat()
    dataset_info = DATASETS.get(dataset_key, {})
    
    status[dataset_key] = {
        **status.get(dataset_key, {}),  # Preserve existing
        'available': success,
        'path': str(output_path) if success else None,
        'size_mb': round(size_mb, 1),
        'resolution': dataset_info.get('resolution'),
        'coverage': dataset_info.get('coverage'),
        'download_timestamp': now if success else None,
        'status': 'downloaded' if success else 'failed'
    }
    if not success:
        logger.warning(f"Failed to download {dataset_key}; status marked as failed")

def download_dataset(downloader: RobustDownloader, dataset_key: str, output_dir: Path, bbox: tuple = None, dry_run: bool = False) -> bool:
    """Download a single dataset, respecting bbox and dry-run."""
    dataset = DATASETS[dataset_key]
    if bbox and not dataset.get('bbox_support', False):
        logger.info(f"{dataset_key} is global; ignoring bbox")
        bbox = None
    
    subdir = Path(dataset['dir'])
    output_path = output_dir / subdir / dataset['filename']
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would download {dataset_key} from {dataset['url']} to {output_path}")
        if bbox:
            logger.info(f"[DRY-RUN] Would filter by bbox: {bbox}")
        return True, 0  # Simulate success with dummy size
    
    if output_path.exists():
        logger.info(f"{dataset_key} already exists at {output_path}; skipping")
        return True
    
    try:
        # For bbox-supported (e.g., Sentinel-1), query API first (simplified; expand with real API)
        if bbox and dataset_key == 'insar_sentinel1':
            # Placeholder: Real impl would use Copernicus API to get scene URLs
            logger.info(f"Querying Sentinel-1 scenes for bbox {bbox}...")
            # Assume fetches list of URLs; here simulate one
            urls = [dataset['url']]  # Replace with actual query
        else:
            urls = [dataset['url']]
        
        success = False
        for url in urls:
            desc = f"{dataset_key}: {dataset.get('resolution', 'N/A')}"
            auth_service = dataset.get('auth_service')
            download_success = downloader.download_with_retry(
                url, output_path, desc=desc, auth_service=auth_service,
                expected_size=int(dataset['size_mb'] * 1024 * 1024) if dataset['size_mb'] else None,
                checksum=dataset.get('checksum')
            )
            if download_success:
                success = True
                break
        
        if success:
            actual_size = output_path.stat().st_size
            logger.info(f"Successfully downloaded {dataset_key}: {actual_size / (1024*1024):.1f} MB")
            return True, actual_size
        else:
            raise Exception("Download failed after retries")
            
    except Exception as e:
        logger.error(f"Failed to download {dataset_key}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False, 0

def status_command(args):
    """Handle 'status' subcommand."""
    status_path = paths_shim.get_data_dir() / 'data_status.json'
    status = load_status_json(status_path)
    
    print("GeoAnomalyMapper Data Status Report")
    print("=" * 40)
    for key, info in status.items():
        avail = "✅" if info.get('available', False) else "❌"
        path = info.get('path', 'N/A')
        size = f"{info.get('size_mb', 0):.1f} MB" if info.get('size_mb') else 'N/A'
        print(f"{avail} {key}: {path} ({size})")
        if info.get('status'):
            print(f"   Status: {info['status']}")
        if info.get('download_timestamp'):
            print(f"   Downloaded: {info['download_timestamp']}")
        print()
    
    if args.report:
        # Enhanced report: Save to output dir (not docs/)
        report_path = paths_shim.get_output_dir() / 'data_status_report.md'
        with open(report_path, 'w') as f:
            f.write("# Data Status Report\n\n")
            for key, info in status.items():
                f.write(f"## {key}\n")
                f.write(f"- Available: {info.get('available', False)}\n")
                f.write(f"- Path: {info.get('path', 'N/A')}\n")
                # ... more details
        logger.info(f"Report saved to {report_path}")

def download_command(args):
    """Handle 'download' subcommand."""
    status_path = paths_shim.get_data_dir() / 'data_status.json'
    status = load_status_json(status_path)
    raw_dir = paths_shim.get_data_dir() / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    downloader = RobustDownloader(
        max_retries=5,
        base_delay=2.0,
        timeout=(10, 60),
        circuit_breaker=True
    )
    
    # For 'free' mode, download all free datasets
    if args.mode == 'free':
        to_download = [k for k in DATASETS if 'auth_service' not in DATASETS[k] or not DATASETS[k]['auth_service']]
    else:
        to_download = [args.dataset] if args.dataset else list(DATASETS.keys())
    
    bbox = None
    if args.bbox:
        try:
            lon1, lat1, lon2, lat2 = map(float, args.bbox.split(','))
            bbox = (lon1, lat1, lon2, lat2)
        except ValueError:
            logger.error("Invalid bbox format; use 'lon1,lat1,lon2,lat2'")
            return
    
    success_count = 0
    for dataset_key in to_download:
        if dataset_key not in DATASETS:
            logger.warning(f"Unknown dataset: {dataset_key}")
            continue
        
        success, size = download_dataset(downloader, dataset_key, raw_dir, bbox, args.dry_run)
        output_path = raw_dir / DATASETS[dataset_key]['dir'] / DATASETS[dataset_key]['filename']
        update_dataset_status(status, dataset_key, output_path, success, size)
        if success:
            success_count += 1
    
    save_status_json(status, status_path)
    logger.info(f"Download complete: {success_count}/{len(to_download)} successful")
    
    downloader.close()

def main():
    parser = argparse.ArgumentParser(
        description="GeoAnomalyMapper Unified Data Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gam_data_agent.py status --report
  python gam_data_agent.py download free --bbox "-105,32,-104,33" --dry-run
  python gam_data_agent.py download insar_sentinel1 --dry-run
        """
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Status subcommand
    status_parser = subparsers.add_parser('status', help='Show dataset status')
    status_parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    # Download subcommand
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without executing')
    download_parser.add_argument('--bbox', help='Bounding box: "lon1,lat1,lon2,lat2" (e.g., "-105,32,-104,33")')
    download_parser.add_argument('mode', nargs='?', default='free', choices=['free', 'all'], help='Download mode (default: free)')
    download_parser.add_argument('dataset', nargs='?', help='Specific dataset (e.g., emag2_magnetic)')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        status_command(args)
    elif args.command == 'download':
        download_command(args)

if __name__ == '__main__':
    # Ensure no writes to docs/
    if 'docs' in str(paths_shim.get_output_dir()):
        logger.error("Output dir points to docs/; aborting to preserve GitHub Pages")
        sys.exit(1)
    
    main()