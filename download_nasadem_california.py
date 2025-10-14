#!/usr/bin/env python3
"""
NASADEM California Elevation Data Downloader
=============================================

Downloads NASADEM 30m elevation tiles for California from NASA Earthdata.

Requires NASA Earthdata account (free): https://urs.earthdata.nasa.gov/users/new

Usage:
    # Using .netrc file (recommended):
    python download_nasadem_california.py

    # Using username/password:
    python download_nasadem_california.py --username YOUR_USERNAME --password YOUR_PASSWORD

    # Custom region (default is California):
    python download_nasadem_california.py --bbox -125,25,-66,49  # Full USA

Authentication:
    - Option 1 (Recommended): Create ~/.netrc with:
        machine urs.earthdata.nasa.gov
        login YOUR_USERNAME
        password YOUR_PASSWORD
    - Option 2: Pass --username and --password flags
    - Option 3: Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars
"""

import argparse
import logging
import sys
from pathlib import Path
import requests
from typing import List, Tuple
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: define a no-op tqdm to avoid network installs in restricted environments
    class _Tqdm:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def update(self, n=1):
            pass
    def tqdm(iterable=None, **kwargs):
        # If used as context manager: tqdm(total=..., desc=...)
        if iterable is None:
            return _Tqdm(**kwargs)
        # If used to wrap an iterable: just return the iterable
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nasadem_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elevation" / "nasadem"

# NASA LP DAAC NASADEM base URL
NASADEM_BASE = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11"


class NASADEMDownloader:
    """Downloads NASADEM elevation tiles with NASA Earthdata authentication."""
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        token: str = None,
        output_dir: Path = None
    ):
        self.username = username or os.getenv('EARTHDATA_USERNAME')
        self.password = password or os.getenv('EARTHDATA_PASSWORD')
        self.token = token or os.getenv('EARTHDATA_TOKEN')
        self.output_dir = output_dir or DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session with authentication and retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GeoAnomalyMapper/1.0 (+https://example.local)"
        })

        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("Using Earthdata bearer token for authentication")
        else:
            # Resolve credentials: args > env vars > netrc/ _netrc / NETRC env path
            if self.username and self.password:
                self.session.auth = (self.username, self.password)
            else:
                try:
                    import netrc as _netrc
                    netrc_candidates = [
                        os.getenv('NETRC'),
                        os.path.expanduser('~/.netrc'),
                        os.path.expanduser('~/_netrc'),
                    ]
                    netrc_path = next(
                        (p for p in netrc_candidates if p and os.path.exists(p)),
                        None
                    )
                    if netrc_path:
                        os.environ['NETRC'] = netrc_path  # help requests auto-pick it up
                        n = _netrc.netrc(netrc_path)
                        auth_info = n.authenticators("urs.earthdata.nasa.gov")
                        if auth_info:
                            self.session.auth = (auth_info[0], auth_info[2])
                            logger.info(f"Using netrc at {netrc_path} for authentication")
                        else:
                            logger.warning("No netrc entry found for urs.earthdata.nasa.gov")
                    else:
                        logger.warning("No netrc file found - attempting anonymous access (likely to fail)")
                except Exception as e:
                    logger.warning(f"Unable to read netrc: {e}")

        # Configure conservative retries for transient errors
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            self.session.mount('https://', HTTPAdapter(max_retries=retries))
            self.session.mount('http://', HTTPAdapter(max_retries=retries))
        except Exception:
            pass
        
        self.failed_tiles = []
        
    def generate_tile_list(self, bbox: Tuple[float, float, float, float]) -> List[str]:
        """
        Generate list of NASADEM tile names from bounding box.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max) in decimal degrees
        
        Returns:
            List of tile names like ['n32w117', 'n33w118', ...]
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        
        tiles = []
        for lat in range(int(lat_min), int(lat_max) + 1):
            for lon in range(int(abs(lon_max)), int(abs(lon_min)) + 1):
                # NASADEM tiles use floor of coordinates
                lat_str = f"n{lat:02d}" if lat >= 0 else f"s{abs(lat):02d}"
                
                # For western hemisphere (negative longitude), use w prefix
                if lon_min < 0 and lon_max < 0:  # Western hemisphere
                    lon_str = f"w{lon:03d}"
                else:
                    lon_str = f"e{lon:03d}" if lon >= 0 else f"w{abs(lon):03d}"
                
                tile_name = f"{lat_str}{lon_str}"
                tiles.append(tile_name)
        
        logger.info(f"Generated {len(tiles)} tiles for bounding box {bbox}")
        return tiles
    
    def download_tile(self, tile_name: str) -> bool:
        """
        Download a single NASADEM tile.
        
        Args:
            tile_name: Tile identifier like 'n32w117'
        
        Returns:
            True if successful, False otherwise
        """
        zip_filename = f"NASADEM_HGT_{tile_name}.zip"
        url = f"{NASADEM_BASE}/{zip_filename}"
        
        tile_dir = self.output_dir / tile_name
        zip_path = self.output_dir / zip_filename
        
        # Skip if already downloaded and extracted
        if tile_dir.exists() and any(tile_dir.glob("*.hgt")):
            logger.debug(f"Tile {tile_name} already exists, skipping")
            return True
        
        # Skip if zip exists and is valid
        if zip_path.exists() and zip_path.stat().st_size > 1000000:  # > 1MB
            logger.debug(f"Zip for {tile_name} exists, attempting extraction")
            try:
                self._extract_tile(zip_path, tile_dir)
                return True
            except Exception as e:
                logger.warning(f"Existing zip corrupted for {tile_name}: {e}, re-downloading")
                zip_path.unlink()
        
        try:
            logger.info(f"Downloading {tile_name}...")
            # Attempt authenticated GET with redirects enabled (URS flow)
            response = self.session.get(url, stream=True, timeout=60, allow_redirects=True)
            
            if response.status_code == 404:
                logger.warning(f"Tile {tile_name} not found (likely ocean/no data)")
                return False
            
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
            
            logger.info(f"Downloaded {tile_name} ({zip_path.stat().st_size / (1024**2):.1f} MB)")
            
            # Extract immediately
            self._extract_tile(zip_path, tile_dir)
            
            # Clean up zip to save space (optional - comment out to keep)
            # zip_path.unlink()
            
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"Authentication failed for {tile_name}. Ensure Earthdata credentials are correct and NASADEM terms accepted.")
            else:
                logger.error(f"HTTP error downloading {tile_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to download {tile_name}: {e}")
            return False
    
    def _extract_tile(self, zip_path: Path, tile_dir: Path):
        """Extract NASADEM zip file."""
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tile_dir)
        
        logger.info(f"Extracted {zip_path.name} to {tile_dir}")
    
    def download_all(self, tiles: List[str], max_workers: int = 5) -> dict:
        """
        Download all tiles in parallel.
        
        Args:
            tiles: List of tile names
            max_workers: Number of parallel downloads
        
        Returns:
            Dict with success/failure counts
        """
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Starting download of {len(tiles)} tiles with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_tile = {
                executor.submit(self.download_tile, tile): tile 
                for tile in tiles
            }
            
            # Process results as they complete
            with tqdm(total=len(tiles), desc="Downloading NASADEM tiles") as pbar:
                for future in as_completed(future_to_tile):
                    tile = future_to_tile[future]
                    try:
                        success = future.result()
                        if success:
                            results['success'] += 1
                        else:
                            results['failed'] += 1
                            self.failed_tiles.append(tile)
                    except Exception as e:
                        logger.error(f"Exception processing {tile}: {e}")
                        results['failed'] += 1
                        self.failed_tiles.append(tile)
                    
                    pbar.update(1)
        
        # Save failed tiles for retry
        if self.failed_tiles:
            failed_file = self.output_dir / "failed_tiles.json"
            with open(failed_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'tiles': self.failed_tiles
                }, f, indent=2)
            logger.warning(f"Failed tiles saved to {failed_file}")
        
        return results
    
    def create_status_report(self, results: dict, bbox: Tuple[float, float, float, float]):
        """Create a download status report."""
        report_file = self.output_dir / "download_report.md"
        
        total = results['success'] + results['failed']
        
        report = f"""# NASADEM California Download Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Download Summary
- **Region**: {bbox[0]}° to {bbox[2]}° lon, {bbox[1]}° to {bbox[3]}° lat
- **Total Tiles Attempted**: {total}
- **Successfully Downloaded**: {results['success']}
- **Failed/Not Found**: {results['failed']}
- **Success Rate**: {(results['success']/total*100) if total > 0 else 0:.1f}%

## Output Directory
- **Location**: `{self.output_dir.relative_to(PROJECT_ROOT)}`
- **Format**: Extracted .hgt files (30m elevation) and .num files (metadata)

## Failed Tiles
"""
        
        if self.failed_tiles:
            report += "The following tiles could not be downloaded (likely ocean/no land coverage):\n"
            for tile in self.failed_tiles:
                report += f"- {tile}\n"
        else:
            report += "All tiles downloaded successfully!\n"
        
        report += f"""
## Next Steps
1. **Verify Data**: Check that .hgt files exist in subdirectories
2. **Process Data**: Run `python GeoAnomalyMapper/process_data.py` to integrate with project
3. **Quality Check**: Use QGIS or `gdalinfo` to inspect tile quality

## Integration
NASADEM tiles are ready for use in void detection pipeline. The `process_data.py` script will:
- Mosaic all tiles into a single California DEM
- Reproject to common CRS (EPSG:4326)
- Resample to match gravity/magnetic data resolution
- Use for topographic correction in anomaly detection

**File Format**: SRTM .hgt format (16-bit signed integer, height in meters, WGS84/EGM96 geoid)
**Resolution**: 1 arc-second (~30m at equator, ~23-26m in California)
**Coverage**: Land areas 60°N to 56°S (NASADEM reprocessed SRTM with void-filling)
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Status report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download NASADEM elevation data for California (or custom region)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--bbox',
        type=str,
        default="-124.5,32.5,-114,42",
        help="Bounding box as 'lon_min,lat_min,lon_max,lat_max' (default: California)"
    )
    
    parser.add_argument(
        '--username',
        type=str,
        help="NASA Earthdata username (or use .netrc)"
    )
    
    parser.add_argument(
        '--password',
        type=str,
        help="NASA Earthdata password (or use .netrc)"
    )

    parser.add_argument(
        '--token',
        type=str,
        help="Earthdata bearer token (alternative to username/password)"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})"
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help="Number of parallel downloads (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Parse bounding box
    try:
        bbox = tuple(map(float, args.bbox.split(',')))
        if len(bbox) != 4:
            raise ValueError
    except ValueError:
        logger.error("Invalid bbox format. Use: lon_min,lat_min,lon_max,lat_max")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("NASADEM ELEVATION DATA DOWNLOADER")
    logger.info("="*70)
    logger.info(f"Region: {bbox}")
    logger.info(f"Output: {args.output}")
    
    # Check authentication (token, username/password, or netrc-based)
    env_token = os.getenv('EARTHDATA_TOKEN')
    has_token = bool(args.token or env_token)
    
    env_user = os.getenv('EARTHDATA_USERNAME')
    env_pass = os.getenv('EARTHDATA_PASSWORD')
    has_basic = (
        (args.username and (args.password or env_pass))
        or (env_user and env_pass)
    )

    netrc_candidates = [
        os.getenv('NETRC'),
        os.path.expanduser('~/.netrc'),
        os.path.expanduser('~/_netrc'),
    ]
    has_netrc = any(path and os.path.exists(path) for path in netrc_candidates)

    if not (has_token or has_basic or has_netrc):
        logger.error("No authentication provided!")
        logger.error("Either:")
        logger.error("  1. Create ~/.netrc with NASA Earthdata credentials")
        logger.error("  2. Pass --username and --password")
        logger.error("  3. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars")
        logger.error("  4. Set NETRC env var to a .netrc file path")
        logger.error("  5. Pass --token or set EARTHDATA_TOKEN env var")
        logger.error("\nRegister free account at: https://urs.earthdata.nasa.gov/users/new")
        sys.exit(1)
    
    # Initialize downloader
    downloader = NASADEMDownloader(
        username=args.username,
        password=args.password,
        token=args.token,
        output_dir=args.output
    )
    
    # Generate tile list
    tiles = downloader.generate_tile_list(bbox)
    
    logger.info(f"Estimated download size: ~{len(tiles) * 7} MB")
    logger.info(f"Estimated extraction size: ~{len(tiles) * 20} MB")
    
    # Download all tiles
    results = downloader.download_all(tiles, max_workers=args.workers)
    
    # Create report
    downloader.create_status_report(results, bbox)
    
    logger.info("="*70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Success: {results['success']}/{len(tiles)} tiles")
    logger.info(f"Output: {args.output}")
    logger.info("Check download_report.md for details")


if __name__ == "__main__":
    main()
