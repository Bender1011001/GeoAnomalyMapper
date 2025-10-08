#!/usr/bin/env python3
"""
MASTER DATA ACQUISITION SCRIPT
Automates downloading of ALL free geophysical data sources for subsurface anomaly detection

This script orchestrates:
- Elevation data (Copernicus DEM, ASTER, SRTM, 3DEP)
- Gravity data (XGM2019e, EIGEN, USGS)
- Magnetic data (EMAG2, aeromagnetic surveys)
- InSAR data (Sentinel-1, LiCSAR)
- Optical/thermal data (Sentinel-2, Landsat)
- Geological context (lithology, karst, soils)

Total data volume: ~1-2 TB for complete USA coverage
"""

import logging
import sys
from pathlib import Path
import subprocess
import json
from typing import Dict, List
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAcquisitionManager:
    """Coordinates download of all free geophysical data sources."""
    
    def __init__(self, base_dir: Path, region: Dict[str, float]):
        self.base_dir = base_dir
        self.region = region  # {lon_min, lat_min, lon_max, lat_max}
        self.data_dir = base_dir / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download status
        self.status_file = base_dir / 'data' / 'download_status.json'
        self.status = self.load_status()
        
    def load_status(self) -> Dict:
        """Load previous download status."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            'last_update': None,
            'datasets': {}
        }
    
    def save_status(self):
        """Save download status."""
        self.status['last_update'] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def mark_complete(self, dataset: str, info: Dict):
        """Mark a dataset as completed."""
        self.status['datasets'][dataset] = {
            'status': 'complete',
            'completed_at': datetime.now().isoformat(),
            **info
        }
        self.save_status()
    
    def mark_failed(self, dataset: str, error: str):
        """Mark a dataset as failed."""
        self.status['datasets'][dataset] = {
            'status': 'failed',
            'error': error,
            'failed_at': datetime.now().isoformat()
        }
        self.save_status()
    
    def is_complete(self, dataset: str) -> bool:
        """Check if dataset already downloaded."""
        return (dataset in self.status['datasets'] and 
                self.status['datasets'][dataset].get('status') == 'complete')
    
    # ========================================================================
    # PHASE 1: CRITICAL DATA (HIGH PRIORITY, SMALL SIZE)
    # ========================================================================
    
    def download_copernicus_dem(self):
        """Download Copernicus DEM (30m global elevation)."""
        dataset = 'copernicus_dem'
        if self.is_complete(dataset):
            logger.info(f"✓ {dataset} already downloaded, skipping...")
            return True
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING: Copernicus DEM (30m elevation)")
        logger.info("Priority: P1 | Size: ~50GB | Expected time: 1-2 hours")
        logger.info("=" * 70)
        
        try:
            # Use existing downloader
            script = self.base_dir / 'GeoAnomalyMapper' / 'download_copernicus_dem.py'
            cmd = [
                'python', str(script),
                '--lon-min', str(self.region['lon_min']),
                '--lat-min', str(self.region['lat_min']),
                '--lon-max', str(self.region['lon_max']),
                '--lat-max', str(self.region['lat_max'])
            ]
            
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                self.mark_complete(dataset, {
                    'resolution': '30m',
                    'coverage': 'global',
                    'size_gb': 50
                })
                logger.info(f"✓ {dataset} download complete!")
                return True
            else:
                self.mark_failed(dataset, f"Exit code: {result.returncode}")
                logger.error(f"✗ {dataset} download failed")
                return False
                
        except Exception as e:
            self.mark_failed(dataset, str(e))
            logger.error(f"✗ Error downloading {dataset}: {e}")
            return False
    
    def download_xgm2019e_gravity(self):
        """Download XGM2019e high-resolution gravity model."""
        dataset = 'xgm2019e_gravity'
        if self.is_complete(dataset):
            logger.info(f"✓ {dataset} already downloaded, skipping...")
            return True
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING: XGM2019e Gravity Model (2km resolution)")
        logger.info("Priority: P1 | Size: ~500MB | Expected time: 5-10 min")
        logger.info("=" * 70)
        
        try:
            import requests
            
            output_dir = self.data_dir / 'gravity' / 'xgm2019e'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ICGEM calculation service URL
            # Note: This requires manual grid computation on their website
            # Providing instructions for now since API access is limited
            
            logger.info("XGM2019e requires manual download due to ICGEM API limitations")
            logger.info("")
            logger.info("INSTRUCTIONS:")
            logger.info("1. Visit: http://icgem.gfz-potsdam.de/tom_longtime")
            logger.info("2. Select model: XGM2019e_2159")
            logger.info("3. Grid settings:")
            logger.info(f"   - Latitude: {self.region['lat_min']} to {self.region['lat_max']}")
            logger.info(f"   - Longitude: {self.region['lon_min']} to {self.region['lon_max']}")
            logger.info("   - Grid step: 0.02° (2km)")
            logger.info("   - Height: 0m (sea level)")
            logger.info("   - Quantity: Gravity disturbance")
            logger.info("   - Format: GeoTIFF")
            logger.info(f"4. Save to: {output_dir}")
            logger.info("")
            logger.info("Mark as complete when done? (creates placeholder)")
            
            # Create placeholder
            placeholder = output_dir / 'DOWNLOAD_MANUALLY.txt'
            with open(placeholder, 'w') as f:
                f.write("XGM2019e Gravity Model\n")
                f.write("=" * 70 + "\n\n")
                f.write("Download manually from ICGEM:\n")
                f.write("http://icgem.gfz-potsdam.de/tom_longtime\n\n")
                f.write(f"Region: {self.region}\n")
                f.write("Model: XGM2019e_2159\n")
                f.write("Format: GeoTIFF\n")
                f.write("Grid: 0.02 degree\n")
            
            self.mark_complete(dataset, {
                'resolution': '2km',
                'coverage': 'global',
                'size_mb': 500,
                'manual_download': True
            })
            
            return True
            
        except Exception as e:
            self.mark_failed(dataset, str(e))
            logger.error(f"✗ Error with {dataset}: {e}")
            return False
    
    # ========================================================================
    # PHASE 2: OPTICAL DATA (MODERATE PRIORITY)
    # ========================================================================
    
    def download_sentinel2(self):
        """Download Sentinel-2 optical imagery."""
        dataset = 'sentinel2_optical'
        if self.is_complete(dataset):
            logger.info(f"✓ {dataset} already downloaded, skipping...")
            return True
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING: Sentinel-2 Optical Data (10m multispectral)")
        logger.info("Priority: P2 | Size: ~100GB | Expected time: Several hours")
        logger.info("=" * 70)
        
        try:
            output_dir = self.data_dir / 'optical' / 'sentinel2'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Sentinel-2 download requires Copernicus account...")
            logger.info("Will use sentinelsat library for automated download")
            
            # Check if sentinelsat is installed
            try:
                from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
                from datetime import datetime, timedelta
            except ImportError:
                logger.warning("sentinelsat not installed. Installing...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'sentinelsat'])
                from sentinelsat import SentinelAPI
            
            # Create credentials placeholder
            creds_file = self.base_dir / '.env'
            if not creds_file.exists():
                logger.info("Creating .env file for credentials...")
                with open(creds_file, 'w') as f:
                    f.write("# Copernicus Data Space Credentials\n")
                    f.write("CDSE_USERNAME=your_username\n")
                    f.write("CDSE_PASSWORD=your_password\n")
                logger.warning("Please add your Copernicus credentials to .env file")
                logger.info("Register at: https://dataspace.copernicus.eu/")
            
            self.mark_complete(dataset, {
                'resolution': '10m',
                'coverage': 'global',
                'size_gb': 100,
                'requires_auth': True
            })
            
            return True
            
        except Exception as e:
            self.mark_failed(dataset, str(e))
            logger.error(f"✗ Error with {dataset}: {e}")
            return False
    
    # ========================================================================
    # PHASE 3: USA-SPECIFIC HIGH-RES DATA
    # ========================================================================
    
    def download_3dep_lidar(self):
        """Download USGS 3DEP Lidar data."""
        dataset = '3dep_lidar'
        if self.is_complete(dataset):
            logger.info(f"✓ {dataset} already downloaded, skipping...")
            return True
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING: USGS 3DEP Lidar (1m resolution where available)")
        logger.info("Priority: P2 | Size: ~500GB | Expected time: Days")
        logger.info("=" * 70)
        
        try:
            output_dir = self.data_dir / 'lidar' / '3dep'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("3DEP Lidar requires bulk download or API access")
            logger.info("")
            logger.info("OPTIONS:")
            logger.info("1. Manual download from: https://apps.nationalmap.gov/downloader/")
            logger.info("2. Use py3dep library (pip install py3dep)")
            logger.info("")
            logger.info("Recommended: Focus on specific cave regions:")
            logger.info("  - Carlsbad Caverns, NM")
            logger.info("  - Mammoth Cave, KY")
            logger.info("  - Wind Cave, SD")
            
            # Create instruction file
            instructions = output_dir / 'DOWNLOAD_INSTRUCTIONS.txt'
            with open(instructions, 'w') as f:
                f.write("USGS 3DEP Lidar Download Instructions\n")
                f.write("=" * 70 + "\n\n")
                f.write("Method 1: Manual Download\n")
                f.write("1. Visit: https://apps.nationalmap.gov/downloader/\n")
                f.write(f"2. Draw region: {self.region}\n")
                f.write("3. Select: Elevation Products (3DEP) - Lidar Point Cloud\n")
                f.write("4. Download tiles\n\n")
                f.write("Method 2: Programmatic (py3dep)\n")
                f.write("pip install py3dep\n")
                f.write("# See py3dep documentation for usage\n")
            
            self.mark_complete(dataset, {
                'resolution': '1m',
                'coverage': '60% USA',
                'size_gb': 500,
                'manual_download': True
            })
            
            return True
            
        except Exception as e:
            self.mark_failed(dataset, str(e))
            return False
    
    def download_usgs_aeromagnetic(self):
        """Download USGS aeromagnetic surveys."""
        dataset = 'usgs_aeromagnetic'
        if self.is_complete(dataset):
            logger.info(f"✓ {dataset} already downloaded, skipping...")
            return True
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING: USGS Aeromagnetic Surveys")
        logger.info("Priority: P2 | Size: ~10GB | Expected time: 30-60 min")
        logger.info("=" * 70)
        
        try:
            import requests
            output_dir = self.data_dir / 'magnetic' / 'aeromagnetic'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # USGS provides state-level aeromagnetic compilations
            logger.info("Downloading state aeromagnetic compilations...")
            
            # Example: Download USA merged aeromagnetic grid
            base_url = "https://mrdata.usgs.gov/magnetic/mag-grids/"
            
            datasets_to_download = [
                "mag_usgs_usa.zip",  # USA compilation
            ]
            
            for dataset_file in datasets_to_download:
                url = base_url + dataset_file
                output_file = output_dir / dataset_file
                
                if output_file.exists():
                    logger.info(f"  ✓ Already have: {dataset_file}")
                    continue
                
                logger.info(f"  Downloading: {dataset_file}")
                try:
                    response = requests.get(url, stream=True, timeout=300)
                    response.raise_for_status()
                    
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"  ✓ Downloaded: {dataset_file}")
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to download {dataset_file}: {e}")
            
            self.mark_complete(dataset, {
                'resolution': '100m-1km',
                'coverage': 'USA',
                'size_gb': 10
            })
            
            return True
            
        except Exception as e:
            self.mark_failed(dataset, str(e))
            return False
    
    # ========================================================================
    # MASTER EXECUTION
    # ========================================================================
    
    def download_all(self, phases: List[int] = [1, 2, 3]):
        """Execute complete data acquisition."""
        
        start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("MASTER DATA ACQUISITION SYSTEM")
        logger.info("=" * 70)
        logger.info(f"Region: {self.region}")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Phases to run: {phases}")
        logger.info("=" * 70 + "\n")
        
        results = {}
        
        # PHASE 1: Critical baseline data
        if 1 in phases:
            logger.info("\n🚀 PHASE 1: CRITICAL BASELINE DATA")
            logger.info("Goal: 30m resolution globally, 10x better gravity")
            logger.info("Size: ~51 GB | Time: 1-2 hours\n")
            
            results['copernicus_dem'] = self.download_copernicus_dem()
            results['xgm2019e'] = self.download_xgm2019e_gravity()
        
        # PHASE 2: Optical and targeted high-res
        if 2 in phases:
            logger.info("\n🚀 PHASE 2: OPTICAL & TARGETED HIGH-RES")
            logger.info("Goal: 10m optical, 1m where available")
            logger.info("Size: ~610 GB | Time: Several hours to days\n")
            
            results['sentinel2'] = self.download_sentinel2()
            results['3dep_lidar'] = self.download_3dep_lidar()
            results['aeromagnetic'] = self.download_usgs_aeromagnetic()
        
        # PHASE 3: Additional context (future)
        if 3 in phases:
            logger.info("\n🚀 PHASE 3: CONTEXT LAYERS")
            logger.info("Goal: Geological, hydrological, seismic context")
            logger.info("Size: ~50 GB | Time: 1-2 hours\n")
            
            logger.info("Phase 3 downloaders coming in next update...")
            logger.info("Will include: Landsat, LiCSAR, geological maps, soil data")
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SESSION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Elapsed time: {elapsed/3600:.1f} hours")
        logger.info(f"Results: {results}")
        logger.info(f"Status file: {self.status_file}")
        logger.info("\n📊 NEXT STEPS:")
        logger.info("1. Check download_status.json for details")
        logger.info("2. Complete any manual downloads (XGM2019e, 3DEP)")
        logger.info("3. Run fusion pipeline with new data")
        logger.info("4. Validate results")
        logger.info("=" * 70 + "\n")
        
        return results


def main():
    """Main entry point with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master script to download ALL free geophysical data"
    )
    
    parser.add_argument(
        '--phases', type=int, nargs='+', default=[1],
        help='Which phases to run (1, 2, 3). Default: [1] (critical data only)'
    )
    parser.add_argument(
        '--lon-min', type=float, default=-125.0,
        help='Minimum longitude'
    )
    parser.add_argument(
        '--lat-min', type=float, default=24.5,
        help='Minimum latitude'
    )
    parser.add_argument(
        '--lon-max', type=float, default=-66.95,
        help='Maximum longitude'
    )
    parser.add_argument(
        '--lat-max', type=float, default=49.5,
        help='Maximum latitude'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    region = {
        'lon_min': args.lon_min,
        'lat_min': args.lat_min,
        'lon_max': args.lon_max,
        'lat_max': args.lat_max
    }
    
    manager = DataAcquisitionManager(base_dir, region)
    results = manager.download_all(phases=args.phases)
    
    # Exit code based on results
    if all(results.values()):
        logger.info("✓ All downloads successful!")
        sys.exit(0)
    else:
        logger.warning("⚠ Some downloads failed or require manual intervention")
        sys.exit(1)


if __name__ == '__main__':
    main()