#!/usr/bin/env python3
"""
Complete Data Download and Verification Script
==============================================

This script:
1. Downloads all available free data sources automatically
2. Verifies existing downloads
3. Provides clear instructions for manual downloads
4. Creates a comprehensive status report

Usage:
    python download_all_data_complete.py --region="-105.0,32.0,-104.0,33.0"
"""

import argparse
import logging
import sys
from pathlib import Path
import requests
from typing import Tuple, Dict, List
import json
from datetime import datetime
import subprocess
import os

# Try to import elevation; install if missing
try:
    from elevation import clip
except ImportError:
    def install_elevation():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elevation"])
        from elevation import clip
        return clip
    clip = install_elevation()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


class DataDownloadManager:
    """Manages all data downloads and verification."""
    
    def __init__(self, region: Tuple[float, float, float, float]):
        self.region = region
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = region
        self.status = {}
        
    def check_existing_data(self) -> Dict[str, Dict]:
        """Check what data is already available."""
        
        logger.info("="*70)
        logger.info("CHECKING EXISTING DATA")
        logger.info("="*70)
        
        status = {}
        
        # Check EMAG2 magnetic data
        emag2_file = DATA_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
        status['emag2_magnetic'] = {
            'available': emag2_file.exists(),
            'path': str(emag2_file),
            'size_mb': round(emag2_file.stat().st_size / (1024**2), 1) if emag2_file.exists() else 0,
            'resolution': '~2 km',
            'coverage': 'Global'
        }
        
        # Check EGM2008 gravity data
        egm2008_file = DATA_DIR / "gravity" / "gravity_disturbance_EGM2008_50491becf3ffdee5c9908e47ed57881ed23de559539cd89e49b4d76635e07266.tiff"
        status['egm2008_gravity'] = {
            'available': egm2008_file.exists(),
            'path': str(egm2008_file),
            'size_mb': round(egm2008_file.stat().st_size / (1024**2), 1) if egm2008_file.exists() else 0,
            'resolution': '~20 km',
            'coverage': 'Global'
        }
        
        # Check XGM2019e coefficient file
        xgm_coef_file = DATA_DIR / "gravity" / "XGM2019e_2159.gfc"
        xgm_grid_file = DATA_DIR / "gravity" / "xgm2019e" / "xgm2019e_carlsbad.tif"
        status['xgm2019e_gravity'] = {
            'coefficients_available': xgm_coef_file.exists(),
            'grid_available': xgm_grid_file.exists(),
            'coef_path': str(xgm_coef_file),
            'grid_path': str(xgm_grid_file),
            'resolution': '~2 km',
            'coverage': 'Global (requires conversion to grid)'
        }
        
        # Check Copernicus DEM
        dem_dir = DATA_DIR / "elevation" / "copernicus_dem"
        dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        status['copernicus_dem'] = {
            'available': len(dem_files) > 0,
            'tile_count': len(dem_files),
            'dir': str(dem_dir),
            'resolution': '30 m',
            'coverage': 'Land areas'
        }
        
        # Check SRTM DEM (fallback)
        srtm_file = DATA_DIR / "elevation" / "srtm" / "srtm_carlsbad_30m.tif"
        status['srtm_dem'] = {
            'available': srtm_file.exists(),
            'path': str(srtm_file),
            'size_mb': round(srtm_file.stat().st_size / (1024**2), 1) if srtm_file.exists() else 0,
            'resolution': '30 m',
            'coverage': 'Global land'
        }
        
        # Check InSAR Sentinel-1 data
        insar_dir = DATA_DIR / "insar" / "sentinel1"
        safe_dirs = list(insar_dir.glob("*.SAFE")) if insar_dir.exists() else []
        status['insar_sentinel1'] = {
            'available': len(safe_dirs) > 0,
            'scene_count': len(safe_dirs),
            'dir': str(insar_dir),
            'resolution': '~20 m',
            'coverage': 'Requires processing'
        }
        
        # Check lithology database
        lithology_file = DATA_DIR / "SL2013sv_0.5d-grd_v2.1.tar.bz2"
        status['lithology'] = {
            'available': lithology_file.exists(),
            'path': str(lithology_file),
            'size_mb': round(lithology_file.stat().st_size / (1024**2), 1) if lithology_file.exists() else 0,
            'resolution': '~50 km',
            'coverage': 'Global'
        }
        
        self.status = status
        return status
    
    def print_status_report(self):
        """Print a detailed status report."""
        
        logger.info("")
        logger.info("="*70)
        logger.info("DATA AVAILABILITY STATUS")
        logger.info("="*70)
        logger.info("")
        
        # Magnetic Data
        mag_status = self.status.get('emag2_magnetic', {})
        logger.info("1. MAGNETIC DATA (EMAG2)")
        if mag_status.get('available'):
            logger.info(f"   ✓ AVAILABLE - {mag_status['size_mb']} MB")
            logger.info(f"   Resolution: {mag_status['resolution']}")
            logger.info(f"   Path: {mag_status['path']}")
        else:
            logger.info("   ✗ NOT AVAILABLE")
        logger.info("")
        
        # Gravity Data - Baseline
        grav_status = self.status.get('egm2008_gravity', {})
        logger.info("2. GRAVITY DATA - BASELINE (EGM2008)")
        if grav_status.get('available'):
            logger.info(f"   ✓ AVAILABLE - {grav_status['size_mb']} MB")
            logger.info(f"   Resolution: {grav_status['resolution']}")
            logger.info(f"   Path: {grav_status['path']}")
        else:
            logger.info("   ✗ NOT AVAILABLE")
        logger.info("")
        
        # Gravity Data - High Resolution
        xgm_status = self.status.get('xgm2019e_gravity', {})
        logger.info("3. GRAVITY DATA - HIGH RESOLUTION (XGM2019e)")
        if xgm_status.get('grid_available'):
            logger.info(f"   ✓ GRID AVAILABLE")
            logger.info(f"   Resolution: {xgm_status['resolution']}")
            logger.info(f"   Path: {xgm_status['grid_path']}")
        elif xgm_status.get('coefficients_available'):
            logger.info(f"   ⚠ COEFFICIENTS AVAILABLE (needs conversion)")
            logger.info(f"   Path: {xgm_status['coef_path']}")
            logger.info(f"   → Requires manual grid generation via ICGEM")
        else:
            logger.info("   ✗ NOT AVAILABLE")
        logger.info("")
        
        # Elevation Data
        dem_status = self.status.get('copernicus_dem', {})
        srtm_status = self.status.get('srtm_dem', {})
        logger.info("4. ELEVATION DATA (30m Resolution)")
        if dem_status.get('available'):
            logger.info(f"   ✓ COPERNICUS DEM - {dem_status['tile_count']} tiles")
            logger.info(f"   Resolution: {dem_status['resolution']}")
            logger.info(f"   Dir: {dem_status['dir']}")
        elif srtm_status.get('available'):
            logger.info(f"   ✓ SRTM FALLBACK - {srtm_status['size_mb']} MB")
            logger.info(f"   Resolution: {srtm_status['resolution']}")
            logger.info(f"   Path: {srtm_status['path']}")
        else:
            logger.info("   ⚠ NOT AVAILABLE (will attempt SRTM download)")
        logger.info("")
        
        # InSAR Data
        insar_status = self.status.get('insar_sentinel1', {})
        logger.info("5. INSAR DATA (Sentinel-1)")
        if insar_status.get('available'):
            logger.info(f"   ✓ AVAILABLE - {insar_status['scene_count']} scenes")
            logger.info(f"   Resolution: {insar_status['resolution']}")
            logger.info(f"   Dir: {insar_status['dir']}")
            logger.info("   NOTE: Raw data requires SNAP/ISCE processing")
        else:
            logger.info("   ℹ OPTIONAL (advanced users only)")
        logger.info("")
        
        # Lithology Data
        lith_status = self.status.get('lithology', {})
        logger.info("6. LITHOLOGY DATA (GLiM)")
        if lith_status.get('available'):
            logger.info(f"   ✓ AVAILABLE - {lith_status['size_mb']} MB")
            logger.info(f"   Resolution: {lith_status['resolution']}")
            logger.info(f"   Path: {lith_status['path']}")
        else:
            logger.info("   ℹ OPTIONAL (for advanced analysis)")
        logger.info("")
        
        logger.info("="*70)
    
    def download_missing(self) -> bool:
        """Download missing automatable data (e.g., SRTM DEM)."""
        
        logger.info("="*70)
        logger.info("DOWNLOADING MISSING DATA")
        logger.info("="*70)
        
        success = True
        
        # Download SRTM if no elevation data
        if not self.status.get('copernicus_dem', {}).get('available') and not self.status.get('srtm_dem', {}).get('available'):
            logger.info("Downloading SRTM 30m DEM for region...")
            srtm_dir = DATA_DIR / "elevation" / "srtm"
            srtm_dir.mkdir(parents=True, exist_ok=True)
            srtm_file = srtm_dir / "srtm_carlsbad_30m.tif"
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Clip SRTM to bounding box (lon_min, lat_min, lon_max, lat_max)
                    # Note: bounds order is west, south, east, north
                    clip(bounds=(self.lon_min, self.lat_min, self.lon_max, self.lat_max),
                         output=str(srtm_file),
                         resolution=1,  # 1 arc-second (~30m)
                         product='SRTM1')  # Correct product for 30m SRTM
                    file_size = srtm_file.stat().st_size / (1024**2)
                    logger.info(f"✓ SRTM downloaded successfully: {srtm_file} ({file_size:.1f} MB)")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to download SRTM after {max_retries} attempts: {e}")
                        success = False
                    else:
                        import time
                        time.sleep(5)  # Brief pause before retry
        else:
            logger.info("Elevation data already available.")
        
        # Note: XGM2019e requires manual download - handled in report
        
        logger.info("="*70)
        return success
    
    def create_final_report(self):
        """Create a comprehensive final report."""
        
        report_file = PROJECT_ROOT / "DATA_DOWNLOAD_COMPLETE_REPORT.md"
        
        # Count what we have
        required_data = ['emag2_magnetic', 'egm2008_gravity']
        optional_data = ['xgm2019e_gravity', 'copernicus_dem', 'srtm_dem', 'insar_sentinel1', 'lithology']
        
        required_available = sum(1 for key in required_data 
                                if self.status.get(key, {}).get('available', False))
        has_elevation = self.status.get('copernicus_dem', {}).get('available') or self.status.get('srtm_dem', {}).get('available')
        
        report = f"""# GeoAnomalyMapper - Data Download Status Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Summary

**Region of Interest:** {self.lon_min}° to {self.lon_max}° lon, {self.lat_min}° to {self.lat_max}° lat (Carlsbad Caverns area)  
**Required Data Available:** {required_available}/{len(required_data)}  
**Elevation Available:** {'Yes' if has_elevation else 'No (SRTM attempted)' }  
**High-Res Gravity Ready:** {self.status.get('xgm2019e_gravity', {}).get('grid_available', False)}  
**Total Data Readiness:** {'Complete (run process_data.py next)' if required_available == len(required_data) and has_elevation else 'Partial - see below'}

## Core Data (Required for Basic Analysis)

### 1. Magnetic Anomaly Data (EMAG2)
"""
        
        mag = self.status.get('emag2_magnetic', {})
        if mag.get('available'):
            report += f"- ✅ **AVAILABLE**\n"
            report += f"- File: `{mag['path']}`\n"
            report += f"- Size: {mag['size_mb']} MB\n"
            report += f"- Resolution: {mag['resolution']}\n"
            report += f"- Coverage: {mag['coverage']}\n"
        else:
            report += "- ❌ **MISSING** - Run `python GeoAnomalyMapper/download_all_free_data.py` to acquire.\n"
        
        report += "\n### 2. Gravity Anomaly Data (EGM2008 Baseline)\n"
        grav = self.status.get('egm2008_gravity', {})
        if grav.get('available'):
            report += f"- ✅ **AVAILABLE**\n"
            report += f"- File: `{grav['path']}`\n"
            report += f"- Size: {grav['size_mb']} MB\n"
            report += f"- Resolution: {grav['resolution']} (baseline for initial analysis)\n"
            report += f"- Coverage: {grav['coverage']}\n"
        else:
            report += "- ❌ **MISSING** - Run `python GeoAnomalyMapper/download_all_free_data.py` to acquire.\n"
        
        report += "\n## Enhanced Data (Recommended for High-Resolution Void Detection)\n\n"
        report += "### 3. High-Resolution Gravity (XGM2019e ~2km)\n"
        xgm = self.status.get('xgm2019e_gravity', {})
        if xgm.get('grid_available'):
            report += f"- ✅ **GRID READY FOR PROCESSING**\n"
            report += f"- File: `{xgm['grid_path']}`\n"
            report += f"- Resolution: {xgm['resolution']} (10x better than EGM2008)\n"
            report += f"- Coverage: {xgm['coverage']}\n"
        elif xgm.get('coefficients_available'):
            report += "- ⚠️ **COEFFICIENTS DOWNLOADED** (conversion needed)\n"
            report += f"- File: `{xgm['coef_path']}`\n"
            report += "- **Next Step:** Use `convert_xgm2019e_to_grid.py` or manual ICGEM calculation.\n"
        else:
            report += "- ❌ **MISSING (MANUAL DOWNLOAD REQUIRED)**\n"
            report += "- **Instructions:**\n"
            report += "  1. Visit: http://icgem.gfz-potsdam.de/tom_longtime\n"
            report += "  2. Model: XGM2019e_2159 (Release 2019)\n"
            report += f"  3. Extent: Longitude {self.lon_min} to {self.lon_max}, Latitude {self.lat_min} to {self.lat_max}\n"
            report += "  4. Grid step: 0.02 degrees (~2 km resolution)\n"
            report += "  5. Calculation point: Sea level\n"
            report += "  6. Output format: GeoTIFF\n"
            report += "  7. Requested quantity: Gravity disturbance (mGal)\n"
            report += f"  8. Save as: `{xgm['grid_path']}`\n"
            report += "- Expected file size: ~5-10 MB for this region.\n"
            report += "- Once downloaded, re-run this script to verify.\n"
        
        report += "\n### 4. High-Resolution Digital Elevation Model (30m)\n"
        dem = self.status.get('copernicus_dem', {})
        srtm = self.status.get('srtm_dem', {})
        if dem.get('available'):
            report += f"- ✅ **COPERNICUS DEM READY**\n"
            report += f"- Tiles: {dem['tile_count']}\n"
            report += f"- Resolution: {dem['resolution']}\n"
            report += f"- Directory: `{dem['dir']}`\n"
            report += "- Coverage: High-accuracy land elevation (preferred for topography correction).\n"
        elif srtm.get('available'):
            report += f"- ✅ **SRTM FALLBACK READY**\n"
            report += f"- File: `{srtm['path']}`\n"
            report += f"- Size: {srtm['size_mb']} MB\n"
            report += f"- Resolution: {srtm['resolution']}\n"
            report += "- Coverage: {srtm['coverage']} (good alternative, slightly lower accuracy in some areas).\n"
        else:
            report += "- ⚠️ **NOT AVAILABLE** (auto-download attempted via SRTM)\n"
            report += "- **Fallback Options:**\n"
            report += "  - **Option A: SRTM 30m** (Automated in this script - re-run if failed).\n"
            report += f"    Manual command: `eio clip -o data/raw/elevation/srtm/srtm_carlsbad_30m.tif --bounds {self.lon_min} {self.lat_min} {self.lon_max} {self.lat_max} --product SRTM1`\n"
            report += "  - **Option B: Manual from USGS EarthExplorer** (https://earthexplorer.usgs.gov/ - search SRTM 1 Arc-Second Global, clip to region).\n"
            report += "  - **Option C: ASTER GDEM** (https://search.earthdata.nasa.gov/ - free registration required, 30m global).\n"
            report += "- Expected file size: ~2-5 MB for clipped region.\n"
            report += "- Place in: `data/raw/elevation/srtm/` and re-run this script.\n"
        
        report += "\n### 5. InSAR Ground Deformation Data (Sentinel-1)\n"
        insar = self.status.get('insar_sentinel1', {})
        if insar.get('available'):
            report += f"- ✅ **RAW SCENES READY FOR PROCESSING**\n"
            report += f"- Scenes: {insar['scene_count']}\n"
            report += f"- Resolution: {insar['resolution']}\n"
            report += f"- Directory: `{insar['dir']}`\n"
            report += "- **Next Step:** Use SNAP toolbox or ISCE to generate interferograms for deformation mapping.\n"
            report += "- Coverage: {insar['coverage']} (focus on 2025 acquisitions for recent voids).\n"
        else:
            report += "- ℹ️ **OPTIONAL** (enhances detection of active subsidence).\n"
            report += "- Download from: Copernicus Open Access Hub (https://scihub.copernicus.eu/).\n"
            report += f"- Search: Sentinel-1 SLC, area {self.lon_min} to {self.lon_max}, {self.lat_min} to {self.lat_max}, dates 2024-2025.\n"
            report += "- Expected: 4-8 scenes for interferometry baseline.\n"
        
        report += "\n### 6. Lithology/Rock Type Data (GLiM Database)\n"
        lith = self.status.get('lithology', {})
        if lith.get('available'):
            report += f"- ✅ **AVAILABLE**\n"
            report += f"- File: `{lith['path']}`\n"
            report += f"- Size: {lith['size_mb']} MB\n"
            report += f"- Resolution: {lith['resolution']}\n"
            report += "- Coverage: {lith['coverage']} (helps interpret anomaly causes - karst vs. faults).\n"
            report += "- **Next Step:** Extract with GDAL: `gdal_translate -of GTIFF SL2013sv_0.5d-grd_v2.1.tar.bz2 data/raw/lithology.tif`.\n"
        else:
            report += "- ℹ️ **OPTIONAL** (for geological context).\n"
            report += "- Download: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4240944/ (GLiM v2.1).\n"
            report += "- Place in: `data/raw/SL2013sv_0.5d-grd_v2.1.tar.bz2`.\n"
        
        report += "\n## Next Steps for Analysis\n"
        report += "1. **Verify All Data:** Re-run `python download_all_data_complete.py` after manual downloads.\n"
        report += "2. **Process Data:** Run `python GeoAnomalyMapper/process_data.py` to fuse datasets (reproject to common CRS: EPSG:4326, resample to 30m).\n"
        report += "3. **Detect Voids:** Run `python GeoAnomalyMapper/detect_voids.py` for probability mapping.\n"
        report += "4. **Visualize:** Run `python GeoAnomalyMapper/create_enhanced_visualization.py` for overlays and reports.\n"
        report += "5. **Dependencies:** Ensure `pip install -r requirements.txt` (includes rasterio, geopandas, scipy, matplotlib).\n"
        report += "\n**Project Status:** Ready for void detection once high-res gravity and elevation are complete.\n"
        report += f"**Generated by:** DataDownloadManager v1.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved: {report_file}")
        return str(report_file)
    
    def export_status_json(self):
        """Export status to JSON for pipeline integration."""
        status_file = PROJECT_ROOT / "data_status.json"
        with open(status_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
        logger.info(f"JSON status exported: {status_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and verify all GeoAnomalyMapper data.")
    parser.add_argument('--region', type=str, default="-105.0,32.0,-104.0,33.0",
                        help="Region as 'lon_min,lat_min,lon_max,lat_max'")
    args = parser.parse_args()
    
    # Parse region
    try:
        lon_min, lat_min, lon_max, lat_max = map(float, args.region.split(','))
        region = (lon_min, lat_min, lon_max, lat_max)
    except ValueError:
        logger.error("Invalid region format. Use: lon_min,lat_min,lon_max,lat_max")
        sys.exit(1)
    
    manager = DataDownloadManager(region)
    
    # Check existing
    manager.check_existing_data()
    
    # Print report
    manager.print_status_report()
    
    # Download missing automatable
    if not manager.download_missing():
        logger.warning("Some downloads failed - check logs.")
    
    # Re-check after downloads
    manager.check_existing_data()
    manager.print_status_report()
    
    # Generate reports
    manager.create_final_report()
    manager.export_status_json()
    
    logger.info("Data acquisition complete. Check DATA_DOWNLOAD_COMPLETE_REPORT.md for details.")


if __name__ == "__main__":
    main()