#!/usr/bin/env python3
"""
Global Anomaly Fusion Pipeline.

Processes global magnetic and gravity datasets into tiled, normalized, and fused anomaly maps.
Outputs Cloud-Optimized GeoTIFFs per tile and a final PMTiles file for web visualization.

Usage: python process_global_map.py
"""

import logging
import subprocess
import sys
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm

# Add the gam module to the python path
# This is a simple way to make the gam package importable without a full install.
# In a real-world scenario, you'd install the package properly.
try:
    from gam.core.tiles import tiles_10x10_ids, tile_bounds_10x10
    from gam.modeling.fuse_simple import robust_z, fuse_layers
    from gam.preprocessing.cog_writer import warp_crop_to_tile, write_cog
except ImportError as e:
    print("Could not import GAM modules. Make sure the 'gam' directory is in the same directory as this script.")
    sys.exit(1)


# --- CONFIGURATION ---
# All paths and parameters are explicitly defined here. No hidden configuration.

# 1. Base directory of the project
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

# 2. Input data paths
# IMPORTANT: You must download these files manually and place them here.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MAG_PATH = RAW_DATA_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
GRAV_PATH = RAW_DATA_DIR / "gravity" / "gravity_disturbance_EGM2008_50491becf3ffdee5c9908e47ed57881ed23de559539cd89e49b4d76635e07266.tiff"

# 3. Output directories
OUTPUT_DIR = DATA_DIR / "outputs"
COG_DIR = OUTPUT_DIR / "cog"
FINAL_PRODUCT_DIR = OUTPUT_DIR / "final"
LOG_FILE = OUTPUT_DIR / "processing.log"

# 4. Tool paths
PMTILES_EXE = Path(r'C:/Users/admin/Downloads/go-pmtiles_1.28.1_Windows_x86_64/pmtiles.exe')

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- PROCESSING FUNCTIONS ---

def _verify_system_dependencies():
    """Checks if required command-line tools (GDAL, pmtiles) are installed."""
    logging.info("Verifying system dependencies (gdalbuildvrt, pmtiles)...")
    if not shutil.which("gdalbuildvrt"):
        logging.error("FATAL: gdalbuildvrt command not found. Is GDAL installed and in your PATH?")
        sys.exit(1)
    if not PMTILES_EXE.exists():
        logging.error(f"FATAL: pmtiles.exe not found at {PMTILES_EXE}")
        sys.exit(1)
    logging.info("System dependencies verified.")


def _process_tile(tile_id: str):
    """
    Processes a single 10x10 degree tile: warp, normalize, fuse, and write COGs.
    """
    try:
        # 1. Get tile boundaries
        bounds = tile_bounds_10x10(tile_id)

        # 2. Warp source rasters to the tile grid (0.1 degree resolution -> 100x100 pixels)
        mag_tile_data = warp_crop_to_tile(str(MAG_PATH), bounds)
        grav_tile_data = warp_crop_to_tile(str(GRAV_PATH), bounds)

        # Check if the tile has valid data
        if np.all(np.isnan(mag_tile_data)) and np.all(np.isnan(grav_tile_data)):
            logging.warning(f"Tile {tile_id}: Both source rasters are empty. Skipping.")
            return

        # 3. Normalize each layer using a robust z-score
        norm_mag = robust_z(mag_tile_data)
        norm_grav = robust_z(grav_tile_data)

        # 4. Fuse the normalized layers by taking the mean
        fused_anomaly = fuse_layers(norm_mag, norm_grav)

        # 5. Write the processed tiles as COGs
        write_cog(str(COG_DIR / "mag" / f"{tile_id}.tif"), norm_mag, bounds)
        write_cog(str(COG_DIR / "grav" / f"{tile_id}.tif"), norm_grav, bounds)
        write_cog(str(COG_DIR / "fused" / f"{tile_id}.tif"), fused_anomaly, bounds)

    except Exception as e:
        logging.error(f"Tile {tile_id}: Processing failed. Error: {e}", exc_info=True)


def _generate_vrt():
    """Generates a VRT mosaic from all the fused COG tiles."""
    logging.info("Generating VRT from fused COG tiles...")
    vrt_path = FINAL_PRODUCT_DIR / "fused_anomaly.vrt"
    cog_files_path = COG_DIR / "fused" / "*.tif"

    cmd = [
        "gdalbuildvrt",
        str(vrt_path),
        str(cog_files_path)
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"VRT successfully created at {vrt_path}")


def _generate_geotiff():
    """Converts the VRT mosaic into a single GeoTIFF for PMTiles conversion."""
    logging.info("Converting VRT to GeoTIFF...")
    vrt_path = FINAL_PRODUCT_DIR / "fused_anomaly.vrt"
    geotiff_path = FINAL_PRODUCT_DIR / "fused_anomaly.tif"
    
    # Use gdal_translate to convert VRT to GeoTIFF with compression
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        str(vrt_path),
        str(geotiff_path)
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"GeoTIFF successfully created at {geotiff_path}")


def _generate_mbtiles():
    """Converts the GeoTIFF into MBTiles using gdal2tiles."""
    logging.info("Generating MBTiles from GeoTIFF using gdal2tiles...")
    geotiff_path = FINAL_PRODUCT_DIR / "fused_anomaly.tif"
    mbtiles_path = FINAL_PRODUCT_DIR / "fused_anomaly.mbtiles"
    
    # Use gdal2tiles.py to create MBTiles
    # -z sets the zoom levels (0-10 for global coverage)
    cmd = [
        "python",
        "-m", "gdal2tiles",
        "-z", "0-8",
        "--processes=4",
        str(geotiff_path),
        str(mbtiles_path).replace('.mbtiles', '')  # gdal2tiles adds the extension
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"MBTiles successfully created at {mbtiles_path}")
        return True
    except Exception as e:
        logging.warning(f"Failed to create MBTiles: {e}")
        logging.info("Skipping MBTiles generation. GeoTIFF is available for use.")
        return False


def _generate_pmtiles():
    """Converts MBTiles to PMTiles for web visualization."""
    mbtiles_path = FINAL_PRODUCT_DIR / "fused_anomaly.mbtiles"
    
    # Check if MBTiles exists
    if not mbtiles_path.exists():
        logging.info("Skipping PMTiles generation (MBTiles not available)")
        logging.info(f"Final outputs available:")
        logging.info(f"  - VRT: {FINAL_PRODUCT_DIR / 'fused_anomaly.vrt'}")
        logging.info(f"  - GeoTIFF: {FINAL_PRODUCT_DIR / 'fused_anomaly.tif'}")
        logging.info(f"  - COG tiles: {COG_DIR / 'fused'}")
        return
    
    logging.info("Converting MBTiles to PMTiles...")
    pmtiles_path = FINAL_PRODUCT_DIR / "fused_anomaly.pmtiles"

    cmd = [
        str(PMTILES_EXE),
        "convert",
        str(mbtiles_path),
        str(pmtiles_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"PMTiles successfully created at {pmtiles_path}")
    except Exception as e:
        logging.warning(f"Failed to convert to PMTiles: {e}")
        logging.info("MBTiles file is available for use")


def main():
    """Main script to run the entire global anomaly processing pipeline."""
    logging.info("--- Starting Global Anomaly Map Processing ---")

    # 1. Verify that GDAL and pmtiles are installed before we start
    _verify_system_dependencies()

    # 2. Ensure all required directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    COG_DIR.joinpath("mag").mkdir(parents=True, exist_ok=True)
    COG_DIR.joinpath("grav").mkdir(parents=True, exist_ok=True)
    COG_DIR.joinpath("fused").mkdir(parents=True, exist_ok=True)
    FINAL_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Check for source data
    if not MAG_PATH.exists() or not GRAV_PATH.exists():
        logging.error("FATAL: Source data not found!")
        logging.error(f"Please download the required GeoTIFFs and place them in: {RAW_DATA_DIR}")
        logging.error(f"Required: {MAG_PATH.name}, {GRAV_PATH.name}")
        sys.exit(1)

    # 4. Process all tiles
    tile_ids = tiles_10x10_ids()
    logging.info(f"Found {len(tile_ids)} tiles to process.")

    with tqdm(total=len(tile_ids), desc="Processing Tiles") as pbar:
        for tile_id in tile_ids:
            # Make the process resumable by checking if the final output for this tile exists
            final_cog_path = COG_DIR / "fused" / f"{tile_id}.tif"
            if final_cog_path.exists():
                pbar.update(1)
                continue

            _process_tile(tile_id)
            pbar.update(1)

    # 5. Post-processing: Generate final data products
    _generate_vrt()
    _generate_geotiff()
    _generate_mbtiles()
    _generate_pmtiles()

    logging.info("--- Global Anomaly Map Processing Finished Successfully ---")


if __name__ == "__main__":
    main()