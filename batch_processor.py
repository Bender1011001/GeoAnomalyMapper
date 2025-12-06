#!/usr/bin/env python3
"""
Batch Processor for GeoAnomalyMapper
====================================

Splits a large region into smaller grid tiles and processes them in parallel
using separate subprocesses. This ensures maximum resolution can be used
without exhausting system RAM.

Usage:
    python batch_processor.py --region "-125,25,-66,49" --tile-size 1.0
"""
import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# Import project paths to ensure data directories exist
from project_paths import OUTPUTS_DIR, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_tiles(
    region: Tuple[float, float, float, float],
    tile_size: float,
    overlap: float
) -> List[Tuple[str, str]]:
    """
    Generate a list of tile bounding boxes covering the region.
    
    Returns:
        List of (tile_name, region_string) tuples.
    """
    lon_min, lat_min, lon_max, lat_max = region
    
    tiles = []
    
    # Iterate through latitudes and longitudes
    current_lat = lat_min
    row_idx = 0
    
    while current_lat < lat_max:
        current_lon = lon_min
        col_idx = 0
        
        # Calculate tile top latitude
        next_lat = min(current_lat + tile_size, lat_max)
        
        while current_lon < lon_max:
            # Calculate tile right longitude
            next_lon = min(current_lon + tile_size, lon_max)
            
            # Apply buffer/overlap (clamp to global max/min if needed, but here simply expanding)
            # We expand the request box, but keep the ID based on the grid
            t_min_lat = max(-90, current_lat - overlap)
            t_max_lat = min(90, next_lat + overlap)
            t_min_lon = max(-180, current_lon - overlap)
            t_max_lon = min(180, next_lon + overlap)
            
            # Create unique ID and region string
            tile_id = f"tile_r{row_idx:03d}_c{col_idx:03d}"
            region_str = f"{t_min_lon:.6f},{t_min_lat:.6f},{t_max_lon:.6f},{t_max_lat:.6f}"
            
            tiles.append((tile_id, region_str))
            
            current_lon += tile_size
            col_idx += 1
            
        current_lat += tile_size
        row_idx += 1
        
    return tiles

def process_tile(
    tile_id: str,
    region_str: str,
    resolution: float,
    output_dir: Path
) -> bool:
    """
    Run workflow.py for a single tile in a subprocess.
    """
    output_name = output_dir / tile_id
    
    cmd = [
        sys.executable, "workflow.py",
        f"--region={region_str}",  # <--- FIXED: Connects value to flag
        "--resolution", str(resolution),
        "--output-name", str(output_name),
        "--skip-visuals",  # Save time/space, generate visuals only for merged result
        # "--validate"     # Skip validation per tile to speed up processing
    ]
    
    logger.info(f"[{tile_id}] Starting process: {region_str}")
    start_time = time.time()
    
    try:
        # Run workflow.py as a separate process to guarantee memory release
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        logger.info(f"[{tile_id}] Completed in {duration:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[{tile_id}] FAILED with return code {e.returncode}")
        logger.error(f"[{tile_id}] STDERR:\n{e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process large regions by tiling.")
    parser.add_argument("--region", required=True, help="Region 'lon_min,lat_min,lon_max,lat_max'")
    parser.add_argument("--tile-size", type=float, default=2.0, help="Tile size in degrees (default: 2.0)")
    parser.add_argument("--overlap", type=float, default=0.05, help="Overlap in degrees (default: 0.05)")
    parser.add_argument("--resolution", type=float, default=0.001, help="Resolution in degrees (default: 0.001)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel processes (RAM dependent)")
    
    args = parser.parse_args()
    
    # Parse region
    try:
        region = tuple(map(float, args.region.split(',')))
        if len(region) != 4: raise ValueError
    except:
        logger.error("Invalid region format. Use 'min_lon,min_lat,max_lon,max_lat'")
        sys.exit(1)

    # Setup output directory
    batch_dir = OUTPUTS_DIR / "batch_run"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate work items
    tiles = generate_tiles(region, args.tile_size, args.overlap)
    logger.info(f"Generated {len(tiles)} tiles for region {args.region}")
    logger.info(f"Output directory: {batch_dir}")
    
    # Execute in parallel
    failed_tiles = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_tile = {
            executor.submit(process_tile, t_id, reg, args.resolution, batch_dir): t_id 
            for t_id, reg in tiles
        }
        
        for future in as_completed(future_to_tile):
            tile_id = future_to_tile[future]
            try:
                success = future.result()
                if not success:
                    failed_tiles.append(tile_id)
            except Exception as e:
                logger.error(f"[{tile_id}] Exception: {e}")
                failed_tiles.append(tile_id)

    if failed_tiles:
        logger.error(f"Batch processing completed with {len(failed_tiles)} failures: {failed_tiles}")
        sys.exit(1)
    else:
        logger.info("All tiles processed successfully.")
        print(f"\nProcessing complete. Tiles saved to: {batch_dir}")
        print("Run 'python merge_results.py' to combine them.")

if __name__ == "__main__":
    main()