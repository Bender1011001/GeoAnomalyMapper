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
    
    # Execute in parallel using direct subprocess management
    # This avoids pickling issues and provides better control over process lifecycle on Windows
    failed_tiles = []
    active_procs = []
    pending_tiles = tiles.copy()
    
    # Environment for subprocesses
    env = os.environ.copy()
    # NOTE: PyTorch CPU-only version detected. GPU cannot be used until PyTorch with CUDA is installed.
    # See instructions in batch_processing.log or run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    env["FORCE_CPU_INVERSION"] = "0"  # Temporarily force CPU until PyTorch+CUDA is installed
    
    # Timeout configuration
    # CPU mode: Using 30 minutes until GPU-enabled PyTorch is installed
    # After GPU setup: reduce to 900 seconds (15 minutes)
    TILE_TIMEOUT_SECONDS = 900  # 30 minutes per tile (CPU mode)

    # Add diagnostic logging for timing
    logger.info(f"Starting batch processing with {len(tiles)} tiles, timeout: {TILE_TIMEOUT_SECONDS}s per tile")

    while pending_tiles or active_procs:
        # Start new processes if slots available
        while len(active_procs) < args.workers and pending_tiles:
            tile_id, region_str = pending_tiles.pop(0)
            output_name = batch_dir / tile_id
            log_file = batch_dir / f"{tile_id}.log"
            
            cmd = [
                sys.executable, "workflow.py",
                f"--region={region_str}",
                "--resolution", str(args.resolution),
                "--output-name", str(output_name),
                "--skip-visuals",
                "--mode", "mineral"
            ]
            
            logger.info(f"[{tile_id}] Starting process: {region_str}")
            
            try:
                # Open log file for writing
                f = open(log_file, "w")
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )
                active_procs.append({
                    'id': tile_id,
                    'proc': proc,
                    'file': f,
                    'start_time': time.time()
                })
            except Exception as e:
                logger.error(f"[{tile_id}] Failed to start: {e}")
                failed_tiles.append(tile_id)

        # Check active processes
        for p_info in active_procs[:]:
            ret = p_info['proc'].poll()

            # Check for timeout
            elapsed = time.time() - p_info['start_time']
            if ret is None and elapsed > TILE_TIMEOUT_SECONDS:
                logger.error(f"[{p_info['id']}] TIMEOUT after {elapsed:.1f}s. Killing process.")
                p_info['proc'].kill()
                # Wait for kill to take effect
                try:
                    p_info['proc'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p_info['proc'].terminate() # Force terminate if kill fails

                ret = -999 # Custom code for timeout

            if ret is not None:
                # Process finished
                duration = time.time() - p_info['start_time']
                p_info['file'].close()

                if ret == 0:
                    logger.info(f"[{p_info['id']}] Completed in {duration:.1f}s")
                else:
                    if ret == -999:
                        logger.error(f"[{p_info['id']}] FAILED (Timeout)")
                    else:
                        logger.error(f"[{p_info['id']}] FAILED with return code {ret}")
                    logger.error(f"[{p_info['id']}] Check log for details: {batch_dir / f'{p_info['id']}.log'}")
                    failed_tiles.append(p_info['id'])

                active_procs.remove(p_info)

        # Periodic status update
        if active_procs and int(time.time()) % 30 == 0:  # Every 30 seconds
            status_lines = []
            for p_info in active_procs:
                elapsed = time.time() - p_info['start_time']
                status_lines.append(f"[{p_info['id']}] running for {elapsed:.1f}s")
            logger.info(f"Active processes ({len(active_procs)}): {'; '.join(status_lines)}")

        # Avoid busy loop
        if active_procs:
            time.sleep(0.5)

    if failed_tiles:
        logger.error(f"Batch processing completed with {len(failed_tiles)} failures: {failed_tiles}")
        sys.exit(1)
    else:
        logger.info("All tiles processed successfully.")
        print(f"\nProcessing complete. Tiles saved to: {batch_dir}")
        print("Run 'python merge_results.py' to combine them.")

if __name__ == "__main__":
    main()