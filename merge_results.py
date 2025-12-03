#!/usr/bin/env python3
"""
Efficient GeoTIFF Merger
========================

Merges tiled GeoTIFFs using GDAL Virtual Rasters (VRT) to avoid high memory usage.
Requires: gdal (installed via conda) or rasterio.

Usage:
    python merge_results.py --input-dir data/outputs/batch_run --output final_merged.tif
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import rasterio
from rasterio.merge import merge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_using_rasterio(input_files: list[Path], output_path: Path):
    """
    Merge files using rasterio.merge (Pure Python).
    Good for smaller datasets, but can be memory intensive for massive grids.
    """
    logger.info(f"Merging {len(input_files)} files using Rasterio...")
    
    src_files_to_mosaic = []
    try:
        for fp in input_files:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        
        # Merge - this loads the result into memory!
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "deflate",
            "tiled": True,
            "bigtiff": "YES"  # Important for >4GB files
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            
        logger.info(f"Successfully saved to {output_path}")
        
    finally:
        for src in src_files_to_mosaic:
            src.close()

def merge_using_gdal(input_files: list[Path], output_path: Path):
    """
    Merge using gdal_merge.py or gdalbuildvrt (System Call).
    Extremely memory efficient as it processes line-by-line.
    """
    logger.info(f"Merging {len(input_files)} files using GDAL (VRT method)...")
    
    vrt_path = output_path.with_suffix(".vrt")
    
    # Step 1: Build Virtual Raster (indexes input files, doesn't copy data)
    # Equivalent to: gdalbuildvrt merged.vrt tile1.tif tile2.tif ...
    cmd_vrt = ["gdalbuildvrt", str(vrt_path)] + [str(p) for p in input_files]
    
    try:
        subprocess.run(cmd_vrt, check=True)
        logger.info(f"Created virtual raster: {vrt_path}")
        
        # Step 2: Translate VRT to GeoTIFF (performs the heavy lifting)
        # Equivalent to: gdal_translate merged.vrt output.tif -co COMPRESS=DEFLATE
        cmd_translate = [
            "gdal_translate", 
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=YES",
            str(vrt_path), 
            str(output_path)
        ]
        
        logger.info("Translating VRT to final GeoTIFF (this may take time)...")
        subprocess.run(cmd_translate, check=True)
        logger.info(f"Successfully created: {output_path}")
        
        # Optional: Clean up VRT
        # vrt_path.unlink() 
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("GDAL tools not found in PATH. Falling back to Rasterio (high RAM usage).")
        merge_using_rasterio(input_files, output_path)

def main():
    parser = argparse.ArgumentParser(description="Merge tiled GeoTIFFs")
    parser.add_argument("--input-dir", required=True, help="Directory containing tile .tif files")
    parser.add_argument("--output", required=True, help="Path for merged output")
    parser.add_argument("--pattern", default="*_void_probability.tif", help="File pattern to match (e.g. *_void_probability.tif)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob(args.pattern))
    
    if not input_files:
        logger.error(f"No files found in {input_dir} matching {args.pattern}")
        sys.exit(1)
        
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prefer GDAL for memory efficiency
    merge_using_gdal(input_files, output_path)

if __name__ == "__main__":
    main()