import os
import json
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent.absolute()))

try:
    from pinn_gravity_inversion import invert_gravity
    from fetch_lithology_density import fetch_and_rasterize
except ImportError as e:
    print(f"Critical Error: Could not import project modules. {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("batch_workflow.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

STATE_FILE = "processed_tiles.json"

def get_tiles(width: int, height: int, tile_size: int, overlap: int) -> List[Dict]:
    """
    Generate tile metadata. 
    """
    tiles = []
    step = tile_size - overlap
    if step <= 0: raise ValueError("Overlap must be < tile_size")

    for row in range(0, height, step):
        for col in range(0, width, step):
            tiles.append({
                "id": f"tile_{col}_{row}",
                "col_off": col,
                "row_off": row,
                # These are the requested full dimensions (padding handled later)
                "req_width": tile_size,
                "req_height": tile_size
            })
    return tiles

def save_padded_tile(src_dataset, dest_path: str, col: int, row: int, size: int):
    """
    Reads a window from the OPEN dataset. 
    If the window goes out of bounds, it PADS the data (reflect mode).
    """
    # Calculate the actual valid window within the image bounds
    w_width = min(size, src_dataset.width - col)
    w_height = min(size, src_dataset.height - row)
    window = Window(col, row, w_width, w_height)
    
    # Read data
    data = src_dataset.read(1, window=window)
    
    # Calculate required padding to reach model input size
    pad_h = size - w_height
    pad_w = size - w_width
    
    if pad_h > 0 or pad_w > 0:
        # Pad with reflection to minimize edge effects in the PINN
        data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Determine transform for this specific window
    window_transform = src_dataset.window_transform(window)
    
    profile = src_dataset.profile.copy()
    profile.update({
        "height": size,
        "width": size,
        "transform": window_transform,
        "count": 1, 
        "driver": "GTiff"
    })
    
    with rasterio.open(dest_path, "w", **profile) as dst:
        dst.write(data, 1)

def crop_and_save_output(source_path: Path, final_path: Path, overlap: int, 
                         col_off: int, row_off: int, 
                         full_w: int, full_h: int, 
                         tile_size: int):
    """
    Crops the PINN output to remove overlap AND padding.
    Ensures seamless stitching by only keeping the 'valid center' and real data.
    """
    with rasterio.open(source_path) as src:
        data = src.read(1)
        src_transform = src.transform 

    # 1. Determine Valid Data Region (Unpadded Dimensions)
    # How much of this tile was actual data from the source map?
    valid_read_w = min(tile_size, full_w - col_off)
    valid_read_h = min(tile_size, full_h - row_off)

    # 2. Determine Crop Indices (Relative to tile pixels)
    margin = overlap // 2

    # Left & Top: If at the very start, keep the edge. Otherwise, cut overlap.
    start_x = 0 if col_off == 0 else margin
    start_y = 0 if row_off == 0 else margin

    # Right:
    # If this tile extends past the image boundary (was padded), we clip strictly to valid data.
    # Otherwise, we clip the overlap margin.
    if col_off + tile_size >= full_w:
        end_x = valid_read_w  # Strip padding
    else:
        end_x = tile_size - margin # Strip overlap

    # Bottom:
    if row_off + tile_size >= full_h:
        end_y = valid_read_h
    else:
        end_y = tile_size - margin

    # 3. Perform Crop
    cropped_data = data[start_y:end_y, start_x:end_x]

    # 4. Update Transform (CRS Drift Fix)
    # Shift the origin by the crop amount (start_x, start_y)
    new_transform = src_transform * src_transform.translation(start_x, start_y)

    # 5. Save
    with rasterio.open(final_path, 'w', driver='GTiff', 
                       height=cropped_data.shape[0], width=cropped_data.shape[1],
                       count=1, dtype=cropped_data.dtype,
                       crs=src.crs, transform=new_transform, compress='deflate') as dst:
        dst.write(cropped_data, 1)

def process_tile(tile: Dict, src_dataset, output_dir: Path, temp_dir: Path, 
                 tile_size: int, overlap: int, full_shape: Tuple[int, int]) -> bool:
    tile_id = tile["id"]
    temp_grav = temp_dir / f"{tile_id}_grav.tif"
    temp_lith = temp_dir / f"{tile_id}_lith.tif"
    temp_density = temp_dir / f"{tile_id}_density_raw.tif"
    final_tile = output_dir / f"{tile_id}_density.tif"
    
    try:
        # 1. Prepare Input (Pass the open dataset handle)
        save_padded_tile(src_dataset, str(temp_grav), 
                         tile["col_off"], tile["row_off"], tile_size)
        
        # 2. Fetch Lithology (Optional)
        has_lithology = False
        try:
            if fetch_and_rasterize(str(temp_grav), str(temp_lith)):
                has_lithology = True
        except Exception:
            pass # proceed without lithology

        # 3. Invert (PINN) - Output is tile_size x tile_size
        lith_arg = str(temp_lith) if has_lithology else None
        invert_gravity(str(temp_grav), str(temp_density), lith_arg)
        
        # 4. Crop & Clean (Fixing Edge + Padding Artifacts)
        crop_and_save_output(temp_density, final_tile, overlap, 
                             tile["col_off"], tile["row_off"],
                             full_shape[0], full_shape[1], tile_size)
        
        return True

    except Exception as e:
        logger.error(f"[{tile_id}] Failed: {e}")
        return False
    finally:
        # Aggressive Cleanup
        for p in [temp_grav, temp_lith, temp_density]:
            if p.exists(): p.unlink()

def merge_tiles(output_dir: Path, final_output: str):
    logger.info("Merging tiles...")
    files = list(output_dir.glob("*_density.tif"))
    if not files: return

    vrt_path = output_dir / "mosaic.vrt"
    list_file = output_dir / "files.txt"
    
    # Check for GDAL
    if not shutil.which("gdalbuildvrt"):
        logger.error("CRITICAL: 'gdalbuildvrt' not found. Merge aborted.")
        logger.error("Please install GDAL core tools or run in a geospatial container.")
        return

    with open(list_file, "w") as f:
        for p in files: f.write(str(p.absolute()) + "\n")
        
    try:
        logger.info("Building VRT...")
        subprocess.run(["gdalbuildvrt", "-input_file_list", str(list_file), str(vrt_path)], 
                       check=True, stdout=subprocess.DEVNULL)
        
        logger.info("Translating to Final GeoTIFF...")
        result = subprocess.run([
            "gdal_translate", "-of", "GTiff", 
            "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-co", "TILED=YES",
            str(vrt_path), final_output
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"GDAL Translate Error: {result.stderr}")
        else:
            logger.info("Merge Complete.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Merge process failed: {e}")
    finally:
        if vrt_path.exists(): vrt_path.unlink()
        if list_file.exists(): list_file.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_dir", default="output_tiles", type=Path)
    parser.add_argument("--final_output", default="final_density_map.tif", type=str)
    parser.add_argument("--tile_size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit("Input file not found")

    temp_dir = args.output_dir / "temp"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Open Source Once (File I/O Optimization)
    try:
        with rasterio.open(args.input) as src_dataset:
            width, height = src_dataset.width, src_dataset.height
            logger.info(f"Input: {width}x{height}, CRS: {src_dataset.crs}")

            tiles = get_tiles(width, height, args.tile_size, args.overlap)
            
            # Resume Logic
            state_path = args.output_dir / STATE_FILE
            processed = {}
            if args.resume and state_path.exists():
                try: processed = json.loads(state_path.read_text())
                except: pass
            
            tiles_to_do = [t for t in tiles if processed.get(t["id"]) != "success"]
            
            pbar = tqdm(tiles_to_do, desc="Processing Tiles")
            for tile in pbar:
                # Pass src_dataset to process_tile instead of path
                success = process_tile(
                    tile, src_dataset, args.output_dir, temp_dir, 
                    args.tile_size, args.overlap, (width, height)
                )
                processed[tile["id"]] = "success" if success else "failed"
                
                with open(state_path, "w") as f:
                    json.dump(processed, f, indent=2)
                    
    except Exception as e:
        logger.error(f"Failed to open source dataset: {e}")
        sys.exit(1)

    merge_tiles(args.output_dir, args.final_output)
    if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()