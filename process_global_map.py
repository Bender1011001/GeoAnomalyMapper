#!/usr/bin/env python3
"""
Global Anomaly Fusion Pipeline - Standalone Version

Processes global magnetic and gravity datasets into tiled, normalized, and fused anomaly maps.
All utilities are embedded - no external package dependencies beyond standard libraries.

Usage: python process_global_map.py
"""

import logging
import subprocess
import sys
from pathlib import Path
import shutil
import re
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from affine import Affine

# ============================================================================
# EMBEDDED UTILITIES (from gam.core.tiles)
# ============================================================================

TILE_SIZE_DEG: float = 10.0
RESOLUTION_0P1: float = 0.1
PIXELS_PER_TILE: int = int(round(TILE_SIZE_DEG / RESOLUTION_0P1))  # 100
WORLD_BOUNDS: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
_TILE_ID_RE = re.compile(r"^t_([NS])(\d{2})_([EW])(\d{3})$")


def _lat_label_from_min(min_lat: float) -> str:
    """Format latitude band label from band minimum latitude in degrees."""
    if min_lat % 10 != 0:
        raise ValueError("Latitude band minimum must be multiple of 10 degrees")
    if min_lat < 0:
        return f"S{int(abs(min_lat)):02d}"
    else:
        return f"N{int(min_lat):02d}"


def _lon_label_from_min(min_lon: float) -> str:
    """Format longitude band label from band minimum longitude in degrees."""
    if min_lon % 10 != 0:
        raise ValueError("Longitude band minimum must be multiple of 10 degrees")
    if min_lon < 0:
        return f"W{int(abs(min_lon)):03d}"
    else:
        return f"E{int(min_lon):03d}"


def _lat_bands_10deg() -> List[Tuple[str, float, float]]:
    """Return list of (label, miny, maxy) for 10° latitude bands covering [-90,90)."""
    bands: List[Tuple[str, float, float]] = []
    for miny in range(-90, 90, 10):
        maxy = miny + 10
        if maxy > 90:
            break
        label = _lat_label_from_min(float(miny))
        bands.append((label, float(miny), float(maxy)))
    return bands


def _lon_bands_10deg() -> List[Tuple[str, float, float]]:
    """Return list of (label, minx, maxx) for 10° longitude bands covering [-180,180)."""
    bands: List[Tuple[str, float, float]] = []
    for minx in range(-180, 180, 10):
        maxx = minx + 10
        if maxx > 180:
            break
        label = _lon_label_from_min(float(minx))
        bands.append((label, float(minx), float(maxx)))
    return bands


def tiles_10x10_ids() -> List[str]:
    """Enumerate all 10°×10° tile IDs in deterministic order."""
    ids: List[str] = []
    for lat_label, _, _ in _lat_bands_10deg():
        for lon_label, _, _ in _lon_bands_10deg():
            ids.append(f"t_{lat_label}_{lon_label}")
    return ids


def _parse_tile_id(tile_id: str) -> Tuple[str, int, str, int]:
    """Validate and parse a tile id into components."""
    m = _TILE_ID_RE.match(tile_id)
    if not m:
        raise ValueError("Malformed tile_id. Expected pattern 't_[NS]\\d{2}_[EW]\\d{3}'")
    lat_hemi, lat_deg_s, lon_hemi, lon_deg_s = m.groups()
    lat_deg = int(lat_deg_s)
    lon_deg = int(lon_deg_s)
    
    if lat_deg % 10 != 0 or lon_deg % 10 != 0:
        raise ValueError("Degrees in tile_id must be multiples of 10")
    if lat_deg > 90 or lon_deg > 180:
        raise ValueError("Latitude degrees must be ≤ 90, longitude degrees must be ≤ 180")
    if lat_hemi == "N" and lat_deg > 80:
        raise ValueError("N latitude band must be ≤ N80 for 10° tiles")
    if lat_hemi == "S" and lat_deg > 90:
        raise ValueError("S latitude band must be ≤ S90 for 10° tiles")
    if lon_hemi == "E" and lon_deg > 170:
        raise ValueError("E longitude band must be ≤ E170 for 10° tiles")
    if lon_hemi == "W" and lon_deg > 180:
        raise ValueError("W longitude band must be ≤ W180 for 10° tiles")
    
    return lat_hemi, lat_deg, lon_hemi, lon_deg


def tile_bounds_10x10(tile_id: str) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) for a 10°×10° tile id."""
    lat_hemi, lat_deg, lon_hemi, lon_deg = _parse_tile_id(tile_id)
    
    miny = float(lat_deg if lat_hemi == "N" else -lat_deg)
    maxy = miny + TILE_SIZE_DEG
    
    minx = float(lon_deg if lon_hemi == "E" else -lon_deg)
    maxx = minx + TILE_SIZE_DEG
    
    wminx, wminy, wmaxx, wmaxy = WORLD_BOUNDS
    if not (wminx <= minx < maxx <= wmaxx) or not (wminy <= miny < maxy <= wmaxy):
        raise ValueError(f"Tile {tile_id} bounds outside world extents")
    
    return (minx, miny, maxx, maxy)


# ============================================================================
# EMBEDDED UTILITIES (from gam.modeling.fuse_simple)
# ============================================================================

def robust_z(x: np.ndarray, clamp: float = 6.0) -> np.ndarray:
    """Compute robust z-score using median and MAD."""
    x = np.asarray(x, dtype="float32")
    med = np.nanmedian(x)
    if np.isnan(med):
        return np.full_like(x, np.nan, dtype="float32")
    
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        z = np.zeros_like(x, dtype="float32")
        z[np.isnan(x)] = np.nan
        return z
    
    scale = np.float32(1.4826) * np.float32(mad)
    z = (x - np.float32(med)) / scale
    z = np.clip(z, -np.float32(clamp), np.float32(clamp), out=z)
    z[np.isnan(x)] = np.nan
    return z.astype("float32", copy=False)


def fuse_layers(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Fuse two z-score layers by mean ignoring NaNs."""
    z1 = np.asarray(z1, dtype="float32")
    z2 = np.asarray(z2, dtype="float32")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape")
    
    stack = np.stack([z1, z2], axis=0)
    fused = np.nanmean(stack, axis=0)
    return fused.astype("float32", copy=False)


# ============================================================================
# EMBEDDED UTILITIES (from gam.preprocessing.cog_writer)
# ============================================================================

def _resampling_from_str(name: str) -> Resampling:
    name_l = (name or "bilinear").strip().lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
    }
    return mapping.get(name_l, Resampling.bilinear)


def warp_crop_to_tile(
    input_path: str,
    tile_bounds: Tuple[float, float, float, float],
    dst_res: float = 0.1,
    dst_crs: str = "EPSG:4326",
    resampling: str = "bilinear",
) -> np.ndarray:
    """Warp + crop source raster to a fixed 10°×10° tile grid."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input raster not found: {in_path}")
    
    minx, miny, maxx, maxy = tile_bounds
    rows = int(round((maxy - miny) / dst_res))
    cols = int(round((maxx - minx) / dst_res))
    if rows != PIXELS_PER_TILE or cols != PIXELS_PER_TILE:
        raise ValueError(f"Tile size mismatch: expected {PIXELS_PER_TILE}x{PIXELS_PER_TILE}")
    
    dst_transform: Affine = from_origin(minx, maxy, dst_res, dst_res)
    rs = _resampling_from_str(resampling)
    
    with rasterio.open(in_path) as src:
        vrt_opts = {
            "crs": dst_crs,
            "transform": dst_transform,
            "height": rows,
            "width": cols,
            "resampling": rs,
            "src_nodata": src.nodata,
            "dst_nodata": np.float32(np.nan),
        }
        with WarpedVRT(src, **vrt_opts) as vrt:
            data = vrt.read(1, out_dtype="float32", masked=True, resampling=rs)
    
    arr = np.asarray(data.filled(np.nan), dtype="float32")
    return arr


def write_cog(
    output_path: str,
    array: np.ndarray,
    tile_bounds: Tuple[float, float, float, float],
    dst_res: float = 0.1,
) -> None:
    """Write a float32 NaN-nodata array as a COG."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if array.ndim != 2:
        raise ValueError("Array must be 2D (rows, cols)")
    if array.shape[0] != PIXELS_PER_TILE or array.shape[1] != PIXELS_PER_TILE:
        raise ValueError(f"Array shape must be {PIXELS_PER_TILE}x{PIXELS_PER_TILE}")
    
    minx, miny, maxx, maxy = tile_bounds
    transform = from_origin(minx, maxy, dst_res, dst_res)
    height, width = array.shape
    
    # Try rio-cogeo if available
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        from rasterio.io import MemoryFile
        
        profile = {
            **cog_profiles.get("deflate"),
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "DEFLATE",
            "predictor": 2,
            "zlevel": 9,
            "bigtiff": "IF_SAFER",
        }
        
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=np.float32(np.nan),
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="DEFLATE",
                predictor=2,
                zlevel=9,
            ) as mem_ds:
                mem_ds.write(array, 1)
            
            cog_translate(
                memfile,
                str(out_path),
                profile,
                nodata=np.float32(np.nan),
                overview_levels=[2, 4, 8, 16],
                overview_resampling="average",
                web_optimized=False,
                add_mask=False,
                quiet=True,
            )
        return
    except Exception:
        pass
    
    # Fallback to plain GTiff
    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.float32(np.nan),
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        predictor=2,
        zlevel=9,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="ALL_CPUS",
    ) as dst:
        dst.write(array, 1)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        dst.update_tags(OVERVIEW_RESAMPLING="AVERAGE")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MAG_PATH = RAW_DATA_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
GRAV_PATH = RAW_DATA_DIR / "gravity" / "gravity_disturbance_EGM2008_50491becf3ffdee5c9908e47ed57881ed23de559539cd89e49b4d76635e07266.tiff"

OUTPUT_DIR = DATA_DIR / "outputs"
COG_DIR = OUTPUT_DIR / "cog"
FINAL_PRODUCT_DIR = OUTPUT_DIR / "final"
LOG_FILE = OUTPUT_DIR / "processing.log"

PMTILES_EXE = Path(r'C:/Users/admin/Downloads/go-pmtiles_1.28.1_Windows_x86_64/pmtiles.exe')

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def _verify_system_dependencies():
    """Checks if required command-line tools are installed."""
    logging.info("Verifying system dependencies...")
    if not shutil.which("gdalbuildvrt"):
        logging.error("FATAL: gdalbuildvrt not found. Is GDAL installed?")
        sys.exit(1)
    if not PMTILES_EXE.exists():
        logging.warning(f"pmtiles.exe not found at {PMTILES_EXE} - PMTiles generation will be skipped")
    logging.info("System dependencies verified.")


def _process_tile(tile_id: str):
    """Process a single 10x10 degree tile."""
    try:
        bounds = tile_bounds_10x10(tile_id)
        
        mag_tile_data = warp_crop_to_tile(str(MAG_PATH), bounds)
        grav_tile_data = warp_crop_to_tile(str(GRAV_PATH), bounds)
        
        if np.all(np.isnan(mag_tile_data)) and np.all(np.isnan(grav_tile_data)):
            logging.warning(f"Tile {tile_id}: Both source rasters empty. Skipping.")
            return
        
        norm_mag = robust_z(mag_tile_data)
        norm_grav = robust_z(grav_tile_data)
        fused_anomaly = fuse_layers(norm_mag, norm_grav)
        
        write_cog(str(COG_DIR / "fused" / f"{tile_id}.tif"), fused_anomaly, bounds)
        
    except Exception as e:
        logging.error(f"Tile {tile_id}: Processing failed. Error: {e}", exc_info=True)


def _generate_vrt():
    """Generate VRT mosaic from fused COG tiles."""
    logging.info("Generating VRT from fused COG tiles...")
    vrt_path = FINAL_PRODUCT_DIR / "fused_anomaly.vrt"
    cog_files_path = COG_DIR / "fused" / "*.tif"
    
    cmd = ["gdalbuildvrt", str(vrt_path), str(cog_files_path)]
    subprocess.run(cmd, check=True)
    logging.info(f"VRT successfully created at {vrt_path}")


def _generate_geotiff():
    """Convert VRT to GeoTIFF."""
    logging.info("Converting VRT to GeoTIFF...")
    vrt_path = FINAL_PRODUCT_DIR / "fused_anomaly.vrt"
    geotiff_path = FINAL_PRODUCT_DIR / "fused_anomaly.tif"
    
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


def main():
    """Main processing pipeline."""
    logging.info("--- Starting Global Anomaly Map Processing ---")
    
    _verify_system_dependencies()
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    COG_DIR.joinpath("fused").mkdir(parents=True, exist_ok=True)
    FINAL_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MAG_PATH.exists() or not GRAV_PATH.exists():
        logging.error("FATAL: Source data not found!")
        logging.error(f"Please download required GeoTIFFs to: {RAW_DATA_DIR}")
        sys.exit(1)
    
    tile_ids = tiles_10x10_ids()
    logging.info(f"Found {len(tile_ids)} tiles to process.")
    
    with tqdm(total=len(tile_ids), desc="Processing Tiles") as pbar:
        for tile_id in tile_ids:
            final_cog_path = COG_DIR / "fused" / f"{tile_id}.tif"
            if final_cog_path.exists():
                pbar.update(1)
                continue
            
            _process_tile(tile_id)
            pbar.update(1)
    
    _generate_vrt()
    _generate_geotiff()
    
    logging.info("--- Global Anomaly Map Processing Finished Successfully ---")


if __name__ == "__main__":
    main()