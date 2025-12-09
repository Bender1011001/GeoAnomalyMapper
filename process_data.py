#!/usr/bin/env python3
"""
GeoAnomalyMapper - Unified Data Processing
==========================================

This utility prepares gravity, magnetic, elevation, InSAR and lithology data
so that the fusion, detection and visualisation steps can run without any
manual file wrangling.  The script works purely on locally available rasters;
downloaded source data must be placed under ``data/raw/`` (or the directory
pointed to by ``GEOANOMALYMAPPER_DATA_DIR``) beforehand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from project_paths import PROCESSED_DIR, RAW_DIR, ensure_directories
from utils.raster_utils import clip_and_reproject_raster
import pywt
from download_lithology import create_synthetic_lithology

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION = 0.001  # ≈100 m

# Ensure predictable directory structure on first run.
ensure_directories(
    [
        RAW_DIR,
        RAW_DIR / "gravity",
        RAW_DIR / "magnetic",
        RAW_DIR / "dem",
        RAW_DIR / "insar",
        PROCESSED_DIR,
    ]
)


def wavelet_decompose_gravity(
    gravity_path: Path,
    output_residual_path: Path,
    wavelet: str = "db4",
    level: Optional[int] = None,
) -> None:
    """
    Compute shallow gravity residual using Discrete Wavelet Transform (DWT).
    
    Decomposes the gravity signal into approximation (deep/low-freq) and detail
    (shallow/high-freq) coefficients. Reconstructs the signal using ONLY the
    detail coefficients to isolate shallow anomalies.

    Args:
        gravity_path: Input gravity GeoTIFF path.
        output_residual_path: Output residual GeoTIFF path.
        wavelet: Wavelet family (default: "db4").
        level: Decomposition level. If None, calculated based on image size.

    Returns:
        None. Writes output file.
    """
    with rasterio.open(gravity_path) as src:
        grav = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        
    # Fill NaNs
    grav = np.where(np.isnan(grav), np.nanmean(grav), grav)

    # Determine max level if not provided
    if level is None:
        level = pywt.dwt_max_level(min(grav.shape), pywt.Wavelet(wavelet).dec_len)
        # Limit level to avoid over-smoothing, usually 3-4 is sufficient for regional/residual separation
        level = min(level, 4)

    # 2D Multilevel decomposition using symmetric padding to reduce edge artifacts
    coeffs = pywt.wavedec2(grav, wavelet, level=level, mode='symmetric')
    
    # coeffs[0] is the approximation (cA) at the coarsest level (deep sources)
    # coeffs[1:] are tuples of details (cH, cV, cD) at finer scales
    
    # To get the residual (shallow sources), we set the approximation to zero
    # Handle list structure of coeffs
    coeffs_residual = [np.zeros_like(c) if i == 0 else c for i, c in enumerate(coeffs)]
    if isinstance(coeffs[0], np.ndarray):
        coeffs_residual[0] = np.zeros_like(coeffs[0])
    
    # Reconstruct
    residual = pywt.waverec2(coeffs_residual, wavelet, mode='symmetric')
    
    # Crop to original shape if reconstruction is slightly larger due to padding
    residual = residual[:grav.shape[0], :grav.shape[1]]

    profile.update(dtype="float32", nodata=np.nan)
    output_residual_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_residual_path, "w", **profile) as dst:
        dst.write(residual.astype(np.float32), 1)
    logger.info("Phase 1: Saved gravity residual (DWT): %s", output_residual_path)


def compute_tilt_derivative(
    residual_path: Path,
    output_tdr_path: Path,
) -> None:
    """
    Compute Tilt Derivative (TDR) = arctan2(VDR, THG).

    VDR ≈ vertical derivative proxy (row gradient, axis=0),
    THG = horizontal gradient magnitude sqrt(dx^2 + dy^2).
    Handles THG=0 via arctan2 (sign(VDR)*pi/2).
    Output range: [-pi/2, pi/2] rad.

    Args:
        residual_path: Input residual GeoTIFF.
        output_tdr_path: Output TDR GeoTIFF.

    Returns:
        None. Writes output file.
    """
    with rasterio.open(residual_path) as src:
        res = src.read(1).astype(np.float64)  # Use float64 for accuracy
        profile = src.profile.copy()
        transform = src.transform

    # Replace NaN with 0 only for gradient calculation
    res_filled = np.nan_to_num(res, nan=0.0)

    # Compute gradients properly using pixel spacing
    dy, dx = np.gradient(res_filled)  # dy = north-south, dx = east-west
    pixel_size_y = transform.a  # Actually x resolution
    pixel_size_x = abs(transform.e)  # y resolution (usually negative)

    # Horizontal gradient magnitude
    hg = np.sqrt((dx / pixel_size_x)**2 + (dy / pixel_size_y)**2)

    # Vertical derivative approximation: often use Laplacian or first vertical
    # But most common TDR uses the *vertical* first derivative ≈ -northward gradient
    vdr = -dy / pixel_size_y  # Negative northward gradient ≈ upward first vertical derivative

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tdr = np.arctan2(vdr, hg)
        tdr = np.where(hg == 0, np.sign(vdr) * np.pi/2, tdr)

    # Restore original nodata
    tdr = np.where(np.isnan(res), np.nan, tdr)

    profile.update(dtype="float32", nodata=np.nan)
    output_tdr_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_tdr_path, "w", **profile) as dst:
        dst.write(tdr.astype(np.float32), 1)
    logger.info("Phase 1: Saved gravity TDR: %s", output_tdr_path)


def _write_text_file(path: Path, content: str) -> None:
    """Write text content to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    logger.info("Wrote %s", path)


def process_gravity_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    """Process gravity data for the requested region."""
    logger.info("\n%s\nPROCESSING GRAVITY DATA\n%s", "=" * 70, "=" * 70)

    gravity_dir = RAW_DIR / "gravity"
    output_dir = PROCESSED_DIR / "gravity"
    output_dir.mkdir(parents=True, exist_ok=True)

    gravity_files = sorted(gravity_dir.glob("*.tif*"))
    if not gravity_files:
        instructions = f"""GRAVITY DATA PROCESSING INSTRUCTIONS
============================================================
1. Visit http://icgem.gfz-potsdam.de/calcgrid
2. Select model: XGM2019e_2159
3. Grid type: gravity_disturbance
4. Region: {region[0]:.2f}, {region[1]:.2f}, {region[2]:.2f}, {region[3]:.2f}
5. Grid step: 0.01 degrees (or finer)
6. Output format: GeoTIFF
7. Save the downloaded file to data/raw/gravity/
"""
        _write_text_file(output_dir / "GRAVITY_PROCESSING_INSTRUCTIONS.txt", instructions)
        logger.warning("No gravity GeoTIFF found; instructions generated.")
        return False

    # Handle multiple gravity files - use most recent if multiple exist
    if len(gravity_files) > 1:
        # Sort by modification time, newest first
        gravity_files.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
        logger.info(f"Multiple gravity files found ({len(gravity_files)}). Using most recent: {gravity_files[0].name}")
    else:
        logger.info(f"Using gravity file: {gravity_files[0].name}")
    
    output_file = output_dir / "gravity_processed.tif"
    success = clip_and_reproject_raster(gravity_files[0], output_file, region, resolution)
    
    if success:
        # Automatically run wavelet + TDR after gravity clipping
        residual_path = output_dir / "gravity_residual_wavelet.tif"
        tdr_path = output_dir / "gravity_tdr.tif"
        
        try:
            wavelet_decompose_gravity(output_file, residual_path)
            compute_tilt_derivative(residual_path, tdr_path)
        except Exception as e:
            logger.error(f"Failed to compute gravity derivatives: {e}")
            # Don't fail the whole step, just log error
            
    return success


def process_magnetic_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    logger.info("\n%s\nPROCESSING MAGNETIC DATA\n%s", "=" * 70, "=" * 70)
    magnetic_dir = RAW_DIR / "magnetic"
    magnetic_files = sorted(magnetic_dir.glob("EMAG2*.tif*")) + sorted(magnetic_dir.glob("*.tif"))
    
    if not magnetic_files:
        instructions = f"""MAGNETIC DATA MISSING
================================
Download the global EMAG2 or WDMAM model:
- EMAG2 v3: https://www.ngdc.noaa.gov/geomag/emag2.html
- Or regional high-res data from Geoscience Australia / USGS
Place the GeoTIFF in data/raw/magnetic/
"""
        (PROCESSED_DIR / "magnetic").mkdir(parents=True, exist_ok=True)
        _write_text_file(PROCESSED_DIR / "magnetic" / "MAGNETIC_INSTRUCTIONS.txt", instructions)
        logger.warning("No magnetic data found in %s", magnetic_dir)
        return False

    # Use most recent
    magnetic_file = sorted(magnetic_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logger.info("Using magnetic file: %s", magnetic_file.name)
    
    output_file = PROCESSED_DIR / "magnetic" / "magnetic_processed.tif"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return clip_and_reproject_raster(magnetic_file, output_file, region, resolution)


def process_dem_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    logger.info("\n%s\nPROCESSING DEM DATA\n%s", "=" * 70, "=" * 70)
    dem_dir = RAW_DIR / "dem"
    dem_files = sorted(dem_dir.glob("*.tif"))
    if not dem_files:
        logger.warning("No DEM tiles found in %s", dem_dir)
        logger.warning("Download Copernicus DEM 30 m tiles for your region.")
        return False

    dem_output = PROCESSED_DIR / "dem" / "dem_processed.tif"

    # Ensure DEM uses the same target resolution as other layers to maintain a common grid
    return clip_and_reproject_raster(dem_files[0], dem_output, region, resolution)


def process_insar_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    logger.info("\n%s\nPROCESSING INSAR DATA\n%s", "=" * 70, "=" * 70)
    insar_dir = RAW_DIR / "insar"
    output_dir = PROCESSED_DIR / "insar"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not insar_dir.exists() or not any(insar_dir.iterdir()):
        logger.warning("No Sentinel-1 stacks found in %s (optional step).", insar_dir)
        return False

    processed_files = sorted(insar_dir.glob("*.tif"))
    if not processed_files:
        guide = f"""# InSAR Processing Guide

## Option 1: COMET LiCSAR (Recommended)
1. Visit https://comet.nerc.ac.uk/COMET-LiCS-portal/
2. Search for the region {region}
3. Download interferograms (already geocoded)
4. Place them under data/raw/insar/

## Option 2: SNAP / ISCE
See ``process_insar_data.py`` for a full walkthrough of processing steps.
"""
        _write_text_file(output_dir / "INSAR_PROCESSING_GUIDE.md", guide)
        logger.warning("Interferograms not found; guide generated.")
        return False

    output_file = output_dir / "insar_processed.tif"
    return clip_and_reproject_raster(processed_files[0], output_file, region, resolution)


def process_lithology_data(region: Tuple[float, float, float, float]) -> bool:
    logger.info("\n%s\nPROCESSING LITHOLOGY DATA\n%s", "=" * 70, "=" * 70)
    lithology_dir = RAW_DIR / "lithology"
    if not lithology_dir.exists():
        logger.warning("No lithology directory found (optional).")
        return False

    litho_files = sorted(lithology_dir.glob("*.tif")) + sorted(lithology_dir.glob("*.shp"))
    if not litho_files:
        logger.warning("Lithology files not detected; consult USGS/state surveys.")
        return False

    logger.info("Found lithology candidate: %s (processing not yet automated).", litho_files[0].name)
    return False


def process_all_data(
    region: Tuple[float, float, float, float],
    resolution: float = DEFAULT_RESOLUTION,
) -> Dict[str, bool]:
    """Run all individual processors and return a success summary."""

    logger.info("\n%s\nGEOANOMALYMAPPER - DATA PROCESSING\n%s", "=" * 70, "=" * 70)
    logger.info("Region: %s", region)
    logger.info("Output resolution: %.6f° (~%.0f m)\n", resolution, resolution * 111_000)

    results: Dict[str, bool] = {
        'gravity': process_gravity_data(region, resolution),
        'magnetic': process_magnetic_data(region, resolution),
        'dem': process_dem_data(region, resolution),
        'insar': process_insar_data(region, resolution),
        'lithology': process_lithology_data(region),
    }

    logger.info("\n%s\nPROCESSING SUMMARY\n%s", "=" * 70, "=" * 70)
    for dataset, success in results.items():
        status = "SUCCESS" if success else "SKIPPED"
        logger.info("%-12s %s", dataset.upper(), status)

    if results['gravity'] or results['magnetic']:
        logger.info("\nMinimum data available for downstream processing.")
    else:
        logger.warning("\nInsufficient data for fusion/detection (need gravity or magnetic).")

    log_file = PROCESSED_DIR / "processing_log.json"
    log_payload = {
        'region': region,
        'resolution_deg': resolution,
        'results': results,
    }
    log_file.write_text(json.dumps(log_payload, indent=2), encoding='utf-8')
    logger.info("Processing log saved to %s", log_file)
    return results


def _parse_region(value: str) -> Tuple[float, float, float, float]:
    parts = [float(v.strip()) for v in value.split(',')]
    if len(parts) != 4:
        raise ValueError("Expected format lon_min,lat_min,lon_max,lat_max")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("Minimum coordinates must be less than maximum coordinates.")
    return lon_min, lat_min, lon_max, lat_max


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clip and harmonise gravity, magnetic, DEM and optional InSAR data.",
        epilog="Example: python process_data.py --region \"-105.0,32.0,-104.0,33.0\"",
    )
    parser.add_argument(
        '--region',
        required=True,
        type=str,
        help='Bounding box as "lon_min,lat_min,lon_max,lat_max"',
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=DEFAULT_RESOLUTION,
        help='Output grid resolution in degrees (default: 0.001 ≈ 100 m).',
    )
    return parser


def main(argv: list[str] | None = None) -> Dict[str, bool]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        region = _parse_region(args.region)
    except ValueError as exc:
        logger.error("Invalid region: %s", exc)
        sys.exit(1)

    return process_all_data(region, resolution=args.resolution)


if __name__ == "__main__":
    main()
