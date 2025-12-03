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
    return clip_and_reproject_raster(gravity_files[0], output_file, region, resolution)


def process_magnetic_data(region: Tuple[float, float, float, float], resolution: float) -> bool:
    logger.info("\n%s\nPROCESSING MAGNETIC DATA\n%s", "=" * 70, "=" * 70)
    magnetic_file = RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
    output_file = PROCESSED_DIR / "magnetic" / "magnetic_processed.tif"
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
    insar_dir = RAW_DIR / "insar" / "sentinel1"
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
