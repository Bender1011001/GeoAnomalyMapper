#!/usr/bin/env python3
"""
High-level workflow runner for GeoAnomalyMapper.

Instead of invoking each script manually, this CLI wires the processing,
fusion, void detection, visualisation and optional validation steps into a
single command.  The implementation intentionally reuses the existing module
APIs so the same functions power both the CLI and any automated runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import create_visualization
import detect_voids
import multi_resolution_fusion as fusion
import process_data
import validate_against_known_features as validator
from project_paths import OUTPUTS_DIR, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_region(region_str: str) -> Tuple[float, float, float, float]:
    """Parse region string handling spaces and quotes."""
    # Remove quotes and extra spaces
    cleaned = region_str.strip().strip('"').strip("'")
    parts = [float(x.strip()) for x in cleaned.split(',')]
    if len(parts) != 4:
        raise ValueError(f"Region must have 4 values, got {len(parts)}")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("Minimum coordinates must be smaller than maximum coordinates.")
    return tuple(parts)


def run_workflow(
    *,
    region: Tuple[float, float, float, float],
    resolution: float,
    output_name: str,
    skip_preprocessing: bool,
    skip_visuals: bool,
    run_validation: bool,
    validation_buffer_km: float,
) -> Dict[str, object]:
    """Execute the end-to-end workflow and return a summary."""

    ensure_directories()
    summary: Dict[str, object] = {
        "region": region,
        "output_name": output_name,
    }

    # Auto-adjust resolution for large regions to prevent memory exhaustion
    import numpy as np
    lon_min, lat_min, lon_max, lat_max = region
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    mid_lat_rad = np.deg2rad((lat_min + lat_max) / 2)
    approx_area_deg2 = lon_span * lat_span * np.cos(mid_lat_rad)
    target_pixels = 40_000_000  # Conservative limit (~160 MB float32 array)
    auto_resolution = max(resolution, np.sqrt(approx_area_deg2 / target_pixels))
    if auto_resolution > resolution * 1.1:  # 10% tolerance
        logger.warning(
            "Large region detected (approx. area: %.0f deg²). "
            "Auto-adjusting resolution from %.6f° → %.6f° (~%.0f m/pixel) "
            "to prevent out-of-memory errors.",
            approx_area_deg2, resolution, auto_resolution, auto_resolution * 111_000
        )
        resolution = auto_resolution
    summary["resolution_deg"] = resolution
    logger.info("Effective processing resolution: %.6f°", resolution)

    if not skip_preprocessing:
        logger.info("Step 1/4: Processing raw datasets")
        summary["processing"] = process_data.process_all_data(region, resolution=resolution)
    else:
        logger.info("Skipping preprocessing step (per CLI option).")

    logger.info("Step 2/4: Multi-resolution fusion")
    try:
        fused_tif = fusion.process_multi_resolution(
            region,
            target_resolution=resolution,
            output_name=output_name,
            data_sources=None,
        )
        summary["fusion_tif"] = fused_tif
    except Exception as e:
        logger.error(f"Multi-resolution fusion failed: {e}", exc_info=True)
        summary["fusion_tif"] = None
        summary["fusion_error"] = str(e)
        # Continue with partial results - other steps may still be valuable

    logger.info("Step 3/4: Void probability mapping")
    void_output_name = f"{output_name}_void_probability"
    void_tif = detect_voids.process_region(region, resolution, void_output_name)
    summary["void_probability_tif"] = void_tif

    if not skip_visuals:
        logger.info("Step 4/4: Visualization bundle")
        visuals = create_visualization.generate_visualization_bundle(void_tif)
        summary["visuals"] = visuals
    else:
        logger.info("Skipping visualization generation.")

    if run_validation:
        logger.info("Running validation against known features (buffer %.1f km)", validation_buffer_km)
        validation = validator.validate_features(
            fused_tif,
            validator.KNOWN_FEATURES,
            buffer_km=validation_buffer_km,
        )
        validation_dir = OUTPUTS_DIR / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        report_path = validation_dir / f"{output_name}_validation_report.txt"
        map_path = validation_dir / f"{output_name}_validation_map.png"
        validator.generate_validation_report(validation, report_path)
        validator.create_validation_map(validation, fused_tif, map_path)
        summary["validation"] = {
            "report": report_path,
            "map": map_path,
            "summary": validation["summary"],
        }
    else:
        logger.info("Validation skipped. Use --validate to enable it.")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full GeoAnomalyMapper processing + analysis workflow.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help='Bounding box as "lon_min,lat_min,lon_max,lat_max".',
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=process_data.DEFAULT_RESOLUTION,
        help="Target grid resolution in degrees (default: 0.001 ≈ 100 m).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fused_anomaly",
        help="Base name for fused outputs.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Assume processed rasters already exist and skip process_data.py.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip PNG/KMZ generation (useful for headless runs).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Sample the fused raster against the built-in feature catalogue.",
    )
    parser.add_argument(
        "--validation-buffer",
        type=float,
        default=2.0,
        help="Validation sampling buffer radius in km (default: 2 km).",
    )
    return parser


def main(argv: list[str] | None = None) -> Dict[str, object]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        region = _parse_region(args.region)
    except ValueError as exc:
        logger.error("Invalid region: %s", exc)
        sys.exit(1)

    summary = run_workflow(
        region=region,
        resolution=args.resolution,
        output_name=args.output_name,
        skip_preprocessing=args.skip_preprocessing,
        skip_visuals=args.skip_visuals,
        run_validation=args.validate,
        validation_buffer_km=args.validation_buffer,
    )

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Region           : {summary['region']}")
    print(f"Resolution (deg) : {summary['resolution_deg']}")
    print(f"Fusion raster    : {summary['fusion_tif']}")
    print(f"Void probability : {summary['void_probability_tif']}")

    visuals = summary.get("visuals")
    if visuals:
        print("Visualization    :")
        for label, path in visuals.items():
            print(f"  - {label.replace('_', ' ').title()}: {path}")

    validation = summary.get("validation")
    if validation:
        vs = validation["summary"]
        print("Validation       :")
        print(f"  - Success rate : {vs['success_rate']:.1%}")
        print(f"  - Report       : {validation['report']}")
        print(f"  - Map          : {validation['map']}")

    print("=" * 70 + "\n")
    return summary


if __name__ == "__main__":
    main()
