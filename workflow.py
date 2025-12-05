#!/usr/bin/env python3
"""
High-level workflow runner for GeoAnomalyMapper v2.0.
Orchestrates the full v2 pipeline: gravity processing (CWT residual + TDR),
InSAR texture/artificiality features, Poisson magnetic-gravity correlation,
Bayesian downscaling fusion, Dempster-Shafer void belief fusion, OC-SVM/IF classification.

CLI usage matches batch_processor.py expectations:
python workflow.py --region "lon_min,lat_min,lon_max,lat_max" --resolution 0.001 --output-name "path/to/output_prefix"
All intermediate and final outputs prefixed with output-name (e.g., tile001_gravity_residual.tif).

Supports tiled batch processing without file conflicts (parallel-safe).
Robust skipping of optional data sources (InSAR, DEM).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
import project_paths
from utils.raster_utils import clip_and_reproject_raster

# Phase-specific imports
from process_data import wavelet_decompose_gravity, compute_tilt_derivative
from insar_features import (
    compute_coherence_change_detection,
    compute_glcm_texture,
    compute_structural_artificiality,
)
from poisson_analysis import analyze_poisson_correlation
from multi_resolution_fusion import bayesian_downscaling
from detect_voids import dempster_shafer_fusion
from classify_anomalies import classify_dumb_candidates


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("workflow.log")],
)
logger = logging.getLogger(__name__)


def parse_region(region_str: str) -> Tuple[float, float, float, float]:
    """Parse region string to bounding box tuple."""
    parts = [float(v.strip()) for v in region_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Region must be 'lon_min,lat_min,lon_max,lat_max'")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("Invalid bounds: min >= max")
    return tuple(parts)


def run_full_workflow(
    region: Tuple[float, float, float, float],
    resolution: float,
    output_prefix: str,
    skip_visuals: bool = True,
) -> dict:
    """
    Execute the complete v2 pipeline.

    Args:
        region: Bounding box (lon_min, lat_min, lon_max, lat_max).
        resolution: Output resolution in degrees (~0.001 = 100m).
        output_prefix: Base path prefix for all outputs (e.g., "outputs/tile001").
        skip_visuals: If True, skip visualization generation.

    Returns:
        Dict summary of successes per step.
    """
    prefix_path = Path(output_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 1: Gravity processing (clip + CWT residual + TDR)
    logger.info("=" * 70)
    logger.info("STEP 1: PROCESSING GRAVITY DATA (CWT Residual + TDR)")
    logger.info("=" * 70)

    gravity_processed_path = f"{output_prefix}_gravity_processed.tif"
    gravity_residual_path = f"{output_prefix}_gravity_residual.tif"
    gravity_tdr_path = f"{output_prefix}_gravity_tdr.tif"

    gravity_files = sorted((project_paths.RAW_DIR / "gravity").glob("*.tif*"))
    step1_success = False
    if gravity_files:
        gravity_file = sorted(gravity_files, key=lambda p: p.stat().st_mtime)[-1]
        logger.info(f"Clipping {gravity_file.name}...")
        if clip_and_reproject_raster(gravity_file, Path(gravity_processed_path), region, resolution):
            logger.info("Computing CWT residual...")
            wavelet_decompose_gravity(Path(gravity_processed_path), Path(gravity_residual_path))
            logger.info("Computing Tilt Derivative...")
            compute_tilt_derivative(Path(gravity_residual_path), Path(gravity_tdr_path))
            step1_success = True
    else:
        logger.warning("No gravity files found in %s", project_paths.RAW_DIR / "gravity")
    results["gravity"] = step1_success
    logger.info("Step 1 complete.")

    # Step 2: InSAR features (CCD + GLCM + Artificiality)
    logger.info("=" * 70)
    logger.info("STEP 2: INSAR FEATURES EXTRACTION")
    logger.info("=" * 70)

    insar_dir = project_paths.RAW_DIR / "insar" / "sentinel1"
    coh_raw_paths = sorted(insar_dir.glob("*.tif"))
    art_path = f"{output_prefix}_structural_artificiality.tif"
    ccd_path = f"{output_prefix}_coherence_change.tif"
    homog_path = f"{output_prefix}_texture_homogeneity.tif"
    entropy_path = f"{output_prefix}_texture_entropy.tif"  # Unused downstream
    coh_mean_path = f"{output_prefix}_coh_mean.tif"

    step2_success = False
    if len(coh_raw_paths) >= 2:
        logger.info(f"Clipping {len(coh_raw_paths)} coherence rasters...")
        coh_clipped_paths: List[str] = []
        for i, coh_path in enumerate(coh_raw_paths):
            coh_clip_path = f"{output_prefix}_coh_{i+1:03d}.tif"
            if clip_and_reproject_raster(coh_path, Path(coh_clip_path), region, resolution):
                coh_clipped_paths.append(coh_clip_path)
        if len(coh_clipped_paths) >= 2:
            # Compute mean coherence
            with rasterio.open(coh_clipped_paths[0]) as src:
                profile = src.profile
            coh_stack = [rasterio.open(p).read(1, masked=True).filled(np.nan) for p in coh_clipped_paths]
            coh_mean_data = np.nanmean(coh_stack, axis=0)
            with rasterio.open(coh_mean_path, "w", **profile) as dst:
                dst.write(coh_mean_data.astype(np.float32), 1)

            # Features
            logger.info("Computing coherence change detection...")
            compute_coherence_change_detection(coh_clipped_paths, ccd_path)
            logger.info("Computing GLCM texture...")
            compute_glcm_texture(coh_mean_path, homog_path, entropy_path)
            logger.info("Computing structural artificiality...")
            compute_structural_artificiality(ccd_path, homog_path, art_path)
            step2_success = Path(art_path).exists()
    else:
        logger.warning("Insufficient InSAR files (<2) in %s", insar_dir)
    results["insar"] = step2_success
    logger.info("Step 2 complete.")

    # Step 3: Poisson correlation (gravity residual + magnetic pseudo-gravity)
    logger.info("=" * 70)
    logger.info("STEP 3: POISSON CORRELATION ANALYSIS")
    logger.info("=" * 70)

    mag_raw_path = project_paths.RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
    mag_path = f"{output_prefix}_magnetic.tif"
    poisson_path = f"{output_prefix}_poisson_correlation.tif"

    step3_success = False
    if step1_success and mag_raw_path.exists():
        logger.info("Clipping magnetic data...")
        if clip_and_reproject_raster(mag_raw_path, Path(mag_path), region, resolution):
            logger.info("Computing Poisson correlation...")
            analyze_poisson_correlation(gravity_residual_path, mag_path, poisson_path)
            step3_success = True
    else:
        logger.warning("Skipping Poisson: missing gravity or magnetic.")
    results["poisson"] = step3_success
    logger.info("Step 3 complete.")

    # Step 4: Multi-resolution fusion (BCS downscaling)
    logger.info("=" * 70)
    logger.info("STEP 4: BAYESIAN COMPRESSIVE SENSING DOWNSCALING")
    logger.info("=" * 70)

    prior_path = f"{output_prefix}_gravity_prior_highres.tif"
    covariates_paths: List[str] = []

    # Optional DEM
    dem_dir = project_paths.RAW_DIR / "dem"
    dem_files = list(dem_dir.glob("*.tif"))
    if dem_files:
        dem_clip_path = f"{output_prefix}_dem.tif"
        if clip_and_reproject_raster(dem_files[0], Path(dem_clip_path), region, resolution):
            covariates_paths.append(dem_clip_path)

    # InSAR covariates if available
    if Path(ccd_path).exists():
        covariates_paths.append(ccd_path)
    if Path(art_path).exists():
        covariates_paths.append(art_path)

    step4_success = False
    if step1_success and covariates_paths:
        logger.info("Performing BCS downscaling...")
        bayesian_downscaling(
            Path(gravity_tdr_path),
            [Path(p) for p in covariates_paths],
            Path(prior_path),
        )
        step4_success = True
    else:
        logger.warning("Skipping fusion: missing gravity TDR or covariates.")
    results["fusion"] = step4_success
    logger.info("Step 4 complete.")

    # Step 5: Dempster-Shafer fusion (void belief)
    logger.info("=" * 70)
    logger.info("STEP 5: DEMPSTER-SHAFER FUSION")
    logger.info("=" * 70)

    belief_path = f"{output_prefix}_fused_belief_reinforced.tif"

    step5_success = False
    if (
        step1_success
        and step2_success
        and step3_success
        and Path(gravity_tdr_path).exists()
        and Path(art_path).exists()
        and Path(poisson_path).exists()
    ):
        logger.info("Fusing evidence sources...")
        dempster_shafer_fusion(
            Path(gravity_tdr_path),
            Path(art_path),
            Path(poisson_path),
            Path(belief_path),
        )
        step5_success = True
    else:
        logger.warning("Skipping D-S fusion: missing inputs.")
    results["void_detection"] = step5_success
    logger.info("Step 5 complete.")

    # Step 6: Anomaly classification (OC-SVM + Isolation Forest)
    logger.info("=" * 70)
    logger.info("STEP 6: ANOMALY CLASSIFICATION (DUMB Candidates)")
    logger.info("=" * 70)

    dumb_path = f"{output_prefix}_dumb_probability_v2.tif"
    feature_candidates = [
        gravity_residual_path,
        belief_path,
        poisson_path,
        art_path,
    ]
    existing_features = [p for p in feature_candidates if Path(p).exists()]

    step6_success = False
    if len(existing_features) >= 1:
        logger.info("Classifying anomalies...")
        classify_dumb_candidates(existing_features, dumb_path)
        step6_success = True
    else:
        logger.warning("Skipping classification: no features available.")
    results["classification"] = step6_success
    logger.info("Step 6 complete.")

    # Summary
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    total_steps = len(results)
    success_count = sum(results.values())
    logger.info("Successful steps: %d/%d", success_count, total_steps)
    for step, success in results.items():
        status = "✓" if success else "✗"
        logger.info("%s %s", status, step)

    # Save log
    log_path = f"{output_prefix}_workflow_log.json"
    summary = {
        "region": region,
        "resolution_deg": resolution,
        "output_prefix": output_prefix,
        "results": results,
        "final_output": dumb_path if step6_success else None,
    }
    Path(log_path).write_text(json.dumps(summary, indent=2))
    logger.info("Log saved: %s", log_path)

    if not skip_visuals:
        logger.info("Generating visualizations (skipped in batch mode)...")
        # TODO: Integrate create_visualization.py if needed

    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="GeoAnomalyMapper v2.0 Full Pipeline",
        epilog="""Example: python workflow.py --region "-105,32,-104,33" --resolution 0.001 --output-name "outputs/test_tile"
        Outputs: test_tile_gravity_residual.tif, test_tile_structural_artificiality.tif, ..., test_tile_dumb_probability_v2.tif""",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Bounding box: 'lon_min,lat_min,lon_max,lat_max'",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.001,
        help="Grid resolution in degrees (default: 0.001 ~100m)",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output file prefix (e.g., 'outputs/tile001')",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visualization generation (default for batch)",
    )
    args = parser.parse_args(argv)

    try:
        region = parse_region(args.region)
    except ValueError as e:
        logger.error("Invalid region: %s", e)
        sys.exit(1)

    return run_full_workflow(
        region=region,
        resolution=args.resolution,
        output_prefix=args.output_name,
        skip_visuals=args.skip_visuals,
    )


if __name__ == "__main__":
    main()
