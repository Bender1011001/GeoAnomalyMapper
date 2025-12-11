#!/usr/bin/env python3
"""
GeoAnomalyMapper Main Workflow
==============================

This script ties together the entire pipeline:
1. Download required data for a given region
2. Compute InSAR derived features
3. Perform gravity inversion using PINN
4. Combine features into a multi‑resolution mosaic
5. Classify anomalies and produce final outputs
"""
import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
import argparse
import logging
import sys
import shutil
from pathlib import Path
import time
import numpy as np
import gc

# Local imports
from project_paths import DATA_DIR, OUTPUTS_DIR, ensure_directories
from download_usa_seasonal import download_usa_dataset as download_usa_seasonal
from download_usa_coherence import main as download_usa_coherence
from download_lithology import create_synthetic_lithology as download_lithology
from fetch_lithology_density import main as fetch_lithology_density
from download_licsar import main as download_licsar
from generate_residual import main as generate_residual
from process_data import main as process_data_main
from insar_features import extract_insar_features as compute_insar_features
from pinn_gravity_inversion import invert_gravity as run_gravity_inversion
from multi_resolution_fusion import main as fuse_multi_resolution
from classify_anomalies import main as classify_anomalies
from create_visualization import main as create_visualization

# Suppress overly verbose logging from some libraries
logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_logging(output_name: Path):
    """Configure logging to console only (stdout/stderr redirected by parent process)."""
    # First remove any existing handlers to avoid duplicate logs from submodules
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Since stdout/stderr are redirected by batch_processor.py, only use console handler
    # which will write to the redirected streams
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler]
    )


def run_workflow(
    region: tuple,
    resolution: float,
    output_name: Path,
    mode: str = 'mineral',
    skip_visuals: bool = False
):
    """
    Execute the entire pipeline for a given region.

    Args:
        region: (lon_min, lat_min, lon_max, lat_max)
        resolution: degrees per pixel
        output_name: path prefix for outputs (without extension)
        mode: 'mineral' or 'void' – controls classification thresholds
        skip_visuals: whether to skip the final visualization step
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Convert output_name to Path if needed
    output_name = Path(output_name)
    output_dir = output_name.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_name)

    logger.info(f"Starting GeoAnomalyMapper for region {region} at {resolution}°/px")
    logger.info(f"Output files will be prefixed with: {output_name}")

    # 1. Download baseline data
    logger.info("=== STEP 1: Downloading baseline datasets ===")
    
    # Create a unique directory for this run's intermediate data to avoid file locking collisions
    # when running in batch mode.
    tile_dir = output_name.parent / f"{output_name.name}_data"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Some downloads are safe to share (raw data), but processing steps like fetch_lithology_density
        # which create rasterized outputs should be isolated.
        download_usa_seasonal(region=region, resolution=resolution, output_dir=DATA_DIR)
        download_usa_coherence(region=region, resolution=resolution, output_dir=DATA_DIR)
        download_lithology(region=region, resolution=resolution, output_dir=DATA_DIR)
        
        # fetch_lithology_density creates a raster in output_dir/processed/lithology_density.tif
        # We pass tile_dir so it creates tile_dir/processed/lithology_density.tif
        fetch_lithology_density(region=region, resolution=resolution, output_dir=tile_dir)
    except Exception as e:
        logger.exception("Failed to download baseline data")
        sys.exit(1)

    # 2. Process and reproject static data to match the region exactly
    logger.info("=== STEP 2: Pre-processing static rasters ===")
    
    # Create a unique directory for this run's intermediate data to avoid file locking collisions
    # when running in batch mode.
    tile_dir = output_name.parent / f"{output_name.name}_data"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Call process_all_data directly to avoid argparse issues when running as a module
        from process_data import process_all_data
        process_all_data(region=region, resolution=resolution, output_dir=tile_dir / "processed")
    except Exception as e:
        logger.exception("Processing static data failed")
        sys.exit(1)

    # 3. Download LiCSAR products (if region is covered)
    logger.info("=== STEP 3: Downloading LiCSAR InSAR data ===")
    try:
        # Note: download_licsar expects frame IDs, not a region.
        # We skip this step if no frames are provided, or implement a frame finder.
        logger.warning("Automatic LiCSAR frame detection from region not yet implemented. Skipping download.")
        # download_licsar(region=region, output_dir=DATA_DIR)
    except Exception as e:
        logger.exception("LiCSAR download failed – will skip InSAR features")
        pass

    # 4. Compute InSAR‑derived features (if data exists)
    logger.info("=== STEP 4: Computing InSAR features ===")
    insar_success = False
    try:
        insar_dir = DATA_DIR / "raw" / "insar"
        # Find all coherence files recursively (supporting both .geo.cc.tif and _COH12.tif)
        coh_paths = list(insar_dir.rglob("*.geo.cc.tif")) + list(insar_dir.rglob("*_COH12.tif"))
        
        if coh_paths:
             coh_mean_path = DATA_DIR / "processed" / "insar" / "coherence_mean.tif"
             coh_mean_path.parent.mkdir(parents=True, exist_ok=True)
             
             coh_paths_str = [str(p) for p in coh_paths]
             
             # Check if we have enough data to compute features
             if len(coh_paths) >= 2:
                 # Compute mean coherence if it doesn't exist
                 if not coh_mean_path.exists():
                     logger.info("Computing mean coherence from available tiles...")
                     try:
                         import rasterio
                         # Load first to get profile
                         with rasterio.open(coh_paths_str[0]) as src:
                             profile = src.profile
                             shape = src.shape
                         
                         # Accumulate sum
                         sum_arr = np.zeros(shape, dtype=np.float32)
                         count_arr = np.zeros(shape, dtype=np.float32)
                         
                         for p in coh_paths_str:
                             with rasterio.open(p) as src:
                                 data = src.read(1)
                                 # Handle nodata
                                 mask = (data != src.nodata) & np.isfinite(data)
                                 sum_arr[mask] += data[mask]
                                 count_arr[mask] += 1
                         
                         # Compute mean
                         mean_arr = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr > 0)
                         
                         # Save
                         profile.update(dtype=rasterio.float32, count=1, compress='lzw')
                         with rasterio.open(coh_mean_path, 'w', **profile) as dst:
                             dst.write(mean_arr, 1)
                         logger.info(f"Saved mean coherence to {coh_mean_path}")
                     except Exception as e:
                         logger.error(f"Failed to compute mean coherence: {e}")
                 
                 if coh_mean_path.exists():
                     compute_insar_features(coh_paths_str, str(coh_mean_path))
                     insar_success = True
                 else:
                     logger.warning("Mean coherence missing and could not be computed. Skipping InSAR features.")
             else:
                 logger.info("Insufficient InSAR coherence files found. Skipping feature extraction.")
        else:
             logger.info("No InSAR coherence files found. Skipping feature extraction.")

    except Exception as e:
        logger.exception("Could not compute InSAR features – pipeline continues without them")

    # 5. Gravity inversion (PINN)
    logger.info("=== STEP 5: Gravity inversion with PINN ===")
    step_start = time.time()
    try:
        # invert_gravity(tif_path, output_path, lithology_path=None, target_mode=None)

        gravity_input = tile_dir / "processed" / "gravity" / "gravity_residual_wavelet.tif"
        density_output = OUTPUTS_DIR / f"{output_name.name}_density_model.tif"
        # Use the lithology density map created in Step 1 (which uses tile_dir)
        lithology_input = tile_dir / "processed" / "lithology_density.tif"

        if gravity_input.exists():
            logger.info(f"PINN input file exists: {gravity_input}")
            run_gravity_inversion(
                tif_path=str(gravity_input),
                output_path=str(density_output),
                lithology_path=str(lithology_input) if lithology_input.exists() else None,
                target_mode=mode
            )
            logger.info(f"PINN inversion completed in {time.time() - step_start:.1f}s")
        else:
            logger.error(f"Gravity input file not found: {gravity_input}")
            # We can't proceed without gravity in this pipeline
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Gravity inversion failed after {time.time() - step_start:.1f}s")
        sys.exit(1)

    # 6. Multi-resolution feature fusion (mosaic)
    logger.info("=== STEP 6: Multi-resolution feature fusion ===")
    try:
        # Build list of feature rasters that exist
        # Note: We need to check what files were actually produced by previous steps
        # The PINN output name was constructed as: OUTPUTS_DIR / f"{output_name.name}_density_model.tif"
        
        pinn_output = OUTPUTS_DIR / f"{output_name.name}_density_model.tif"
        
        feature_files = [
            tile_dir / "processed" / "magnetic" / "magnetic_processed.tif",
            tile_dir / "processed" / "gravity" / "gravity_processed.tif",
            tile_dir / "processed" / "lithology_density.tif",
            tile_dir / "processed" / "dem" / "dem_processed.tif",
            pinn_output,
        ]
        # Add InSAR features if they exist
        if insar_success:
            feature_files.append(tile_dir / "processed" / "insar" / "insar_velocity.tif")
            feature_files.append(tile_dir / "processed" / "insar" / "insar_decorrelation.tif")

        # Filter for existence
        existing = [str(f) for f in feature_files if f.exists()]
        if not existing:
            logger.error("No feature rasters found! Cannot fuse.")
            sys.exit(1)

        logger.info(f"Fusing {len(existing)} features")

        # Perform fusion
        mosaic_path, uncertainty_path = fuse_multi_resolution(
            region=region,
            resolution=resolution,
            feature_paths=existing,
            output_prefix=output_name,
        )

        # Explicitly delete large objects to free memory
        del existing
        gc.collect()

    except Exception as e:
        logger.exception("Feature fusion failed")
        sys.exit(1)

    # 7. Anomaly classification
    logger.info("=== STEP 7: Anomaly classification ===")
    try:
        classify_anomalies(
            mosaic_path=mosaic_path,
            output_name=output_name,
            mode=mode,
            uncertainty_path=uncertainty_path,
        )
    except Exception as e:
        logger.exception("Classification failed")
        sys.exit(1)

    # 8. Visualisation (optional)
    if not skip_visuals:
        logger.info("=== STEP 8: Creating visualizations ===")
        try:
            create_visualization(
                region=region,
                output_name=output_name,
                mode=mode,
            )
        except Exception as e:
            logger.exception("Visualization failed – non‑critical, continuing")

    elapsed = time.time() - start_time
    logger.info(f"Workflow completed in {elapsed:.2f} seconds")
    print(f"\n[SUCCESS] Pipeline finished. Results saved to {output_name}.*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoAnomalyMapper main workflow")
    parser.add_argument("--region", required=True, help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--resolution", type=float, default=0.001, help="Output resolution (degrees)")
    parser.add_argument("--output-name", required=True, help="Path prefix for output files (no extension)")
    parser.add_argument("--mode", choices=["mineral", "void"], default="mineral", help="Detection mode")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip final visualisation step")

    args = parser.parse_args()

    # Parse region string
    try:
        region = tuple(map(float, args.region.split(',')))
        if len(region) != 4:
            raise ValueError
    except Exception:
        print("Invalid region format. Use: min_lon,min_lat,max_lon,max_lat")
        sys.exit(1)

    # Ensure required directories exist
    ensure_directories()

    # Run the workflow
    run_workflow(
        region=region,
        resolution=args.resolution,
        output_name=Path(args.output_name),
        mode=args.mode,
        skip_visuals=args.skip_visuals
    )
