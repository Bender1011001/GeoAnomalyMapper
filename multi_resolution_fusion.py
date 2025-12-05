#!/usr/bin/env python3
"""
Bayesian Compressive Sensing (BCS) Downscaling for High-Resolution Gravity Prior.

Approximates BCS using Gaussian Process Regression (GPR) to downscale low-resolution
gravity data (e.g., gravity_tdr.tif at ~4km effective scale) using high-resolution
covariates (DEM, InSAR texture, structural artificiality).

Workflow:
1. Load low-res gravity (target variable y).
2. Load high-res covariates (features X).
3. Coarsen covariates to match low-res gravity scale.
4. Train GPR: y_low ~ f(X_low).
5. Predict high-res gravity: y_high = f(X_high).
6. Output: gravity_prior_highres.tif.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from scipy.ndimage import uniform_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from project_paths import DATA_DIR, OUTPUTS_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = OUTPUTS_DIR / "fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_match_grid(
    src_path: Path,
    match_path: Path,
    resampling: Resampling = Resampling.bilinear
) -> Tuple[np.ndarray, dict]:
    """
    Load a raster and reproject/resample it to match the grid of another raster.
    """
    with rasterio.open(match_path) as match_ds:
        match_profile = match_ds.profile
        match_transform = match_ds.transform
        match_crs = match_ds.crs
        match_shape = (match_ds.height, match_ds.width)

    with rasterio.open(src_path) as src_ds:
        dst_array = np.zeros(match_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src_ds, 1),
            destination=dst_array,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=match_transform,
            dst_crs=match_crs,
            resampling=resampling,
            dst_nodata=np.nan
        )
        
        # Update profile
        dst_profile = match_profile.copy()
        dst_profile.update({
            'dtype': 'float32',
            'nodata': np.nan,
            'driver': 'GTiff'
        })
        
    return dst_array, dst_profile

def bayesian_downscaling(
    lowres_path: Path,
    hires_covariates_paths: List[Path],
    output_highres_path: Path,
    target_res_meters: float = 100.0,
    lowres_scale_meters: float = 4000.0
):
    """
    Perform BCS downscaling using GPR.

    Args:
        lowres_path: Path to low-resolution gravity raster (e.g., TDR).
        hires_covariates_paths: List of paths to high-res covariates (DEM, Texture, etc.).
        output_highres_path: Output path for downscaled gravity.
        target_res_meters: Target resolution in meters (approx).
        lowres_scale_meters: Effective resolution of input gravity in meters.
    """
    logger.info(f"Starting BCS downscaling for {lowres_path.name}")
    
    if not lowres_path.exists():
        logger.error(f"Low-res file not found: {lowres_path}")
        return

    # 1. Establish High-Res Grid (using the first covariate or the lowres file if no covariates)
    # Ideally, we use the DEM as the master grid
    master_grid_path = hires_covariates_paths[0] if hires_covariates_paths else lowres_path
    if not master_grid_path.exists():
        logger.error(f"Master grid file not found: {master_grid_path}")
        return

    logger.info(f"Using {master_grid_path.name} as master grid definition.")
    
    # Load covariates on high-res grid
    X_high_list = []
    master_profile = None
    
    for cov_path in hires_covariates_paths:
        if not cov_path.exists():
            logger.warning(f"Covariate not found: {cov_path}, skipping.")
            continue
        
        data, profile = load_and_match_grid(cov_path, master_grid_path)
        if master_profile is None:
            master_profile = profile
            
        # Normalize covariate
        valid_mask = ~np.isnan(data)
        if np.any(valid_mask):
            mean = np.nanmean(data)
            std = np.nanstd(data)
            if std > 0:
                data = (data - mean) / std
            else:
                data = data - mean
        
        X_high_list.append(data)

    if not X_high_list:
        logger.error("No valid covariates found. Cannot perform downscaling.")
        return

    # Stack high-res covariates
    # Shape: (H, W, n_features)
    X_high_stack = np.stack(X_high_list, axis=-1)
    height, width, n_features = X_high_stack.shape
    
    # 2. Prepare Training Data (Low-Res)
    # Load low-res target variable, aligned to high-res grid (interpolated)
    y_high_interp, _ = load_and_match_grid(lowres_path, master_grid_path)
    
    # Coarsen to effective resolution for training
    # Calculate downsampling factor
    # Assuming master grid is ~target_res_meters
    # We want to sample at lowres_scale_meters
    scale_factor = int(lowres_scale_meters / target_res_meters)
    scale_factor = max(1, scale_factor)
    
    logger.info(f"Coarsening factor: {scale_factor} (Target: {target_res_meters}m, Source: {lowres_scale_meters}m)")
    
    # Create training set by decimating the grid
    # We take every k-th pixel to simulate the low-res observations
    # Alternatively, we could block-average, but decimation is simpler for point-wise GPR mapping
    
    # Subsample indices
    rows = np.arange(0, height, scale_factor)
    cols = np.arange(0, width, scale_factor)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    
    # Extract training samples
    X_train = X_high_stack[rr, cc, :].reshape(-1, n_features)
    y_train = y_high_interp[rr, cc].reshape(-1)
    
    # Filter NaNs
    valid_train = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    
    logger.info(f"Training samples: {len(y_train)}")
    
    # Subsample if too many points (GPR scales poorly > 10k)
    MAX_TRAIN = 5000
    if len(y_train) > MAX_TRAIN:
        logger.info(f"Subsampling training data to {MAX_TRAIN} points...")
        indices = np.random.choice(len(y_train), MAX_TRAIN, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
    # 3. Train GPR
    logger.info("Training Gaussian Process Regressor...")
    # Kernel: Constant * RBF + WhiteNoise (implicit in alpha)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, random_state=42)
    
    gpr.fit(X_train, y_train)
    logger.info(f"GPR trained. Kernel: {gpr.kernel_}")
    
    # 4. Predict High-Res
    logger.info("Predicting high-resolution gravity prior...")
    
    # Reshape X_high for prediction
    X_pred = X_high_stack.reshape(-1, n_features)
    
    # Predict in chunks to avoid memory issues
    y_pred_flat = np.full(height * width, np.nan, dtype=np.float32)
    
    # Mask for valid prediction pixels (where covariates are valid)
    valid_pred_mask = ~np.any(np.isnan(X_pred), axis=1)
    valid_indices = np.where(valid_pred_mask)[0]
    
    CHUNK_SIZE = 100000
    total_chunks = (len(valid_indices) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(valid_indices))
        chunk_indices = valid_indices[start_idx:end_idx]
        
        if len(chunk_indices) == 0:
            continue
            
        X_chunk = X_pred[chunk_indices]
        y_chunk = gpr.predict(X_chunk)
        y_pred_flat[chunk_indices] = y_chunk
        
        if i % 10 == 0:
            logger.info(f"Predicted chunk {i+1}/{total_chunks}")
            
    y_pred = y_pred_flat.reshape(height, width)
    
    # 5. Save Output
    logger.info(f"Saving output to {output_highres_path}")
    with rasterio.open(
        output_highres_path,
        'w',
        **master_profile
    ) as dst:
        dst.write(y_pred, 1)
        dst.set_band_description(1, "BCS High-Res Gravity Prior")
        
    logger.info("BCS downscaling complete.")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: BCS Downscaling")
    parser.add_argument("--target-res", type=float, default=100.0, help="Target resolution in meters")
    args = parser.parse_args()

    # Define inputs
    # Phase 1 output
    gravity_tdr = PROCESSED_DIR / "gravity" / "gravity_tdr.tif"
    if not gravity_tdr.exists():
        # Fallback to residual if TDR not found
        gravity_tdr = PROCESSED_DIR / "gravity" / "gravity_residual.tif"
    
    # Phase 2 outputs (Covariates)
    dem_path = PROCESSED_DIR / "dem" / "dem_processed.tif"
    artif_path = PROCESSED_DIR / "insar" / "structural_artificiality.tif"
    ccd_path = PROCESSED_DIR / "insar" / "coherence_change.tif"
    
    covariates = [p for p in [dem_path, artif_path, ccd_path] if p.exists()]
    
    if not gravity_tdr.exists():
        logger.error("Gravity input not found. Run Phase 1 first.")
        sys.exit(1)
        
    output_path = OUTPUT_DIR / "gravity_prior_highres.tif"
    
    bayesian_downscaling(
        lowres_path=gravity_tdr,
        hires_covariates_paths=covariates,
        output_highres_path=output_path,
        target_res_meters=args.target_res,
        lowres_scale_meters=4000.0 # Approx XGM2019e resolution
    )

if __name__ == "__main__":
    main()
