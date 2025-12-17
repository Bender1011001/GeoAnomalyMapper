#!/usr/bin/env python3
"""
Multi-Resolution Feature Fusion via Random Forest Regression.
(Formerly experimental 'BCS' module)

This module performs statistical downscaling of gravity data using high-resolution
covariates (DEM, InSAR, Magnetics). It uses a Random Forest approach to learn
relationships at coarse scales and predict at fine scales.

Note: Earlier versions referred to this as "Bayesian Compressive Sensing" (BCS).
The current implementation favors a robust Non-Parametric (Random Forest) approach
for better empirical performance on geophysical data.
"""

import os
# Fix for some GDAL/Rasterio versions on newer Linux distros
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Robust import for project structure
try:
    from project_paths import DATA_DIR, OUTPUTS_DIR, PROCESSED_DIR
except ImportError:
    # Fallback if project_paths.py is missing
    print("Warning: project_paths module not found. Using local directories.")
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    PROCESSED_DIR = BASE_DIR / "processed"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = OUTPUTS_DIR / "fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_and_reproject_to_match(
    src_path: Path,
    match_profile: dict,
    resampling: Resampling = Resampling.bilinear
) -> np.ndarray:
    """Read source and reproject to match the target grid (for training)."""
    with rasterio.open(src_path) as src:
        dst_shape = (match_profile['height'], match_profile['width'])
        dst_array = np.zeros(dst_shape, dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=match_profile['transform'],
            dst_crs=match_profile['crs'],
            resampling=resampling,
            dst_nodata=np.nan
        )
        return dst_array

def train_model(
    lowres_path: Path,
    covariate_paths: List[Path]
) -> Tuple[RandomForestRegressor, SimpleImputer]:
    """Train RF model: Low_Res_Gravity ~ f(Resampled_Covariates)."""
    logger.info("Preparing training data...")
    
    # 1. Read Target (Low Res)
    with rasterio.open(lowres_path) as src:
        y_data = src.read(1)
        lowres_profile = src.profile
        # Create mask: True where data exists
        if src.nodata is not None:
            lowres_mask = (y_data != src.nodata) & (~np.isnan(y_data))
        else:
            lowres_mask = ~np.isnan(y_data)
        
    # 2. Read Covariates (Resampled to Low Res)
    X_list = []
    for cov_path in covariate_paths:
        logger.info(f"  - Resampling covariate for training: {cov_path.name}")
        cov_data = read_and_reproject_to_match(cov_path, lowres_profile)
        X_list.append(cov_data.flatten())
        
    # 3. Prepare Training Arrays
    y_flat = y_data.flatten()
    X_flat = np.column_stack(X_list)
    
    # Filter valid data (intersection of valid gravity AND valid covariates)
    valid_mask = (lowres_mask.flatten()) & (~np.any(np.isnan(X_flat), axis=1))
    
    X_train = X_flat[valid_mask]
    y_train = y_flat[valid_mask]
    
    logger.info(f"Total valid training samples: {len(y_train)}")
    
    if len(y_train) == 0:
        raise ValueError("No valid training data found (no spatial overlap between gravity and covariates).")

    # Subsample if too large to speed up RF training
    MAX_SAMPLES = 1000000
    if len(y_train) > MAX_SAMPLES:
        logger.info(f"Subsampling to {MAX_SAMPLES} points for efficiency...")
        idx = np.random.choice(len(y_train), MAX_SAMPLES, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # 4. Train
    logger.info("Training Random Forest Regressor...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    rf = RandomForestRegressor(
        n_estimators=60,    
        max_depth=20,       # Slight increase in depth for better high-freq capture
        n_jobs=1,          
        random_state=42,
        min_samples_split=5 # Prevent overfitting on noise
    )
    rf.fit(X_train_imputed, y_train)
    logger.info(f"Model trained. Score (R2): {rf.score(X_train_imputed, y_train):.4f}")
    
    return rf, imputer

def bayesian_downscaling(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    covariate_paths: List[Path],
    output_path: Path,
    master_grid_path: Path,
    block_size: int = 2048
):
    """Predict high-res gravity in windows to manage memory."""
    logger.info(f"Starting prediction. Output: {output_path}")
    
    # Open covariates ONCE to avoid I/O overhead in loop
    src_covs = [rasterio.open(p) for p in covariate_paths]
    
    try:
        with rasterio.open(master_grid_path) as src_master:
            profile = src_master.profile.copy()
            
            # --- FIX: FORCE DRIVER TO GTIFF ---
            profile.update({
                'driver': 'GTiff',   # <--- CRITICAL FIX
                'dtype': 'float32',
                'count': 1,
                'nodata': np.nan,
                'compress': 'lzw',
                'predictor': 2, 
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'bigtiff': 'YES'
            })
            
            width = src_master.width
            height = src_master.height
            transform = src_master.transform
            crs = src_master.crs

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Generate windows list for progress bar
            windows = []
            for row_off in range(0, height, block_size):
                for col_off in range(0, width, block_size):
                    w = min(block_size, width - col_off)
                    h = min(block_size, height - row_off)
                    windows.append(Window(col_off, row_off, w, h))
            
            # Iterate with progress bar
            for window in tqdm(windows, desc="Processing Blocks", unit="blk"):
                window_width = window.width
                window_height = window.height
                
                # Calculate window transform relative to master grid
                win_transform = rasterio.windows.transform(window, transform)
                
                X_window_list = []
                
                # Read from already open handles
                for src_cov in src_covs:
                    dst_arr = np.zeros((window_height, window_width), dtype=np.float32)
                    
                    reproject(
                        source=rasterio.band(src_cov, 1),
                        destination=dst_arr,
                        src_transform=src_cov.transform,
                        src_crs=src_cov.crs,
                        dst_transform=win_transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan
                    )
                    X_window_list.append(dst_arr.flatten())

                # Stack features
                X_window_flat = np.column_stack(X_window_list)
                
                # Check for all-NaN pixels (outside coverage)
                all_nan_mask = np.all(np.isnan(X_window_flat), axis=1)
                
                if np.all(all_nan_mask):
                    y_pred_window = np.full((window_height, window_width), np.nan, dtype=np.float32)
                else:
                    # Impute missing values
                    X_window_imputed = imputer.transform(X_window_flat)
                    
                    # Predict
                    y_pred_flat = model.predict(X_window_imputed)
                    
                    # Restore NaNs where NO data existed
                    y_pred_flat[all_nan_mask] = np.nan
                    
                    # Reshape
                    y_pred_window = y_pred_flat.reshape(window_height, window_width).astype(np.float32)

                # Write to disk
                dst.write(y_pred_window, 1, window=window)
                
                # Explicit cleanup
                del X_window_list, X_window_flat, y_pred_window

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Re-raise so the script stops
        raise e

    finally:
        # Always close handles
        for s in src_covs:
            s.close()
            
    logger.info("Prediction complete.")

def main(
    region: Tuple[float, float, float, float] = None,
    resolution: float = None,
    feature_paths: List[str] = None,
    output_prefix: Path = None
):
    # If called from workflow, use arguments. If called as script, parse args.
    if feature_paths is None:
        parser = argparse.ArgumentParser(description="BCS Gravity Downscaling (Random Forest)")
        parser.parse_args()
        # ... (rest of script logic for standalone execution would go here, but we are adapting for workflow)
        # For now, we'll assume workflow usage or implement basic standalone logic if needed.
        # But since we are fixing the workflow call, we prioritize the function signature.
        pass

    # If feature_paths provided (workflow mode)
    if feature_paths:
        feature_paths = [Path(p) for p in feature_paths]
        
        # Identify Gravity (Target) and Covariates
        # We assume the first valid gravity file found in the list is the target
        gravity_target = None
        covariates = []
        
        for p in feature_paths:
            if "gravity" in p.name and "processed" in p.name: # Prefer processed gravity
                 if gravity_target is None:
                     gravity_target = p
                 else:
                     covariates.append(p) # Treat other gravity as covariate? Or just ignore?
            elif "gravity" in p.name:
                 if gravity_target is None:
                     gravity_target = p
                 else:
                     covariates.append(p)
            else:
                covariates.append(p)
        
        # If no specific gravity found, but we have files, maybe the first one is target?
        # But workflow passes: magnetic, gravity, density, topography, pinn_density
        # We want to fuse them into a belief map.
        # Actually, the fusion logic in this script seems to be:
        # Train RF to predict LowResGravity from ResampledCovariates
        # Then Predict HighResGravity using HighResCovariates
        
        # But for "Mineral Exploration", we might want to fuse Beliefs?
        # Or are we doing "Gravity Downscaling" as the fusion step?
        # The workflow calls it "Multi-resolution feature fusion".
        
        # If we want to fuse everything into a single "Mineral Potential" map:
        # We might treat the PINN density as the primary target or one of the inputs?
        
        # Let's stick to the existing logic: Downscale Gravity using others.
        # But wait, we have PINN density which is already high res (same as gravity input).
        
        # If the goal is just to combine them, maybe we should just stack them?
        # But this script is specifically "BCS Gravity Downscaling".
        
        # Let's assume we want to refine the PINN density using other high-res features if available.
        # Or refine the Gravity TDR using Topo/Mag/etc.
        
        if gravity_target is None:
            # Fallback: Use the first file as target if it exists
            if feature_paths:
                gravity_target = feature_paths[0]
                covariates = feature_paths[1:]
            else:
                logger.error("No feature paths provided.")
                return None, None

        if not covariates:
             # If no covariates, just return the target (maybe resampled)
             logger.warning("No covariates for fusion. Returning target as result.")
             # Just copy target to output
             output_path = output_prefix.with_suffix(".fused.tif")
             import shutil
             shutil.copy(gravity_target, output_path)
             return output_path, None

        # Master Grid: Use the highest resolution covariate (or the first one)
        master_grid_path = covariates[0]
        
        output_path = output_prefix.with_suffix(".fused.tif")
        
        try:
            model, imputer = train_model(gravity_target, covariates)
            bayesian_downscaling(model, imputer, covariates, output_path, master_grid_path)
            return output_path, None # Uncertainty not implemented yet
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            raise e

    # 1. Locate Gravity Source (Standalone Mode)
    # Priority list for gravity inputs
    potential_grav = [
        PROCESSED_DIR / "gravity" / "gravity_tdr.tif",
        PROCESSED_DIR / "gravity" / "gravity_residual.tif",
        OUTPUTS_DIR / "gravity_residual.tif",
        OUTPUTS_DIR / "gravity" / "xgm2019e_gravdist_box_mgal.tif"
    ]
    
    gravity_tdr = None
    for p in potential_grav:
        if p.exists():
            gravity_tdr = p
            break
            
    if not gravity_tdr:
        logger.error(f"Gravity input not found. Checked: {[str(p) for p in potential_grav]}")
        sys.exit(1)

    # 2. Locate Covariates
    covariates = []
    
    # DEM
    dem_path = PROCESSED_DIR / "dem" / "dem_processed.tif"
    if dem_path.exists(): covariates.append(dem_path)
        
    # InSAR
    insar_dir = PROCESSED_DIR / "insar" / "mosaics"
    for v in ["usa_winter_vv_COH12.vrt", "usa_summer_vv_COH12.vrt"]:
        p = insar_dir / v
        if p.exists(): covariates.append(p)

    # Density
    density_map = OUTPUTS_DIR / "usa_tiles" / "final_density_map.tif"
    if density_map.exists(): covariates.append(density_map)

    if not covariates:
        logger.error("No covariates found in processed directories.")
        sys.exit(1)

    # 3. Define Master Grid (High Res Target)
    # Prefer winter coherence for max coverage/res, else DEM
    master_grid_path = covariates[0] # Default
    for cov in covariates:
        if "winter" in cov.name:
            master_grid_path = cov
            break
        if "dem" in cov.name and master_grid_path == covariates[0]:
            master_grid_path = cov

    logger.info("=== Configuration ===")
    logger.info(f"Gravity Target: {gravity_tdr.name}")
    logger.info(f"Covariates ({len(covariates)}): {[p.name for p in covariates]}")
    logger.info(f"Master Grid:    {master_grid_path.name}")
    logger.info("=====================")

    # 4. Run Pipeline
    output_path = OUTPUT_DIR / "fused_belief.tif"
    logger.info(f"Output path set to: {output_path}")

    try:
        model, imputer = train_model(gravity_tdr, covariates)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    try:
        bayesian_downscaling(model, imputer, covariates, output_path, master_grid_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)
        
    if output_path.exists():
        logger.info(f"Success! Output file created at {output_path}")
    else:
        logger.error(f"Output file NOT found at {output_path} after prediction completed.")

if __name__ == "__main__":
    main()