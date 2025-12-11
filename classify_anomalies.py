#!/usr/bin/env python3
"""
Phase 5: Anomaly Classification for GeoAnomalyMapper v2.0.

Implements unsupervised anomaly detection using One-Class SVM (OC-SVM) to model
"normal" geology manifold and Isolation Forest (IF) for outlier ranking. Combines
scores into a 0-1 probability map where high values indicate subsurface anomaly candidates.

Optimized for memory efficiency using windowed processing for prediction.

Key Features:
- Aligns multiple feature rasters to a common grid (first raster as reference).
- Handles NaNs via mean imputation per feature.
- Samples background pixels for training (default: 100k pixels).
- Normalizes combined anomaly scores to [0,1] probability.
- Outputs GeoTIFF with matching CRS/transform.
"""

import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from project_paths import OUTPUTS_DIR, PROCESSED_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_training_data(
    feature_paths: List[str],
    sample_size: int = 100000,
    min_samples: int = 10000
) -> Tuple[np.ndarray, SimpleImputer]:
    """
    Load a subsample of data from all features for training.
    
    Critical fix: Adaptive decimation with validation to ensure sufficient samples.
    
    :param feature_paths: List of aligned raster files containing features
    :param sample_size: Target number of training samples
    :param min_samples: Minimum required valid samples (raises error if not met)
    :return: (X_train_imputed, imputer)
    :raises ValueError: If insufficient valid pixels after NaN filtering
    """
    logger.info("Preparing training data...")
    
    if not feature_paths:
        raise ValueError("At least one feature path required.")

    # Load reference profile
    with rasterio.open(feature_paths[0]) as src:
        profile_ref = src.profile
        height, width = src.height, src.width
        
    total_pixels = height * width
    logger.info(f"Reference grid: {height}×{width} = {total_pixels:,} pixels")
    
    # CRITICAL FIX: Adaptive decimation to prevent catastrophic data loss
    # Previous: decimation=10 caused 422×442 → 42×44 = 1,848 pixels (99% loss)
    # With NaN filtering, only 16 valid samples remained
    # New: decimation=2 retains 25% of pixels for robust model training
    if total_pixels < 250000:  # ~500×500
        decimation = 2  # Retain 25% of pixels
    elif total_pixels < 1000000:  # ~1000×1000
        decimation = 3  # Retain 11% of pixels
    else:
        decimation = 5  # Retain 4% for very large grids
    
    logger.info(f"Using adaptive decimation: {decimation} (preserves ~{100/decimation**2:.1f}% of data)")
    
    X_list = []
    
    # We need to ensure all features are resampled to the SAME shape (the reference shape / decimation)
    # The previous implementation relied on src.read(out_shape=...) but if aspect ratios differ slightly or rounding,
    # it might fail. Better to use the reference shape explicitly.
    
    ref_height = int(height // decimation)
    ref_width = int(width // decimation)
    
    for fpath in feature_paths:
        logger.info(f"  - Loading feature for training: {Path(fpath).name}")
        with rasterio.open(fpath) as src:
            # Read decimated and reproject/resample to match reference grid exactly
            # Even if they are different projections or extents, we want them aligned.
            # Ideally we should use reproject() to be safe, like in prediction.
            
            dst_shape = (ref_height, ref_width)
            dst_data = np.zeros(dst_shape, dtype=np.float32)
            
            # We need to construct the transform for the decimated grid
            # The decimated transform scales pixel size by decimation factor
            dst_transform = src.transform * src.transform.scale(decimation, decimation)
            
            # Wait, if we just want to align to the first image's decimated grid:
            # We should use the first image's profile (decimated) as target.
            
            if fpath == feature_paths[0]:
                # This is the reference, just read it
                out_shape = (src.count, ref_height, ref_width)
                data = src.read(
                    out_shape=out_shape,
                    resampling=Resampling.bilinear
                )
                dst_data = data[0]
                
                # Store reference transform/crs for others
                ref_transform = src.transform * src.transform.scale(decimation, decimation)
                ref_crs = src.crs
            else:
                # Reproject to match reference
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=np.nan
                )
            
            # Flatten
            X_list.append(dst_data.flatten())
            
    X_flat = np.column_stack(X_list)
    
    # Filter NaNs
    # We want to train on valid data
    valid_mask = ~np.any(np.isnan(X_flat), axis=1)
    X_train = X_flat[valid_mask]
    
    logger.info(f"Total valid pixels available for training: {len(X_train)}")
    
    # CRITICAL VALIDATION: Ensure sufficient samples for robust model training
    if len(X_train) < min_samples:
        raise ValueError(
            f"Insufficient training data: {len(X_train)} valid pixels (minimum required: {min_samples}). "
            f"This usually indicates excessive NaN values in input rasters. "
            f"Check that all feature rasters have valid data coverage. "
            f"Decimated grid: {ref_height}×{ref_width} = {ref_height*ref_width} pixels, "
            f"Valid: {len(X_train)} ({100*len(X_train)/(ref_height*ref_width):.1f}%)"
        )
    
    if len(X_train) > sample_size:
        logger.info(f"Subsampling to {sample_size} pixels")
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[idx]
    else:
        logger.warning(f"Using all {len(X_train)} valid pixels (less than target {sample_size})")
        
    # Imputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    return X_train_imputed, imputer

def train_models(
    X_train: np.ndarray
) -> Tuple[StandardScaler, OneClassSVM, IsolationForest]:
    """
    Train OC-SVM and Isolation Forest.
    """
    logger.info("Training models...")
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    logger.info("  - Training OneClassSVM...")
    ocsvm = OneClassSVM(
        nu=0.05, kernel="rbf", gamma="scale"
    ).fit(X_train_scaled)

    logger.info("  - Training IsolationForest...")
    iforest = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=100, n_jobs=1
    ).fit(X_train_scaled)

    return scaler, ocsvm, iforest

def classify_anomaly_candidates(
    feature_paths: List[str],
    output_path: str,
    scaler: StandardScaler,
    ocsvm: OneClassSVM,
    iforest: IsolationForest,
    imputer: SimpleImputer,
    mode: str = 'mineral',
    block_size: int = 2048
):
    """
    Predict anomaly scores in windows and save to disk.
    """
    logger.info(f"Starting prediction. Output: {output_path}")
    
    # Use first feature as master grid
    with rasterio.open(feature_paths[0]) as src_ref:
        profile = src_ref.profile.copy()
        height, width = src_ref.height, src_ref.width
        transform = src_ref.transform
        crs = src_ref.crs
        
    profile.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'bigtiff': 'YES'
    })
    
    # We need to normalize scores globally.
    # Problem: We don't know min/max of scores without predicting everything.
    # Solution: Predict everything, store raw scores in a temporary file (or two bands), 
    # find min/max, then normalize? Or just output raw scores and normalize later?
    # Or estimate min/max from training data?
    # Estimating from training data is risky if anomalies are outliers not in training.
    # Let's output raw combined score first, then we can normalize if needed, 
    # OR just use the raw scores (higher is more anomalous).
    # The prompt asks for "mineral_void_probability.tif" which implies 0-1.
    # Let's do a two-pass approach or just load the raw result to normalize if it fits?
    # No, memory constraint.
    # Let's estimate min/max from a subset of predictions (e.g. the training set predictions).
    
    # Estimate normalization params from training set (which we have in memory? No we didn't return it)
    # Let's just use the models to predict on a random subset again to get stats.
    # Actually, let's just output the raw "anomaly score" which is usually what's needed.
    # But to stick to the requested "probability", we can use a sigmoid or just min-max based on theoretical bounds?
    # IF scores are [-1, 1] (roughly). OCSVM decision function is unbounded but usually small.
    # Let's compute scores on a small sample to get a range.
    
    logger.info("Estimating score range for normalization...")
    # Quick sample
    X_sample, _ = prepare_training_data(feature_paths, sample_size=10000)
    X_sample_scaled = scaler.transform(X_sample)
    oc_scores_samp = -ocsvm.decision_function(X_sample_scaled)
    iso_scores_samp = -iforest.score_samples(X_sample_scaled)
    combined_samp = (oc_scores_samp + iso_scores_samp) / 2.0
    
    p_min = np.percentile(combined_samp, 1) # Use percentiles to be robust
    p_max = np.percentile(combined_samp, 99)
    logger.info(f"Score range estimate: {p_min:.4f} to {p_max:.4f}")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for row_off in range(0, height, block_size):
            for col_off in range(0, width, block_size):
                window_width = min(block_size, width - col_off)
                window_height = min(block_size, height - row_off)
                window = Window(col_off, row_off, window_width, window_height)
                
                # Read all features for this window
                X_window_list = []
                
                for fpath in feature_paths:
                    with rasterio.open(fpath) as src:
                        # Reproject window if needed, but assuming aligned inputs for now
                        # If inputs are not aligned, we must reproject.
                        # The previous script did reproject. We should too.
                        
                        win_transform = rasterio.windows.transform(window, transform)
                        dst_arr = np.zeros((window_height, window_width), dtype=np.float32)
                        
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=dst_arr,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=win_transform,
                            dst_crs=crs,
                            resampling=Resampling.bilinear,
                            dst_nodata=np.nan
                        )
                        X_window_list.append(dst_arr.flatten())
                
                X_window_flat = np.column_stack(X_window_list)
                
                # Impute
                X_window_imputed = imputer.transform(X_window_flat)
                
                # Scale
                X_window_scaled = scaler.transform(X_window_imputed)
                
                # Predict
                oc_scores = -ocsvm.decision_function(X_window_scaled)
                iso_scores = -iforest.score_samples(X_window_scaled)
                combined = (oc_scores + iso_scores) / 2.0
                
                # Normalize
                prob = (combined - p_min) / (p_max - p_min + 1e-12)
                prob = np.clip(prob, 0.0, 1.0)

                # MINERAL-PRO MODE: Relax classification threshold.
                # Apply a power transform < 1.0 to boost lower probabilities,
                # allowing more natural voids (which might have weaker signals) to pass through.
                # Original was linear (power=1.0 implicitly).
                if mode == 'mineral':
                    prob = np.power(prob, 0.65)
                else:
                    # Void mode or default: keep linear (power=1.0)
                    pass
                
                # Mask NaNs
                all_nan = np.all(np.isnan(X_window_flat), axis=1)
                prob[all_nan] = np.nan
                
                # Write
                dst.write(prob.reshape(window_height, window_width), 1, window=window)
                
    logger.info("Classification complete.")

def main(
    mosaic_path: Path = None,
    output_name: Path = None,
    mode: str = 'mineral',
    uncertainty_path: Path = None
):
    # If called from workflow, use arguments. If called as script, parse args.
    if mosaic_path is None:
        parser = argparse.ArgumentParser(description="Phase 5: Anomaly Classification")
        parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "mineral_void_probability.tif"), help="Output path")
        parser.add_argument("--mode", choices=["mineral", "void"], default="mineral", help="Detection mode")
        args = parser.parse_args()
        
        # Standalone logic (similar to original)
        # ...
        # For now, we'll just adapt the workflow call.
        pass
    else:
        # Workflow mode
        # mosaic_path is the fused belief map
        # output_name is the prefix for outputs
        
        # We need to define feature paths.
        # In the workflow, we just produced the mosaic (fused belief).
        # The classification step usually takes multiple features to classify anomalies.
        # But the fusion step already combined them into a "belief" map?
        # Or is classification supposed to run ON the fused map + others?
        
        # Looking at the original main():
        # It used: gravity_residual, fused_belief, poisson_corr, insar_mosaic
        
        # In the current workflow, we might not have all of these.
        # We definitely have the mosaic_path (fused belief).
        # We have gravity residual.
        # We might not have poisson_corr (Phase 3 seems skipped/not in workflow list).
        # We might not have insar_mosaic.
        
        # CRITICAL FIX: Construct feature paths relative to workflow-specific output directory
        # The workflow uses output_name with "_data" suffix for processed data
        # output_name is like: Path("data/outputs/california_full_multisource")
        # Processed data is in: Path("data/outputs/california_full_multisource_data/processed/...")
        
        feature_paths = []
        
        # 1. Fused Belief (The mosaic) - this is the primary feature
        if mosaic_path and Path(mosaic_path).exists():
            feature_paths.append(str(mosaic_path))
            
        # 2. Gravity Residual - look in workflow-specific directory
        # Try workflow-specific path first (with _data suffix)
        workflow_data_dir = Path(str(output_name) + "_data") / "processed"
        grav_res = workflow_data_dir / "gravity" / "gravity_residual_wavelet.tif"
        if not grav_res.exists():
            # Fallback to global DATA_DIR if workflow-specific doesn't exist
            grav_res = DATA_DIR / "processed" / "gravity" / "gravity_residual_wavelet.tif"
        if grav_res.exists():
            feature_paths.append(str(grav_res))
            
        # 3. PINN Density Model (if available separately, though it might be in fusion)
        pinn_dens = OUTPUTS_DIR / f"{output_name.name}_density_model.tif"
        if pinn_dens.exists():
            feature_paths.append(str(pinn_dens))
            
        if not feature_paths:
            logger.error("No features found for classification.")
            return

        output_prob_path = str(output_name.with_suffix(".probability.tif"))
        
        logger.info(f"Classifying using {len(feature_paths)} features: {feature_paths}")
        
        # Train
        X_train, imputer = prepare_training_data(feature_paths)
        scaler, ocsvm, iforest = train_models(X_train)
        
        # Predict
        classify_anomaly_candidates(feature_paths, output_prob_path, scaler, ocsvm, iforest, imputer, mode=mode)
        return

    # Define inputs based on previous phases
    # 1. Gravity Residual (Phase 1)
    gravity_residual = OUTPUTS_DIR / "gravity_residual.tif"
    if not gravity_residual.exists():
         gravity_residual = PROCESSED_DIR / "gravity" / "gravity_residual.tif"
    if not gravity_residual.exists():
        gravity_residual = OUTPUTS_DIR / "gravity" / "xgm2019e_gravdist_box_mgal.tif"

    # 2. Fused Belief (Phase 4)
    fused_belief = OUTPUTS_DIR / "fusion" / "fused_belief.tif"
    
    # 3. Poisson Correlation (Phase 3)
    poisson_corr = OUTPUTS_DIR / "poisson_correlation.tif"
    
    # 4. InSAR Mosaic (Phase 2) - using Winter Coherence as a feature
    insar_mosaic = PROCESSED_DIR / "insar" / "mosaics" / "usa_winter_vv_COH12.vrt"

    feature_paths = [
        gravity_residual,
        fused_belief,
        poisson_corr,
        insar_mosaic
    ]
    
    # Filter existing
    valid_paths = [str(p) for p in feature_paths if p.exists()]
    
    if len(valid_paths) < len(feature_paths):
        print("Warning: Some input files are missing:")
        for p in feature_paths:
            if not p.exists():
                print(f"  - {p}")
        if len(valid_paths) == 0:
            print("Error: No input files found.")
            return

    print(f"Classifying using {len(valid_paths)} features: {valid_paths}")
    
    # Train
    X_train, imputer = prepare_training_data(valid_paths)
    scaler, ocsvm, iforest = train_models(X_train)
    
    # Predict
    classify_anomaly_candidates(valid_paths, args.output, scaler, ocsvm, iforest, imputer, mode=args.mode)

if __name__ == "__main__":
    main()