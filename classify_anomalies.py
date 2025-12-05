#!/usr/bin/env python3
"""
Phase 5: Anomaly Classification for GeoAnomalyMapper v2.0.

Implements unsupervised anomaly detection using One-Class SVM (OC-SVM) to model
"normal" geology manifold and Isolation Forest (IF) for outlier ranking. Combines
scores into a 0-1 probability map where high values indicate DUMB candidates.

Key Features:
- Aligns multiple feature rasters to a common grid (first raster as reference).
- Handles NaNs via mean imputation per feature.
- Samples background pixels for training (default: 100k pixels).
- Normalizes combined anomaly scores to [0,1] probability.
- Outputs GeoTIFF with matching CRS/transform.

Dependencies:
- scikit-learn >=1.5.0
- numpy
- rasterio

Example Usage:
```python
feature_paths = [
    "data/processed/gravity_residual.tif",
    "data/processed/fused_belief_reinforced.tif",
    "data/processed/poisson_correlation.tif",
    "data/processed/insar_artificiality.tif"
]
classify_dumb_candidates(feature_paths, "data/processed/dumb_probability_v2.tif")
```
"""

from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def prepare_feature_stack(
    feature_paths: List[str]
) -> Tuple[np.ndarray, int, int, rasterio.Affine, str]:
    """
    Load and align feature rasters to the grid of the first raster.
    Stack into (N_pixels, N_features) array. Impute NaNs with feature-wise mean.

    Aligns all rasters via reproject to ensure consistent shape/CRS/transform.
    Uses bilinear resampling for continuous features.

    Args:
        feature_paths: List of paths to input feature rasters.

    Returns:
        Tuple of (X: stacked features (n_pixels, n_features),
                  height, width, transform, crs)

    Raises:
        ValueError: If no feature paths provided or files not found.
        RuntimeError: If reproject fails.

    Examples
    --------
    >>> feature_paths = ["gravity.tif", "poisson.tif"]
    >>> X, h, w, transform, crs = prepare_feature_stack(feature_paths)
    >>> print(X.shape)  # e.g., (1000000, 2)
    """
    if not feature_paths:
        raise ValueError("At least one feature path required.")

    # Load reference raster
    with rasterio.open(feature_paths[0]) as src_ref:
        profile_ref = src_ref.profile
        height, width = profile_ref["height"], profile_ref["width"]
        transform = src_ref.transform
        crs = src_ref.crs
        data_ref = src_ref.read(1, masked=True).filled(np.nan).astype(np.float32)

    # Impute NaNs in reference
    ref_mean = np.nanmean(data_ref)
    if np.isnan(ref_mean):
        ref_mean = 0.0
    data_ref[np.isnan(data_ref)] = ref_mean
    stack = [data_ref.flatten()]

    # Load and align other features
    for fpath in feature_paths[1:]:
        dst_data = np.empty((height, width), dtype=np.float32)
        with rasterio.open(fpath) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        # Impute NaNs
        feat_mean = np.nanmean(dst_data)
        if np.isnan(feat_mean):
            feat_mean = 0.0
        dst_data[np.isnan(dst_data)] = feat_mean
        stack.append(dst_data.flatten())

    X = np.column_stack(stack)
    return X, height, width, transform, crs


def train_models(
    X: np.ndarray, sample_size: int = 100000
) -> Tuple[StandardScaler, OneClassSVM, IsolationForest]:
    """
    Train OC-SVM and Isolation Forest on subsampled "background" data.

    Samples randomly for training to model normal geology (assumes anomalies rare).
    Uses StandardScaler for feature normalization.

    Args:
        X: Feature stack (n_pixels, n_features).
        sample_size: Max pixels to sample for training (default: 100k).

    Returns:
        Tuple of (scaler, ocsvm_model, iforest_model)

    Examples
    --------
    >>> scaler, ocsvm, iforest = train_models(X)
    """
    n_samples = min(sample_size, X.shape[0])
    train_idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_train = X[train_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    ocsvm = OneClassSVM(
        nu=0.05, kernel="rbf", gamma="scale"
    ).fit(X_train_scaled)

    iforest = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=100
    ).fit(X_train_scaled)

    return scaler, ocsvm, iforest


def compute_anomaly_scores(
    X: np.ndarray,
    scaler: StandardScaler,
    ocsvm: OneClassSVM,
    iforest: IsolationForest,
) -> np.ndarray:
    """
    Compute combined anomaly probability (0-1, high = DUMB candidate).

    - OC-SVM: -decision_function (high = outlier from normal manifold)
    - IF: -score_samples (high = isolated/anomalous)
    - Average + min-max normalization to [0,1]

    Args:
        X: Feature stack.
        scaler: Fitted scaler.
        ocsvm: Fitted OC-SVM.
        iforest: Fitted Isolation Forest.

    Returns:
        prob: Anomaly probabilities (n_pixels,)

    Examples
    --------
    >>> prob = compute_anomaly_scores(X, scaler, ocsvm, iforest)
    >>> print(np.max(prob))  # Should be ~1.0
    """
    X_scaled = scaler.transform(X)

    oc_scores = -ocsvm.decision_function(X_scaled)
    iso_scores = -iforest.score_samples(X_scaled)

    combined = (oc_scores + iso_scores) / 2.0

    p_min = np.min(combined)
    p_max = np.max(combined)
    prob = (combined - p_min) / (p_max - p_min + 1e-12)

    return np.clip(prob, 0.0, 1.0)


def classify_dumb_candidates(
    feature_paths_list: List[str],
    output_path: str,
):
    """
    Main entrypoint: Classify DUMB candidates from feature stack.

    Full pipeline: prepare -> train -> score -> save GeoTIFF.

    Args:
        feature_paths_list: List of feature raster paths.
        output_path: Output path for 'dumb_probability_v2.tif'.

    Examples
    --------
    >>> classify_dumb_candidates(
    ...     ["gravity_residual.tif", "fused_belief_reinforced.tif"], "output.tif"
    ... )
    Phase 5 output: output.tif
    """
    X, height, width, transform, crs = prepare_feature_stack(feature_paths_list)

    scaler, ocsvm, iforest = train_models(X)

    prob_flat = compute_anomaly_scores(X, scaler, ocsvm, iforest)

    prob_image = prob_flat.reshape(height, width).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "BIGTIFF": "YES",
        "nodata": np.nan,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prob_image, 1)

    print(f"Phase 5 output: {output_path}")