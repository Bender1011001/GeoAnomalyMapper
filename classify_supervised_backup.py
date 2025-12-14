#!/usr/bin/env python3
"""
Supervised Classification for GeoAnomalyMapper v2.1

Implements supervised learning using Random Forest to classify mineral exploration
anomalies. Uses known deposit locations as positive training samples and randomly
sampled background pixels as negative samples.

Key Features:
- Extracts feature values at positive training coordinates
- Randomly samples background for negative training data
- Trains RandomForestClassifier with probability outputs
- Outputs calibrated probability maps as GeoTIFF
- Memory-efficient windowed prediction for large rasters
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from project_paths import OUTPUTS_DIR
from utils.feature_engineering import stack_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features_at_points(feature_paths: List[str],
                              coordinates: np.ndarray,
                              buffer_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature values at specific coordinate locations.

    Args:
        feature_paths: List of paths to feature raster files
        coordinates: Array of shape (n_points, 2) with [lat, lon]
        buffer_size: Size of buffer window around each point (pixels)

    Returns:
        Tuple of (features, valid_mask)
        features: shape (n_points, n_features) or (n_points, n_features, window_size, window_size)
        valid_mask: boolean array indicating which points had valid extractions
    """
    if not feature_paths:
        raise ValueError("At least one feature path required")

    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n_points, 2) with [lat, lon]")

    logger.info(f"Extracting features at {len(coordinates)} coordinate locations")

    # Use first raster as reference for coordinate transformation
    with rasterio.open(feature_paths[0]) as src_ref:
        ref_transform = src_ref.transform
        ref_crs = src_ref.crs

    # Transform coordinates to pixel indices
    pixel_coords = []
    valid_indices = []

    for i, (lat, lon) in enumerate(coordinates):
        try:
            # Convert lat/lon to pixel coordinates
            row, col = src_ref.index(lon, lat)  # Note: rasterio uses (lon, lat) for index

            # Check bounds
            if (0 <= row < src_ref.height) and (0 <= col < src_ref.width):
                pixel_coords.append((row, col))
                valid_indices.append(i)
            else:
                logger.warning(f"Coordinate {i} ({lat:.4f}, {lon:.4f}) is outside raster bounds")

        except Exception as e:
            logger.warning(f"Failed to transform coordinate {i}: {e}")
            continue

    if not pixel_coords:
        raise ValueError("No valid coordinates found within raster bounds")

    logger.info(f"Found {len(pixel_coords)} valid coordinate locations")

    # Extract features at each valid location
    n_features = len(feature_paths)
    n_valid_points = len(pixel_coords)

    if buffer_size == 1:
        # Single pixel extraction
        features = np.zeros((n_valid_points, n_features), dtype=np.float32)
    else:
        # Window extraction
        window_size = 2 * buffer_size + 1
        features = np.zeros((n_valid_points, n_features, window_size, window_size), dtype=np.float32)

    valid_mask = np.zeros(len(coordinates), dtype=bool)

    for feat_idx, fpath in enumerate(feature_paths):
        with rasterio.open(fpath) as src:
            for point_idx, (row, col) in enumerate(pixel_coords):
                try:
                    if buffer_size == 1:
                        # Single pixel
                        window = Window(col, row, 1, 1)
                        data = src.read(1, window=window)
                        if data.size > 0 and not np.isnan(data).all():
                            features[point_idx, feat_idx] = data[0, 0]
                        else:
                            features[point_idx, feat_idx] = np.nan
                    else:
                        # Window around point
                        window_size = 2 * buffer_size + 1
                        window = Window(
                            col - buffer_size, row - buffer_size,
                            window_size, window_size
                        )

                        # Ensure window is within bounds
                        window = window.intersection(Window(0, 0, src.width, src.height))

                        data = src.read(1, window=window)
                        if data.size > 0:
                            # Pad if necessary to maintain window size
                            padded_data = np.full((window_size, window_size), np.nan, dtype=np.float32)
                            dest_window = Window(
                                max(0, buffer_size - (col - window.col_off)),
                                max(0, buffer_size - (row - window.row_off)),
                                window.width, window.height
                            )
                            padded_data[dest_window.row_off:dest_window.row_off + window.height,
                                       dest_window.col_off:dest_window.col_off + window.width] = data
                            features[point_idx, feat_idx] = padded_data
                        else:
                            features[point_idx, feat_idx] = np.full((window_size, window_size), np.nan, dtype=np.float32)

                except Exception as e:
                    logger.warning(f"Failed to extract feature {feat_idx} at point {point_idx}: {e}")
                    if buffer_size == 1:
                        features[point_idx, feat_idx] = np.nan
                    else:
                        features[point_idx, feat_idx] = np.full((window_size, window_size), np.nan, dtype=np.float32)

    # Set valid mask
    valid_mask[np.array(valid_indices)] = True

    # Flatten window features if needed for training
    if buffer_size > 1:
        window_size = 2 * buffer_size + 1
        features_flat = features.reshape(n_valid_points, n_features * window_size * window_size)
        return features_flat, valid_mask

    return features, valid_mask


def sample_background_features(feature_paths: List[str],
                              n_samples: int = 10000,
                              exclude_coords: Optional[np.ndarray] = None,
                              exclusion_radius: float = 0.01) -> np.ndarray:
    """
    Randomly sample background features for negative training data.

    Args:
        feature_paths: List of paths to feature raster files
        n_samples: Number of background samples to generate
        exclude_coords: Coordinates to exclude (positive samples) [lat, lon]
        exclusion_radius: Radius around positive samples to exclude (degrees)

    Returns:
        Background feature array of shape (n_samples, n_features)
    """
    if not feature_paths:
        raise ValueError("At least one feature path required")

    logger.info(f"Sampling {n_samples} background locations for negative training")

    # Use first raster as reference for bounds and sampling
    with rasterio.open(feature_paths[0]) as src_ref:
        bounds = src_ref.bounds
        height, width = src_ref.height, src_ref.width

    # Convert exclusion coordinates to pixel coordinates if provided
    exclude_pixels = set()
    if exclude_coords is not None:
        with rasterio.open(feature_paths[0]) as src_ref:
            for lat, lon in exclude_coords:
                try:
                    row, col = src_ref.index(lon, lat)  # Note: rasterio uses (lon, lat)
                    # Add minimal exclusion zone around positive samples
                    radius_pixels = min(2, int(exclusion_radius * 111000 / src_ref.res[0]))  # Max 2 pixels
                    for dr in range(-radius_pixels, radius_pixels + 1):
                        for dc in range(-radius_pixels, radius_pixels + 1):
                            if (dr**2 + dc**2) <= radius_pixels**2:
                                r, c = row + dr, col + dc
                                if 0 <= r < height and 0 <= c < width:
                                    exclude_pixels.add((r, c))
                except:
                    continue

    logger.info(f"Excluding {len(exclude_pixels)} pixels around positive samples")

    # Randomly sample pixel locations
    sampled_features = []
    attempts = 0
    max_attempts = n_samples * 10  # Allow some retries

    while len(sampled_features) < n_samples and attempts < max_attempts:
        # Random pixel location
        row = np.random.randint(0, height)
        col = np.random.randint(0, width)

        # Skip if in exclusion zone
        if (row, col) in exclude_pixels:
            attempts += 1
            continue

        # Extract features at this location
        try:
            # Convert pixel coordinates back to lat/lon
            lon, lat = src_ref.xy(row, col)
            point_features, valid = extract_features_at_points(
                feature_paths, np.array([[lat, lon]])
            )

            # Be more lenient with NaN values - allow some NaN as long as we have some valid data
            if valid[0] and not np.isnan(point_features[0]).all():
                # Replace NaN with 0 for now (will be handled by imputer later)
                features_clean = np.nan_to_num(point_features[0], nan=0.0)
                sampled_features.append(features_clean)
            else:
                attempts += 1
                continue

        except Exception as e:
            attempts += 1
            continue

    if len(sampled_features) < n_samples:
        logger.warning(f"Only sampled {len(sampled_features)}/{n_samples} background points")
        if len(sampled_features) == 0:
            raise ValueError("Failed to sample any valid background points")

    return np.array(sampled_features)


def train_supervised_model(positive_features: np.ndarray,
                          negative_features: np.ndarray,
                          n_estimators: int = 100,
                          max_depth: int = 5,
                          min_samples_leaf: int = 5,
                          max_features: str = 'sqrt',
                          random_state: int = 42) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train Random Forest classifier on positive and negative samples.

    Args:
        positive_features: Feature array for positive samples
        negative_features: Feature array for negative samples
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees (regularization)
        min_samples_leaf: Minimum number of samples required to be at a leaf node (regularization)
        max_features: Number of features to consider when looking for the best split
        random_state: Random state for reproducibility

    Returns:
        Tuple of (trained_classifier, feature_scaler)
    """
    logger.info("Training Random Forest classifier")

    # Combine positive and negative samples
    X = np.vstack([positive_features, negative_features])
    y = np.hstack([np.ones(len(positive_features)), np.zeros(len(negative_features))])

    # Handle NaN values more gracefully - replace with 0 and keep all samples
    X_clean = np.nan_to_num(X, nan=0.0)
    y_clean = y

    logger.info(f"Training on {len(X_clean)} samples ({np.sum(y_clean)} positive, {len(y_clean) - np.sum(y_clean)} negative)")

    # Ensure we have both classes
    if np.sum(y_clean) == 0:
        raise ValueError("No positive samples available for training")
    if np.sum(y_clean) == len(y_clean):
        raise ValueError("No negative samples available for training")

    if len(X_clean) < 100:
        logger.warning("Very few training samples - model may not generalize well")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle class imbalance
    )

    clf.fit(X_scaled, y_clean)

    # Quick validation on training set
    train_probs = clf.predict_proba(X_scaled)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)

    logger.info("Training set performance:")
    logger.info(f"Accuracy: {np.mean(train_preds == y_clean):.3f}")
    logger.info(f"Positive class recall: {np.mean(train_preds[y_clean == 1] == 1):.3f}")
    logger.info(f"Negative class recall: {np.mean(train_preds[y_clean == 0] == 0):.3f}")

    return clf, scaler


def predict_probability_map(feature_paths: List[str],
                           classifier: RandomForestClassifier,
                           scaler: StandardScaler,
                           output_path: str,
                           block_size: int = 2048) -> None:
    """
    Predict probability map across entire raster extent.

    Args:
        feature_paths: List of feature raster paths
        classifier: Trained RandomForestClassifier
        scaler: Trained StandardScaler for features
        output_path: Path to save output GeoTIFF
        block_size: Size of processing blocks for memory efficiency
    """
    logger.info(f"Generating probability map: {output_path}")

    # Use first feature as reference grid
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

    with rasterio.open(output_path, 'w', **profile) as dst:
        for row_off in range(0, height, block_size):
            for col_off in range(0, width, block_size):
                window_width = min(block_size, width - col_off)
                window_height = min(block_size, height - row_off)
                window = Window(col_off, row_off, window_width, window_height)

                logger.info(f"Processing block: {row_off}-{row_off+window_height}, {col_off}-{col_off+window_width}")

                # Read all features for this window
                X_window_list = []

                for fpath in feature_paths:
                    with rasterio.open(fpath) as src:
                        # Reproject window if needed
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

                # Stack features
                X_window_flat = np.column_stack(X_window_list)

                # Handle NaN values
                X_window_imputed = np.nan_to_num(X_window_flat, nan=0.0)  # Simple imputation

                # Scale features
                X_window_scaled = scaler.transform(X_window_imputed)

                # Predict probabilities
                prob_flat = classifier.predict_proba(X_window_scaled)[:, 1]

                # Reshape to window
                prob_window = prob_flat.reshape(window_height, window_width)

                # Mask areas with all-NaN inputs
                all_nan_mask = np.all(np.isnan(X_window_flat), axis=1)
                prob_window[all_nan_mask.reshape(window_height, window_width)] = np.nan

                # Write block
                dst.write(prob_window, 1, window=window)

    logger.info("Probability map generation complete")


def classify_supervised(feature_paths: List[str],
                        positive_coords: np.ndarray,
                        output_path: str,
                        negative_ratio: float = 10.0,
                        n_estimators: int = 100,
                        max_depth: int = 5,
                        min_samples_leaf: int = 5,
                        max_features: str = 'sqrt',
                        random_state: int = 42,
                        gravity_path: Optional[str] = None,
                        magnetic_path: Optional[str] = None) -> RandomForestClassifier:
    """
    Main supervised classification workflow.

    Args:
        feature_paths: List of feature raster file paths
        positive_coords: Array of positive training coordinates [lat, lon]
        output_path: Path to save probability map GeoTIFF
        negative_ratio: Ratio of negative to positive samples
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of the trees (regularization)
        min_samples_leaf: Minimum number of samples required to be at a leaf node (regularization)
        max_features: Number of features to consider when looking for the best split
        random_state: Random seed for reproducibility
        gravity_path: Path to gravity raster for feature engineering
        magnetic_path: Path to magnetic raster for feature engineering

    Returns:
        Trained RandomForestClassifier
    """
    logger.info("Starting supervised classification workflow")
    logger.info(f"Features: {len(feature_paths)} rasters")
    logger.info(f"Positive samples: {len(positive_coords)} locations")
    logger.info(f"Negative ratio: {negative_ratio}:1")

    # Generate engineered features if base rasters are available
    if gravity_path or magnetic_path:
        logger.info("Generating engineered features from base rasters...")
        feature_paths = generate_engineered_features(
            feature_paths, gravity_path, magnetic_path
        )
        logger.info(f"Updated feature paths: {len(feature_paths)} features")

    # Extract positive features
    logger.info("Extracting features at positive training locations...")
    positive_features, pos_valid_mask = extract_features_at_points(feature_paths, positive_coords)
    positive_features = positive_features[pos_valid_mask]

    if len(positive_features) == 0:
        raise ValueError("No valid positive training samples found")

    logger.info(f"Valid positive samples: {len(positive_features)}")

    # Sample negative features
    n_negative = int(len(positive_features) * negative_ratio)
    logger.info(f"Sampling {n_negative} negative training locations...")
    negative_features = sample_background_features(
        feature_paths, n_negative, positive_coords
    )

    # Train model
    classifier, scaler = train_supervised_model(
        positive_features, negative_features, n_estimators, max_depth, min_samples_leaf, max_features, random_state
    )

    # Generate probability map
    predict_probability_map(feature_paths, classifier, scaler, output_path)

    return classifier


def generate_engineered_features(feature_paths: List[str],
                                 gravity_path: Optional[str] = None,
                                 magnetic_path: Optional[str] = None) -> List[str]:
    """
    Generate engineered features from base rasters using the Expert Version
    and return updated feature paths.
    
    Args:
        feature_paths: List of existing feature raster paths
        gravity_path: Path to gravity raster for feature engineering
        magnetic_path: Path to magnetic raster for feature engineering
        
    Returns:
        Updated list of feature paths including engineered features
    """
    logger.info("Generating Expert Version engineered features...")
    
    # Check if we have gravity and magnetic data
    gravity_data = None
    magnetic_data = None
    
    # Load gravity data if available
    if gravity_path and Path(gravity_path).exists():
        with rasterio.open(gravity_path) as src:
            gravity_data = src.read(1)
        logger.info("  ✅ Loaded gravity data for feature engineering")
    
    # Load magnetic data if available
    if magnetic_path and Path(magnetic_path).exists():
        with rasterio.open(magnetic_path) as src:
            magnetic_data = src.read(1)
        logger.info("  ✅ Loaded magnetic data for feature engineering")
    
    # Generate engineered features if we have base data
    if gravity_data is not None or magnetic_data is not None:
        # Stack features using the new Expert Version with advanced features
        if gravity_data is not None and magnetic_data is not None:
            stacked_features, feature_names = stack_features(
                gravity_data, magnetic_data,
                roughness_window=3,
                mean_window=5
            )
        elif gravity_data is not None:
            stacked_features, feature_names = stack_features(
                gravity_data, None,
                roughness_window=3,
                mean_window=5
            )
        else:
            stacked_features, feature_names = stack_features(
                magnetic_data, None,
                roughness_window=3,
                mean_window=5
            )
        
        # Save engineered features as temporary files
        engineered_paths = []
        ref_path = gravity_path if gravity_path else magnetic_path
        
        with rasterio.open(ref_path) as src_ref:
            profile = src_ref.profile.copy()
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'nodata': np.nan,
                'compress': 'lzw'
            })
            
            for i, feature_name in enumerate(feature_names):
                temp_path = Path(ref_path).parent / f"engineered_{feature_name}.tif"
                engineered_paths.append(str(temp_path))
                
                with rasterio.open(str(temp_path), 'w', **profile) as dst:
                    dst.write(stacked_features[i], 1)
        
        logger.info(f"  ✅ Generated {len(engineered_paths)} Expert Version features:")
        for name in feature_names:
            logger.info(f"    - {name}")
        return feature_paths + engineered_paths
    
    logger.info("  ⚠️  No base rasters available for feature engineering")
    return feature_paths


def main():
    parser = argparse.ArgumentParser(description="Supervised Classification for Mineral Exploration")
    parser.add_argument("--features", nargs='+', required=True,
                       help="Paths to feature raster files")
    parser.add_argument("--positives", required=True,
                       help="Path to CSV file with positive training coordinates (lat,lon columns)")
    parser.add_argument("--output", required=True,
                       help="Output path for probability GeoTIFF")
    parser.add_argument("--negative-ratio", type=float, default=10.0,
                       help="Ratio of negative to positive samples (default: 10.0)")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of Random Forest estimators (default: 100)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility (default: 42)")

    args = parser.parse_args()

    # Load positive coordinates from CSV
    if not Path(args.positives).exists():
        logger.error(f"Positive coordinates file not found: {args.positives}")
        return

    try:
        coords_df = pd.read_csv(args.positives)
        if 'lat' not in coords_df.columns or 'lon' not in coords_df.columns:
            logger.error("CSV must contain 'lat' and 'lon' columns")
            return

        positive_coords = coords_df[['lat', 'lon']].values
        logger.info(f"Loaded {len(positive_coords)} positive training coordinates")

    except Exception as e:
        logger.error(f"Failed to load positive coordinates: {e}")
        return

    # Validate feature paths
    valid_features = []
    for fpath in args.features:
        if Path(fpath).exists():
            valid_features.append(fpath)
        else:
            logger.warning(f"Feature file not found: {fpath}")

    if not valid_features:
        logger.error("No valid feature files found")
        return

    # Run classification
    try:
        classifier = classify_supervised(
            valid_features, positive_coords, args.output,
            args.negative_ratio, args.n_estimators, args.random_state
        )
        logger.info("Supervised classification completed successfully")

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise


if __name__ == "__main__":
    main()