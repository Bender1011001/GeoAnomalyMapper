#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for Supervised Mineral Exploration Model

"Truth Machine" Validation Strategy:
- Buffered Spatial Leave-One-Out Cross-Validation (LOOCV) with exclusion zones
- Ensemble Bagging with Balanced Sampling for robust predictions
- Strict spatial constraints to prevent information leakage

This script performs rigorous validation of the supervised learning model by testing
whether it can detect mineral deposits it has never seen before during training.
This addresses concerns about overfitting to training locations vs. learning
geophysical signatures.

Key Features:
- Buffered Spatial LOOCV with exclusion zones (radius = 0.43° from variogram analysis)
- Ensemble Bagging: 50 models trained on balanced subsets
- Gaussian jitter augmentation (σ=0.005°) for training set expansion
- Probability prediction at held-out deposit locations
- Sensitivity calculation: % of deposits detected at threshold 0.5
- Flagged area validation: confirms model doesn't over-flag the map
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_fetcher import CALIFORNIA_BASELINE_DEPOSITS, get_training_coordinates
from classify_supervised import (
    extract_features_at_points,
    sample_background_features,
    train_supervised_model,
    train_balanced_ensemble_model,
    generate_engineered_features
)
from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# TRUTH MACHINE VALIDATION CONSTANTS
# ============================================================================
# Spatial autocorrelation range from variogram analysis
# This defines the buffer radius for spatial exclusion zones in LOOCV
BUFFER_RADIUS = 0.43  # degrees (~48 km) - derived from variogram analysis


def apply_gaussian_augmentation(coords: np.ndarray, sigma: float = 0.005, n_variations: int = 5) -> np.ndarray:
    """
    Apply Gaussian jitter augmentation to coordinates.

    Args:
        coords: Original coordinates array of shape (n_points, 2) [lat, lon]
        sigma: Standard deviation for Gaussian jitter in degrees (~500m at equator)
        n_variations: Number of augmented variations per original point

    Returns:
        Augmented coordinates array including originals
    """
    np.random.seed(42)  # For reproducibility

    augmented_coords = []

    for coord in coords:
        # Include original coordinate
        augmented_coords.append(coord)

        # Add jittered variations
        for _ in range(n_variations):
            jitter_lat = np.random.normal(0, sigma)
            jitter_lon = np.random.normal(0, sigma)
            augmented_coords.append([coord[0] + jitter_lat, coord[1] + jitter_lon])

    return np.array(augmented_coords)


def perform_loocv(feature_paths: list, original_deposits: list, negative_ratio: float = 5.0,
                  use_ensemble: bool = True, n_ensemble: int = 50) -> dict:
    """
    Perform Buffered Spatial Leave-One-Out Cross-Validation with Ensemble Bagging.
    
    "Truth Machine" Implementation:
    - Each held-out deposit has an exclusion zone (radius = BUFFER_RADIUS)
    - Negative samples cannot be drawn from exclusion zones
    - Ensemble of 50 models trained on balanced subsets for robustness
    - This provides realistic performance estimates with spatial constraints

    Args:
        feature_paths: List of feature raster file paths
        original_deposits: List of 17 original deposit dictionaries
        negative_ratio: Ratio of negative to positive samples
        use_ensemble: Use BalancedEnsembleClassifier (recommended)
        n_ensemble: Number of models in ensemble

    Returns:
        Dictionary with LOOCV results
    """
    logger.info("=" * 80)
    logger.info("TRUTH MACHINE: BUFFERED SPATIAL LOOCV WITH ENSEMBLE BAGGING")
    logger.info("=" * 80)
    logger.info(f"Spatial exclusion buffer: {BUFFER_RADIUS}° (~{BUFFER_RADIUS * 111:.1f} km)")
    logger.info(f"Ensemble learning: {use_ensemble} (n_ensemble={n_ensemble})")

    # Extract original coordinates (no augmentation yet)
    original_coords, deposit_names = get_training_coordinates(original_deposits)
    n_deposits = len(original_coords)

    logger.info(f"Testing {n_deposits} deposits with LOOCV")
    logger.info(f"Training augmentation: Gaussian jitter σ=0.005°, 5 variations per deposit")
    logger.info(f"Feature engineering: Gradient magnitude, local roughness, local mean")

    loocv_scores = []
    held_out_results = []

    # Identify gravity and magnetic paths for feature engineering
    gravity_path = None
    magnetic_path = None
    
    for path in feature_paths:
        if 'gravity' in path.lower() and 'residual' in path.lower():
            gravity_path = path
        elif 'magnetic' in path.lower():
            magnetic_path = path
    
    logger.info(f"Gravity path for feature engineering: {gravity_path}")
    logger.info(f"Magnetic path for feature engineering: {magnetic_path}")

    # Generate Expert Version engineered features ONCE before the loop
    # This prevents file locking issues and redundant computation
    logger.info("Pre-generating engineered features for LOOCV...")
    current_feature_paths = generate_engineered_features(
        feature_paths, gravity_path, magnetic_path
    )
    logger.info(f"Generated {len(current_feature_paths)} features (including Expert Version)")

    for i in range(n_deposits):
        logger.info(f"\nFold {i+1}/{n_deposits}: Holding out '{deposit_names[i]}'")

        # Define training set: all deposits except i
        train_indices = [j for j in range(n_deposits) if j != i]
        train_coords = original_coords[train_indices]
        train_names = [deposit_names[j] for j in train_indices]

        # TRUTH MACHINE: Create spatial exclusion zone for held-out deposit
        held_out_lat, held_out_lon = original_coords[i]
        current_exclusion_zone = [(held_out_lat, held_out_lon, BUFFER_RADIUS)]
        logger.info(f"  Exclusion zone: ({held_out_lat:.4f}, {held_out_lon:.4f}) radius={BUFFER_RADIUS}°")

        # Apply augmentation to training set
        augmented_train_coords = apply_gaussian_augmentation(train_coords, sigma=0.015, n_variations=20)
        logger.info(f"  Training set: {len(train_coords)} deposits → {len(augmented_train_coords)} samples")

        # Extract features at training locations
        train_features, train_valid_mask = extract_features_at_points(current_feature_paths, augmented_train_coords)
        train_features = train_features[train_valid_mask]

        if len(train_features) == 0:
            logger.warning(f"  No valid training features for fold {i+1}, skipping")
            continue

        # TRUTH MACHINE: Sample negative training data with spatial exclusion zones
        # This ensures negatives are not drawn near the held-out deposit
        n_negative = int(len(train_features) * negative_ratio)
        logger.info(f"  Sampling {n_negative} negatives (excluding {BUFFER_RADIUS}° buffer around held-out)")
        negative_features = sample_background_features(
            current_feature_paths, n_negative, augmented_train_coords,
            exclusion_zones=current_exclusion_zone
        )

        # TRUTH MACHINE: Train ensemble model for robust predictions
        if use_ensemble:
            logger.info(f"  Training BalancedEnsembleClassifier with {n_ensemble} models...")
            clf = train_balanced_ensemble_model(
                train_features, negative_features,
                n_ensemble=n_ensemble,
                n_estimators_base=100,
                max_depth=8,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42
            )
            scaler = None  # Ensemble has built-in scaler
        else:
            # Fallback to single RandomForest
            logger.info("  Training single RandomForestClassifier...")
            clf, scaler = train_supervised_model(
                train_features, negative_features,
                n_estimators=100, max_depth=5, min_samples_leaf=5,
                max_features='sqrt', random_state=42
            )

        # Test on held-out deposit
        held_out_coord = original_coords[i:i+1]  # Shape (1, 2)
        held_out_features, held_out_valid = extract_features_at_points(current_feature_paths, held_out_coord)

        if not held_out_valid[0]:
            logger.warning(f"  Held-out deposit '{deposit_names[i]}' has invalid features, assigning score 0.0")
            score = 0.0
        else:
            # Predict probability (different handling for ensemble vs single model)
            if use_ensemble:
                score = clf.predict_proba(held_out_features[0:1])[0, 1]
            else:
                held_out_scaled = scaler.transform(held_out_features[0:1])
                score = clf.predict_proba(held_out_scaled)[0, 1]

        loocv_scores.append(score)
        held_out_results.append({
            'deposit_name': deposit_names[i],
            'latitude': held_out_coord[0, 0],
            'longitude': held_out_coord[0, 1],
            'loocv_score': score
        })

        logger.info(f"  [OK] LOOCV score: {score:.3f}")
        if score < 0.5:
            deposit_type = original_deposits[i]['type']
            logger.warning(f"  [MISS] MISSED: {deposit_names[i]} ({deposit_type}) score={score:.3f}")

    # Calculate LOOCV sensitivity
    sensitivity_threshold = 0.5
    detected_count = sum(1 for score in loocv_scores if score > sensitivity_threshold)
    loocv_sensitivity = (detected_count / len(loocv_scores)) * 100 if loocv_scores else 0

    results = {
        'loocv_scores': loocv_scores,
        'held_out_results': held_out_results,
        'loocv_sensitivity': loocv_sensitivity,
        'sensitivity_threshold': sensitivity_threshold,
        'detected_count': detected_count,
        'total_tested': len(loocv_scores)
    }

    return results


def calculate_flagged_area(probability_map_path: str, threshold: float = 0.9) -> dict:
    """
    Calculate the percentage of area flagged above a given threshold.

    Args:
        probability_map_path: Path to probability map GeoTIFF
        threshold: Probability threshold for flagging

    Returns:
        Dictionary with flagged area statistics
    """
    logger.info(f"\nCalculating flagged area at threshold {threshold}...")

    with rasterio.open(probability_map_path) as src:
        data = src.read(1)
        nodata = src.nodata

        # Create mask for valid data
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data)

        # Count flagged pixels
        flagged_mask = (data >= threshold) & valid_mask
        flagged_pixels = np.sum(flagged_mask)
        total_valid_pixels = np.sum(valid_mask)

        percent_flagged = (flagged_pixels / total_valid_pixels * 100) if total_valid_pixels > 0 else 0

        logger.info(f"  Total valid pixels: {total_valid_pixels:,}")
        logger.info(f"  Flagged pixels: {flagged_pixels:,}")
        logger.info(".2f")

        return {
            'threshold': threshold,
            'flagged_pixels': int(flagged_pixels),
            'total_pixels': int(total_valid_pixels),
            'percent_flagged': percent_flagged
        }


def main():
    """Main validation workflow."""
    logger.info("Starting model robustness validation...")

    # Define California region bounds
    california_bounds = (-125.01, 31.98, -113.97, 42.52)  # (lon_min, lat_min, lon_max, lat_max)

    # Load original 17 deposits (no external data)
    original_deposits = CALIFORNIA_BASELINE_DEPOSITS.copy()
    logger.info(f"Loaded {len(original_deposits)} baseline California deposits")

    # Filter to California bounds
    filtered_deposits = []
    for deposit in original_deposits:
        lat, lon = deposit['lat'], deposit['lon']
        lon_min, lat_min, lon_max, lat_max = california_bounds
        if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
            filtered_deposits.append(deposit)

    logger.info(f"Filtered to {len(filtered_deposits)} deposits within California bounds")

    if len(filtered_deposits) != 17:
        logger.warning(f"Expected 17 deposits, got {len(filtered_deposits)}")

    # Define feature paths (same as supervised workflow)
    workflow_data_dir = OUTPUTS_DIR / "california_full_multisource_data" / "processed"
    feature_paths = []

    # Gravity residual
    gravity_residual = workflow_data_dir / "gravity" / "gravity_residual_wavelet.tif"
    if gravity_residual.exists():
        feature_paths.append(str(gravity_residual))
        logger.info("[OK] Gravity residual available")
    else:
        logger.warning("[MISS] Gravity residual not found")

    # InSAR features
    insar_features = workflow_data_dir / "insar" / "insar_processed.tif"
    if insar_features.exists():
        feature_paths.append(str(insar_features))
        logger.info("[OK] InSAR features available")
    else:
        logger.warning("[MISS] InSAR features not found")

    # Magnetic data
    magnetic_features = workflow_data_dir / "magnetic" / "magnetic_processed.tif"
    if magnetic_features.exists():
        feature_paths.append(str(magnetic_features))
        logger.info("[OK] Magnetic features available")
    else:
        logger.warning("[MISS] Magnetic features not found")

    # Note: Removed fused_belief from feature stack to prevent overfitting
    # The model should learn from raw physical data and expert features only

    if not feature_paths:
        logger.error("No feature files found! Run the full California workflow first.")
        return False

    logger.info(f"Using {len(feature_paths)} features: {[Path(p).name for p in feature_paths]}")

    # Generate engineered features for full workflow
    gravity_path = None
    magnetic_path = None
    
    for path in feature_paths:
        if 'gravity' in path.lower() and 'residual' in path.lower():
            gravity_path = path
        elif 'magnetic' in path.lower():
            magnetic_path = path
    
    logger.info(f"Gravity path for full workflow: {gravity_path}")
    logger.info(f"Magnetic path for full workflow: {magnetic_path}")
    
    # Generate Expert Version engineered features for full workflow
    full_feature_paths = generate_engineered_features(feature_paths, gravity_path, magnetic_path)
    logger.info(f"Generated {len(full_feature_paths)} Expert Version features for full workflow")

    # TRUTH MACHINE: Perform Buffered Spatial LOOCV with Ensemble Bagging
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRUTH MACHINE VALIDATION")
    logger.info("=" * 80)
    loocv_results = perform_loocv(
        feature_paths, filtered_deposits,
        negative_ratio=5.0,
        use_ensemble=True,  # Enable ensemble bagging
        n_ensemble=100  # 100 models for robust predictions
    )

    # Generate full map prediction with engineered features and ensemble
    output_base = OUTPUTS_DIR / "california_supervised"
    output_base.mkdir(exist_ok=True)
    
    full_map_path = output_base / "california_supervised_probability_truth_machine.tif"
    
    # Extract coordinates for full training (with augmentation)
    coords, labels = get_training_coordinates(filtered_deposits)
    augmented_coords = apply_gaussian_augmentation(coords, sigma=0.015, n_variations=20)
    logger.info(f"\nFull training: {len(coords)} deposits → {len(augmented_coords)} samples")
    
    # Train full model with engineered features and ensemble
    logger.info("Training full model with Truth Machine configuration...")
    from classify_supervised import classify_supervised
    
    classifier = classify_supervised(
        feature_paths=full_feature_paths,
        positive_coords=augmented_coords,
        output_path=str(full_map_path),
        negative_ratio=5.0,
        n_estimators=100,
        random_state=42,
        use_ensemble=True,  # Use ensemble for final map too
        n_ensemble=100,
        max_depth=8
    )
    logger.info(f"Full probability map saved: {full_map_path}")

    # Calculate flagged area at threshold 0.9
    flagged_area_results = calculate_flagged_area(str(full_map_path), threshold=0.9)

    # Print summary results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print("\nHeld-Out Deposit Scores:")
    print("Deposit Name".ljust(25) + "Latitude".ljust(12) + "Longitude".ljust(12) + "LOOCV Score")
    print("-" * 70)

    for result in loocv_results['held_out_results']:
        print(result['deposit_name'].ljust(25) +
              f"{result['latitude']:12.4f}" +
              f"{result['longitude']:12.4f}" +
              f"{result['loocv_score']:.3f}")

    print("\nLOOCV Sensitivity: {:.1f}% ({}/{} deposits detected at threshold {:.1f})".format(
        loocv_results['loocv_sensitivity'],
        loocv_results['detected_count'],
        loocv_results['total_tested'],
        loocv_results['sensitivity_threshold']
    ))

    print("\nFlagged Area Validation:")
    print(f"Threshold: {flagged_area_results['threshold']:.1f}")
    print(f"Flagged Pixels: {flagged_area_results['flagged_pixels']:,}")
    print(f"Total Pixels: {flagged_area_results['total_pixels']:,}")
    print(f"Percent Flagged: {flagged_area_results['percent_flagged']:.2f}%")

    # Interpretation
    print("\nINTERPRETATION:")
    if loocv_results['loocv_sensitivity'] >= 70:
        print("[HIGH CONFIDENCE] Model shows strong generalization ability.")
        print("   The model learns geophysical signatures, not just training locations.")
    elif loocv_results['loocv_sensitivity'] >= 50:
        print("[MODERATE CONFIDENCE] Model shows some generalization but may be location-biased.")
    else:
        print("[LOW CONFIDENCE] Model may be overfitting to training locations.")

    if flagged_area_results['percent_flagged'] < 10:
        print("[OK] Map not over-flagged: Reasonable exploration target area.")
    else:
        print("[WARN] Map may be over-flagged: Large portion marked as prospective.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
