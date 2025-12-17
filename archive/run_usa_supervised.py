#!/usr/bin/env python3
"""
USA-Scale Supervised Learning Workflow Runner
Training Random Forest on 257,709 USA mineral deposits with 20-core parallelization
"""

import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np

from classify_supervised_optimized import classify_supervised
from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Execute USA-scale supervised learning workflow."""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("USA-SCALE SUPERVISED LEARNING WORKFLOW")
    logger.info("Training Random Forest on 257,709 USA mineral deposits")
    logger.info("=" * 80)
    
    # Define paths
    output_dir = OUTPUTS_DIR / "usa_supervised"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Input data
    usa_deposits_csv = Path("usa_deposits.csv")
    gravity_mosaic = output_dir / "usa_gravity_mosaic.tif"
    
    # Pre-generated engineered features
    engineered_features = [
        output_dir / "engineered_gravity.tif",
        output_dir / "engineered_gravity_gradient.tif",
        output_dir / "engineered_gravity_local_mean.tif",
        output_dir / "engineered_gravity_roughness.tif",
        output_dir / "engineered_gravity_shape.tif",
        output_dir / "engineered_magnetic.tif",
        output_dir / "engineered_magnetic_gradient.tif",
        output_dir / "engineered_magnetic_local_mean.tif",
        output_dir / "engineered_magnetic_roughness.tif",
        output_dir / "engineered_magnetic_shape.tif"
    ]
    
    # Output
    probability_map = output_dir / "usa_mineral_probability.tif"
    
    # Validate inputs
    logger.info("\n📋 Validating input files...")
    
    if not usa_deposits_csv.exists():
        raise FileNotFoundError(f"USA deposits CSV not found: {usa_deposits_csv}")
    logger.info(f"  ✅ Deposits CSV: {usa_deposits_csv}")
    
    if not gravity_mosaic.exists():
        raise FileNotFoundError(f"Gravity mosaic not found: {gravity_mosaic}")
    logger.info(f"  ✅ Gravity mosaic: {gravity_mosaic}")
    
    # Check which engineered features exist
    valid_features = []
    for feat in engineered_features:
        if feat.exists():
            valid_features.append(str(feat))
            logger.info(f"  ✅ Feature: {feat.name}")
        else:
            logger.warning(f"  ⚠️  Feature not found (will skip): {feat.name}")
    
    if len(valid_features) == 0:
        raise FileNotFoundError("No engineered features found - check data/outputs/usa_supervised/")
    
    logger.info(f"\n  Total features for training: {len(valid_features)}")
    
    # Load USA mineral deposits
    logger.info("\n📍 Loading USA mineral deposit coordinates...")
    deposits_df = pd.read_csv(usa_deposits_csv)
    
    # Filter for USA deposits only
    if 'country' in deposits_df.columns:
        usa_df = deposits_df[deposits_df['country'] == 'United States'].copy()
    else:
        usa_df = deposits_df.copy()
    
    # Extract coordinates (latitude, longitude)
    if 'latitude' in usa_df.columns and 'longitude' in usa_df.columns:
        positive_coords = usa_df[['latitude', 'longitude']].values
    else:
        raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
    
    # Remove NaN coordinates
    valid_mask = ~np.isnan(positive_coords).any(axis=1)
    positive_coords = positive_coords[valid_mask]
    
    logger.info(f"  ✅ Loaded {len(positive_coords):,} valid USA mineral deposit locations")
    logger.info(f"  Latitude range: {positive_coords[:, 0].min():.2f} to {positive_coords[:, 0].max():.2f}")
    logger.info(f"  Longitude range: {positive_coords[:, 1].min():.2f} to {positive_coords[:, 1].max():.2f}")
    
    # Filter deposits to continental USA only (exclude Puerto Rico, Hawaii, Alaska territories)
    logger.info("\n🗺️  Filtering to continental USA only...")
    continental_usa_mask = (
        (positive_coords[:, 0] >= 24.5) & (positive_coords[:, 0] <= 49.0) &   # Latitude
        (positive_coords[:, 1] >= -125.0) & (positive_coords[:, 1] <= -66.5)   # Longitude
    )
    positive_coords = positive_coords[continental_usa_mask]
    logger.info(f"  ✅ Filtered to {len(positive_coords):,} continental USA deposits (excluding territories)")
    logger.info(f"  New latitude range: {positive_coords[:, 0].min():.2f} to {positive_coords[:, 0].max():.2f}")
    logger.info(f"  New longitude range: {positive_coords[:, 1].min():.2f} to {positive_coords[:, 1].max():.2f}")
    
    # Configuration - AGGRESSIVE REGULARIZATION to prevent overfitting
    negative_ratio = 1.0  # 1:1 negative to positive (balanced sampling to reduce memorization)
    n_estimators = 50     # Reduced from 200 to 50 (fewer trees to prevent overfitting)
    max_depth = 5         # Shallow trees for better generalization
    min_samples_leaf = 50 # Increased from 10 to 50 (force larger leaves to prevent memorization)
    n_workers = 20        # Use all 20 cores
    block_size = 8192     # Large blocks for better RAM utilization
    
    logger.info("\n⚙️  Training Configuration:")
    logger.info(f"  Positive samples: {len(positive_coords):,}")
    logger.info(f"  Negative ratio: {negative_ratio}:1")
    logger.info(f"  Expected negative samples: {int(len(positive_coords) * negative_ratio):,}")
    logger.info(f"  Random Forest estimators: {n_estimators}")
    logger.info(f"  Max tree depth: {max_depth}")
    logger.info(f"  Min samples per leaf: {min_samples_leaf}")
    logger.info(f"  Parallel workers: {n_workers} cores")
    logger.info(f"  Block size: {block_size}x{block_size}")
    logger.info(f"  Custom class weights: {{0: 1, 1: 10}} (to improve probability calibration)")
    
    # Execute supervised learning workflow
    logger.info("\n🚀 Starting supervised learning workflow...")
    logger.info("This may take 30-60 minutes depending on raster sizes...")
    
    try:
        classifier = classify_supervised(
            feature_paths=valid_features,
            positive_coords=positive_coords,
            output_path=str(probability_map),
            negative_ratio=negative_ratio,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            class_weight={0: 1, 1: 10},  # Custom weights instead of 'balanced' for better calibration
            random_state=42,
            parallel=True,
            n_workers=n_workers,
            block_size=block_size
        )
        
        # Display feature importances
        logger.info("\n📊 FEATURE IMPORTANCES (Top 10):")
        logger.info("=" * 60)
        
        feature_names = [Path(f).stem for f in valid_features]
        
        # Extract feature importances from calibrated classifier
        # CalibratedClassifierCV wraps the base estimator
        if hasattr(classifier, 'calibrated_classifiers_'):
            # Get base estimator from first fold
            base_estimator = classifier.calibrated_classifiers_[0].estimator
            importances = base_estimator.feature_importances_
        else:
            # Fallback if not calibrated
            importances = classifier.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        for i, idx in enumerate(indices[:min(10, len(indices))]):
            logger.info(f"  {i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")
        
        logger.info("=" * 60)
        
        # Verify output
        if probability_map.exists():
            file_size_mb = probability_map.stat().st_size / (1024 * 1024)
            logger.info(f"\n✅ Probability map generated: {probability_map}")
            logger.info(f"   File size: {file_size_mb:.1f} MB")
        else:
            logger.error(f"\n❌ Probability map not found: {probability_map}")
        
        # Calculate execution time
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 USA supervised learning workflow completed successfully!")
        logger.info(f"⏱️  Total execution time: {elapsed_min:.1f} minutes ({elapsed:.0f} seconds)")
        logger.info("=" * 80)
        
        # Final summary
        logger.info("\n📁 OUTPUT FILES:")
        logger.info(f"  - Probability map: {probability_map}")
        logger.info(f"  - File size: {file_size_mb:.1f} MB")
        logger.info("\n📊 MODEL SUMMARY:")
        logger.info(f"  - Training samples: {len(positive_coords) + int(len(positive_coords) * negative_ratio):,}")
        logger.info(f"  - Features used: {len(valid_features)}")
        logger.info(f"  - Random Forest trees: {n_estimators}")
        logger.info(f"  - Top feature: {feature_names[indices[0]]}")
        
    except Exception as e:
        logger.error(f"\n❌ Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()
