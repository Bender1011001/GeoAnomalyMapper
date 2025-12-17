#!/usr/bin/env python3
"""
USA Supervised Model Validation Script
Anti-Cheating Validation for Random Forest Mineral Classifier

Performs rigorous validation to ensure the model is not:
1. Memorizing training locations (Cross-Validation test)
2. Flagging the entire map to get high detection rates (Map Coverage test)
3. Making random predictions with no precision (Precision test)

Success Criteria for PASS:
- Cross-validation sensitivity >30% (shows it generalizes beyond memorization)
- Map coverage <30% at threshold 0.5 (not flagging everything)
- Precision >0.01% (better than random flagging)
"""

import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from classify_supervised_optimized import (
    extract_features_at_points,
    sample_background_features,
    train_supervised_model
)
from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def perform_cross_validation(positive_coords, feature_paths, n_folds=10, 
                            n_negative_per_fold=None, random_state=42):
    """
    Perform K-Fold Cross-Validation to test model generalization.
    
    Args:
        positive_coords: Array of positive sample coordinates [lat, lon]
        feature_paths: List of feature raster paths
        n_folds: Number of cross-validation folds
        n_negative_per_fold: Number of negative samples per fold (default: 2x positive)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with CV results including sensitivities and statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION: {n_folds}-Fold Test")
    logger.info(f"{'='*80}")
    
    n_samples = len(positive_coords)
    logger.info(f"Total positive samples: {n_samples:,}")
    
    # Setup K-Fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_sensitivities = []
    fold_specificities = []
    fold_accuracies = []
    
    # Limit to subset if dataset is very large (for speed)
    max_samples_per_fold = 20000
    if n_samples > max_samples_per_fold * n_folds:
        logger.info(f"Dataset very large - using {max_samples_per_fold} samples per fold")
        # Randomly subsample
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, min(n_samples, max_samples_per_fold * n_folds), replace=False)
        positive_coords_cv = positive_coords[indices]
    else:
        positive_coords_cv = positive_coords
    
    logger.info(f"Using {len(positive_coords_cv):,} samples for CV")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(positive_coords_cv)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        start_time = time.time()
        
        # Split coordinates
        train_coords = positive_coords_cv[train_idx]
        test_coords = positive_coords_cv[test_idx]
        
        logger.info(f"Train: {len(train_coords):,} | Test: {len(test_coords):,}")
        
        try:
            # Extract features for training positives
            logger.info("Extracting training positive features...")
            train_pos_features, train_valid_mask = extract_features_at_points(
                feature_paths, train_coords
            )
            train_pos_features = train_pos_features[train_valid_mask]
            
            # Handle NaN values
            train_pos_features = np.nan_to_num(train_pos_features, nan=0.0)
            
            # Sample negative training data (use 2x positives as negatives)
            if n_negative_per_fold is None:
                n_negative_per_fold = len(train_pos_features) * 2
            
            logger.info(f"Sampling {n_negative_per_fold:,} negative samples...")
            train_neg_features = sample_background_features(
                feature_paths,
                n_samples=n_negative_per_fold,
                exclude_coords=train_coords,
                exclusion_radius=0.01,
                batch_size=20000
            )
            
            # Train model on this fold
            logger.info("Training fold model...")
            fold_clf, fold_scaler = train_supervised_model(
                train_pos_features,
                train_neg_features,
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=random_state
            )
            
            # Extract features for test positives
            logger.info("Extracting test positive features...")
            test_pos_features, test_valid_mask = extract_features_at_points(
                feature_paths, test_coords
            )
            test_pos_features = test_pos_features[test_valid_mask]
            test_pos_features = np.nan_to_num(test_pos_features, nan=0.0)
            
            # Sample test negatives (smaller set for speed)
            n_test_negative = min(len(test_pos_features) * 2, 5000)
            logger.info(f"Sampling {n_test_negative:,} test negative samples...")
            test_neg_features = sample_background_features(
                feature_paths,
                n_samples=n_test_negative,
                exclude_coords=test_coords,
                exclusion_radius=0.01,
                batch_size=10000
            )
            
            # Combine test data
            X_test = np.vstack([test_pos_features, test_neg_features])
            y_test = np.hstack([
                np.ones(len(test_pos_features)),
                np.zeros(len(test_neg_features))
            ])
            
            # Scale and predict
            X_test_scaled = fold_scaler.transform(X_test)
            y_pred_proba = fold_clf.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            accuracy = (tp + tn) / len(y_test)
            
            fold_sensitivities.append(sensitivity)
            fold_specificities.append(specificity)
            fold_accuracies.append(accuracy)
            
            elapsed = time.time() - start_time
            logger.info(f"Fold {fold_idx + 1} Results:")
            logger.info(f"  Sensitivity (Recall): {sensitivity*100:.1f}%")
            logger.info(f"  Specificity: {specificity*100:.1f}%")
            logger.info(f"  Accuracy: {accuracy*100:.1f}%")
            logger.info(f"  Time: {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Fold {fold_idx + 1} failed: {e}")
            # Use 0 for failed folds
            fold_sensitivities.append(0.0)
            fold_specificities.append(0.0)
            fold_accuracies.append(0.0)
            continue
    
    # Calculate statistics
    results = {
        'sensitivities': fold_sensitivities,
        'specificities': fold_specificities,
        'accuracies': fold_accuracies,
        'mean_sensitivity': np.mean(fold_sensitivities),
        'std_sensitivity': np.std(fold_sensitivities),
        'mean_specificity': np.mean(fold_specificities),
        'mean_accuracy': np.mean(fold_accuracies)
    }
    
    return results


def analyze_map_coverage(probability_map_path, deposits_coords, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    """
    Analyze probability map coverage at multiple thresholds.
    
    Args:
        probability_map_path: Path to probability GeoTIFF
        deposits_coords: Array of deposit coordinates [lat, lon]
        thresholds: List of probability thresholds to test
        
    Returns:
        Dictionary with coverage analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info("MAP COVERAGE ANALYSIS")
    logger.info(f"{'='*80}")
    
    logger.info(f"Loading probability map: {probability_map_path}")
    
    with rasterio.open(probability_map_path) as src:
        prob_map = src.read(1)
        transform = src.transform
        height, width = src.height, src.width
        
        # Get total valid pixels (non-NaN)
        valid_pixels = ~np.isnan(prob_map)
        total_pixels = np.sum(valid_pixels)
        
        logger.info(f"Map size: {height} x {width} = {height * width:,} pixels")
        logger.info(f"Valid pixels: {total_pixels:,}")
        
        # Convert deposit coordinates to pixel indices
        deposit_pixels = []
        for lat, lon in deposits_coords:
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < height and 0 <= col < width:
                    deposit_pixels.append((row, col))
            except:
                continue
        
        logger.info(f"Deposits within map bounds: {len(deposit_pixels):,}")
        
        # Analyze at each threshold
        results = []
        for thresh in thresholds:
            # Count flagged pixels
            flagged = (prob_map >= thresh) & valid_pixels
            n_flagged = np.sum(flagged)
            pct_flagged = (n_flagged / total_pixels * 100) if total_pixels > 0 else 0.0
            
            # Count detected deposits
            n_detected = 0
            for row, col in deposit_pixels:
                if flagged[row, col]:
                    n_detected += 1
            
            detection_rate = (n_detected / len(deposit_pixels) * 100) if len(deposit_pixels) > 0 else 0.0
            
            # Calculate precision (deposits per flagged pixel)
            precision = (n_detected / n_flagged * 100) if n_flagged > 0 else 0.0
            
            results.append({
                'threshold': thresh,
                'n_flagged': n_flagged,
                'pct_flagged': pct_flagged,
                'n_detected': n_detected,
                'detection_rate': detection_rate,
                'precision': precision
            })
            
            logger.info(f"Threshold {thresh:.1f}: {pct_flagged:.2f}% flagged, "
                       f"{detection_rate:.1f}% detected, precision {precision:.3f}%")
    
    return {
        'total_pixels': total_pixels,
        'n_deposits': len(deposit_pixels),
        'threshold_results': results
    }


def anti_cheating_validation(cv_results, coverage_results, training_sensitivity=0.92):
    """
    Perform anti-cheating validation tests.
    
    Args:
        cv_results: Dictionary from cross-validation
        coverage_results: Dictionary from map coverage analysis
        training_sensitivity: Reported training sensitivity for comparison
        
    Returns:
        Dictionary with pass/fail results for each test
    """
    logger.info(f"\n{'='*80}")
    logger.info("ANTI-CHEATING VALIDATION")
    logger.info(f"{'='*80}")
    
    tests = {}
    
    # Test 1: Map coverage at threshold 0.5
    thresh_05_result = [r for r in coverage_results['threshold_results'] if r['threshold'] == 0.5][0]
    map_coverage_pct = thresh_05_result['pct_flagged']
    
    test1_pass = map_coverage_pct < 30.0
    tests['map_coverage'] = {
        'name': 'Map coverage reasonable (<30% at 0.5)',
        'value': map_coverage_pct,
        'threshold': 30.0,
        'pass': test1_pass,
        'symbol': '✅' if test1_pass else '❌'
    }
    
    logger.info(f"{tests['map_coverage']['symbol']} Map Coverage: {map_coverage_pct:.2f}% "
               f"{'PASS' if test1_pass else 'FAIL'} (threshold: <30%)")
    
    # Test 2: Precision at threshold 0.5
    precision_pct = thresh_05_result['precision']
    
    test2_pass = precision_pct > 0.01
    tests['precision'] = {
        'name': 'Precision acceptable (>0.01%)',
        'value': precision_pct,
        'threshold': 0.01,
        'pass': test2_pass,
        'symbol': '✅' if test2_pass else '❌'
    }
    
    logger.info(f"{tests['precision']['symbol']} Precision: {precision_pct:.3f}% "
               f"{'PASS' if test2_pass else 'FAIL'} (threshold: >0.01%)")
    
    # Test 3: Cross-validation generalization
    cv_sensitivity_pct = cv_results['mean_sensitivity'] * 100
    
    test3_pass = cv_sensitivity_pct > 30.0
    tests['generalization'] = {
        'name': 'Generalization confirmed (CV sens >30%)',
        'value': cv_sensitivity_pct,
        'threshold': 30.0,
        'pass': test3_pass,
        'symbol': '✅' if test3_pass else '❌'
    }
    
    logger.info(f"{tests['generalization']['symbol']} CV Sensitivity: {cv_sensitivity_pct:.1f}% "
               f"{'PASS' if test3_pass else 'FAIL'} (threshold: >30%)")
    
    # Test 4: Check for memorization (CV much lower than training)
    sensitivity_drop = (training_sensitivity * 100) - cv_sensitivity_pct
    
    # It's normal for CV to be lower, but not by a huge amount if generalizing well
    # We'll flag if CV is <20% but training was >90%
    test4_pass = not (cv_sensitivity_pct < 20.0 and training_sensitivity > 0.90)
    tests['memorization'] = {
        'name': 'Not memorizing locations',
        'train_sens': training_sensitivity * 100,
        'cv_sens': cv_sensitivity_pct,
        'drop': sensitivity_drop,
        'pass': test4_pass,
        'symbol': '✅' if test4_pass else '❌'
    }
    
    logger.info(f"{tests['memorization']['symbol']} Memorization Check: "
               f"Training {training_sensitivity*100:.1f}% vs CV {cv_sensitivity_pct:.1f}% "
               f"{'PASS' if test4_pass else 'FAIL'}")
    
    # Overall verdict
    all_pass = all(t['pass'] for t in tests.values())
    
    return {
        'tests': tests,
        'overall_pass': all_pass
    }


def main():
    """Execute USA supervised model validation."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("==== USA SUPERVISED MODEL VALIDATION ====")
    print("="*80)
    
    # Define paths
    output_dir = OUTPUTS_DIR / "usa_supervised"
    usa_deposits_csv = Path("usa_deposits.csv")
    probability_map = output_dir / "usa_mineral_probability.tif"
    
    # Engineered features (use same as training)
    feature_paths = [
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
    
    # Filter existing features
    valid_features = [str(f) for f in feature_paths if f.exists()]
    
    if len(valid_features) == 0:
        raise FileNotFoundError(f"No engineered features found in {output_dir}")
    
    logger.info(f"\nFound {len(valid_features)} feature files")
    
    # Check probability map exists
    if not probability_map.exists():
        raise FileNotFoundError(f"Probability map not found: {probability_map}")
    
    logger.info(f"Probability map: {probability_map}")
    logger.info(f"Map size: {probability_map.stat().st_size / 1e6:.1f} MB")
    
    # Load USA deposits
    logger.info("\nLoading USA mineral deposits...")
    deposits_df = pd.read_csv(usa_deposits_csv)
    
    # Filter for USA
    if 'country' in deposits_df.columns:
        usa_df = deposits_df[deposits_df['country'] == 'United States'].copy()
    else:
        usa_df = deposits_df.copy()
    
    # Extract coordinates
    if 'latitude' in usa_df.columns and 'longitude' in usa_df.columns:
        positive_coords = usa_df[['latitude', 'longitude']].values
    else:
        raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
    
    # Remove NaN
    valid_mask = ~np.isnan(positive_coords).any(axis=1)
    positive_coords = positive_coords[valid_mask]
    
    logger.info(f"Loaded {len(positive_coords):,} USA mineral deposit locations")
    
    # ========================================
    # STEP 1: Cross-Validation
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Cross-Validation Test")
    logger.info("="*80)
    
    cv_results = perform_cross_validation(
        positive_coords,
        valid_features,
        n_folds=10,
        n_negative_per_fold=None,  # Will use 2x positives
        random_state=42
    )
    
    # ========================================
    # STEP 2: Map Coverage Analysis
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Map Coverage Analysis")
    logger.info("="*80)
    
    coverage_results = analyze_map_coverage(
        probability_map,
        positive_coords,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    
    # ========================================
    # STEP 3: Anti-Cheating Validation
    # ========================================
    validation_results = anti_cheating_validation(
        cv_results,
        coverage_results,
        training_sensitivity=0.92  # From task description
    )
    
    # ========================================
    # FINAL REPORT
    # ========================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("==== VALIDATION RESULTS ====")
    print("="*80)
    
    print("\nCross-Validation Results (10-Fold):")
    print(f"  Average Test Sensitivity: {cv_results['mean_sensitivity']*100:.1f}%")
    print(f"  Std Dev: {cv_results['std_sensitivity']*100:.1f}%")
    print(f"  Average Specificity: {cv_results['mean_specificity']*100:.1f}%")
    print(f"  Average Accuracy: {cv_results['mean_accuracy']*100:.1f}%")
    
    # Interpretation
    cv_sens_pct = cv_results['mean_sensitivity'] * 100
    if cv_sens_pct > 50:
        cv_interp = "EXCELLENT - Model generalizes very well"
    elif cv_sens_pct > 30:
        cv_interp = "PASS - Model shows good generalization"
    elif cv_sens_pct > 20:
        cv_interp = "MARGINAL - Some generalization but weak"
    else:
        cv_interp = "FAIL - Poor generalization, likely memorizing"
    
    print(f"  Interpretation: {cv_interp}")
    
    print("\nMap Coverage Analysis:")
    for result in coverage_results['threshold_results']:
        symbol = "✅" if result['threshold'] == 0.5 else "  "
        status = ""
        if result['threshold'] == 0.5:
            status = " [PASS]" if result['pct_flagged'] < 30.0 else " [FAIL]"
        
        print(f"  {symbol} Threshold {result['threshold']:.1f}: "
              f"{result['pct_flagged']:6.2f}% flagged, "
              f"{result['detection_rate']:5.1f}% sensitivity, "
              f"precision {result['precision']:6.3f}%{status}")
    
    print("\nAnti-Cheating Validation:")
    for test_key, test_data in validation_results['tests'].items():
        print(f"  {test_data['symbol']} {test_data['name']}")
    
    print(f"\nOVERALL VERDICT: {'✅ PASS' if validation_results['overall_pass'] else '❌ FAIL'}")
    
    print("\n" + "-"*80)
    print("Comparison to California-Only Baseline:")
    print("  California: 11.8% LOOCV sensitivity, 4.85% map flagged")
    print(f"  USA Model:  {cv_results['mean_sensitivity']*100:.1f}% CV sensitivity, "
          f"{coverage_results['threshold_results'][2]['pct_flagged']:.2f}% map flagged (@ 0.5)")
    
    if cv_results['mean_sensitivity'] * 100 > 11.8:
        improvement = cv_results['mean_sensitivity'] * 100 / 11.8
        print(f"  ✅ USA model shows {improvement:.1f}x improvement in generalization!")
    else:
        print(f"  ⚠️  USA model underperforms California baseline")
    
    print("\n" + "-"*80)
    print("Recommended Next Steps:")
    
    if validation_results['overall_pass']:
        print("  ✅ Model passed all validation tests!")
        print("  → Model is suitable for production use")
        print("  → Consider ensemble with unsupervised methods for even better coverage")
        print("  → Tune threshold based on use case (balance sensitivity vs precision)")
    else:
        # Identify which tests failed
        failed_tests = [k for k, v in validation_results['tests'].items() if not v['pass']]
        
        print("  ⚠️  Model failed validation tests:")
        for test_key in failed_tests:
            print(f"    - {validation_results['tests'][test_key]['name']}")
        
        print("\n  Recommended fixes:")
        
        if 'map_coverage' in failed_tests:
            print("    → Model is over-flagging the map")
            print("      • Increase regularization (max_depth, min_samples_leaf)")
            print("      • Use more negative samples in training")
            print("      • Consider feature selection to remove noisy features")
        
        if 'precision' in failed_tests:
            print("    → Model has poor precision (flagging too randomly)")
            print("      • Add more informative features")
            print("      • Increase class weights for positive class")
            print("      • Use probability calibration (Platt scaling)")
        
        if 'generalization' in failed_tests:
            print("    → Model not generalizing well")
            print("      • May need more diverse training data")
            print("      • Check for data leakage or duplicate samples")
            print("      • Try different model architectures (XGBoost, CatBoost)")
        
        if 'memorization' in failed_tests:
            print("    → Model is memorizing training locations")
            print("      • Increase regularization significantly")
            print("      • Reduce model complexity (fewer trees, shallower depth)")
            print("      • Check for coordinate-based features leaking location info")
    
    print(f"\n{'='*80}")
    print(f"Validation completed in {elapsed:.1f} seconds")
    print(f"{'='*80}\n")
    
    return validation_results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise
