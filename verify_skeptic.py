#!/usr/bin/env python3
"""
Skeptic's Verification Script
=============================
Rigorous stress-testing of the GeoAnomalyMapper model results.
Performs 3 tests:
1. Spatial Block Holdout (West vs East): Detects spatial leakage.
2. Null Hypothesis (Random Labels): Baseline check.
3. Feature Audit: Checks for coordinate cheating.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# Import utilities from existing scripts
from classify_supervised import extract_features_at_points, sample_background_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the features and labels exactly as the pipeline does."""
    logger.info("Loading Data...")
    
    # 1. Feature Paths
    output_dir = Path("data/outputs/usa_supervised")
    features = [
        output_dir / "usa_gravity_mosaic.tif",
        output_dir / "usa_magnetic_mosaic.tif",
        Path("data/outputs/usa_density_model.tif")
    ]
    
    # Verify files
    valid_features = [str(f) for f in features if f.exists()]
    if len(valid_features) < 3:
        logger.error(f"Missing features! Found: {valid_features}")
        raise FileNotFoundError("Run the full pipeline first.")
    
    feature_names = ["Gravity", "Magnetics", "Density"]
    
    # 2. Deposit Locations (Positives)
    usgs_csv = Path("data/usgs_goldilocks.csv")
    if not usgs_csv.exists():
        raise FileNotFoundError(f"Missing {usgs_csv}")
        
    df = pd.read_csv(usgs_csv)
    # Ensure columns
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    
    pos_coords = df[[lat_col, lon_col]].values
    # Filter to USA approx bounds (already done in main pipeline, but good to be safe)
    # Actually, let's just use all valid ones to match pipeline
    pos_coords = pos_coords[~np.isnan(pos_coords).any(axis=1)]
    
    logger.info(f"Loaded {len(pos_coords)} positive training sites.")
    
    # 3. Extract Features for Positives
    logger.info("Extracting features for known deposits...")
    pos_features, valid_mask = extract_features_at_points(valid_features, pos_coords)
    pos_features = pos_features[valid_mask]
    pos_coords = pos_coords[valid_mask]
    
    # 4. Sample Background (Negatives)
    # Use 1:1 ratio for fairness
    n_neg = len(pos_features)
    logger.info(f"Sampling {n_neg} background points...")
    neg_features = sample_background_features(valid_features, n_samples=n_neg)
    
    # Synthesize Negative Coords (Random logic from sample_background_features doesn't return coords easily)
    # We need coords for Spatial Split. 
    # CRITICAL: sample_background_features returns only features.
    # We must RE-IMPLEMENT a simple sampler that returns coords too, OR modify the import.
    # Modifying import is risky. Let's just do a quick random sample here to get coords + extraction.
    
    # Re-doing simplified negative sampling to get coordinates
    with import_rasterio().open(valid_features[0]) as src:
        bounds = src.bounds
        neg_lats = np.random.uniform(bounds.bottom, bounds.top, n_neg)
        neg_lons = np.random.uniform(bounds.left, bounds.right, n_neg)
        neg_coords = np.column_stack([neg_lats, neg_lons])
        
    # Re-extract to be sure matches coords
    neg_features, valid_neg = extract_features_at_points(valid_features, neg_coords)
    neg_features = neg_features[valid_neg]
    neg_coords = neg_coords[valid_neg]
    
    # Prepare X and y
    X = np.vstack([pos_features, neg_features])
    y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
    coords = np.vstack([pos_coords, neg_coords]) # Lat, Lon
    
    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)
    
    logger.info(f"Final Dataset: {len(X)} samples.")
    return X, y, coords, feature_names

def import_rasterio():
    import rasterio
    return rasterio

def print_result(test_name, result, status):
    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"\n{icon} {test_name}: {status}")
    print(f"   {result}")

def test_1_spatial_holdout(X, y, coords):
    logger.info("\n--- TEST 1: SPATIAL BLOCK HOLDOUT (The 'Kansas Test') ---")
    logger.info("Splitting USA into WEST (Train) and EAST (Test) at Longitude -100")
    
    # Split
    west_mask = coords[:, 1] < -100
    east_mask = ~west_mask
    
    X_train, y_train = X[west_mask], y[west_mask]
    X_test, y_test = X[east_mask], y[east_mask]
    
    logger.info(f"Training on West ({len(X_train)}), Testing on East ({len(X_test)})")
    
    if len(X_test) < 100:
        logger.warning("Not enough samples in East for valid test. Moving split to median.")
        median_lon = np.median(coords[:, 1])
        west_mask = coords[:, 1] < median_lon
        east_mask = ~west_mask
        X_train, y_train = X[west_mask], y[west_mask]
        X_test, y_test = X[east_mask], y[east_mask]
        logger.info(f"New split at {median_lon:.2f}: Train {len(X_train)}, Test {len(X_test)}")

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    if len(np.unique(y_test)) < 2:
        logger.error("Test set does not contain both classes. Cannot compute ROC/AUC.")
        return
        
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    result_str = f"Accuracy: {acc:.1%}, AUC: {auc:.2f}, Sensitivity (Recall): {recall:.1%}"
    
    # Criteria: AUC > 0.65 suggests some generalization. > 0.8 is great. < 0.55 is failed.
    if auc > 0.65:
        print_result("Spatial Logic", result_str, "PASS")
    else:
        print_result("Spatial Logic", result_str + " (Model cannot predict outside training area)", "FAIL")

def test_2_null_hypothesis(X, y):
    logger.info("\n--- TEST 2: NULL HYPOTHESIS (Random Labels) ---")
    logger.info("Shuffling labels to destroy all real patterns...")
    
    y_shuffled = np.random.permutation(y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    # Train/Test split (random)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_shuffled, test_size=0.2, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred)
    
    result_str = f"Random Label AUC: {auc:.2f}"
    
    # Criteria: Random labels should yield ~0.5 AUC. If >> 0.5, leakage exists.
    if 0.45 <= auc <= 0.60: # Allow some noise wiggle room
        print_result("No Leakage Check", result_str + " (As expected, random labels fail)", "PASS")
    else:
        print_result("No Leakage Check", result_str + " (Suspicious! Random labels learned something?)", "FAIL")

def test_3_feature_audit(X, y, feature_names):
    logger.info("\n--- TEST 3: FEATURE IMPORTANCE AUDIT ---")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Predictors:")
    valid_predictors = True
    for f in range(len(feature_names)):
        idx = indices[f]
        name = feature_names[idx]
        score = importances[idx]
        print(f"  {f+1}. {name}: {score:.3f}")
        
        # Check against "Lat/Lon" cheating
        if "Lat" in name or "Lon" in name:
             if f < 2 and score > 0.2: # If Lat/Lon is primary driver
                 valid_predictors = False
    
    if valid_predictors:
        print_result("Feature Validity", "Model uses Physics (Gravity/Mag/Density), not coordinates.", "PASS")
    else:
        print_result("Feature Validity", "Model relies heavily on Lat/Lon!", "FAIL")

if __name__ == "__main__":
    try:
        X, y, coords, names = load_data()
        
        test_1_spatial_holdout(X, y, coords)
        test_2_null_hypothesis(X, y)
        test_3_feature_audit(X, y, names)
        
    except Exception as e:
        logger.exception(f"Detailed Verification Failed: {e}")
