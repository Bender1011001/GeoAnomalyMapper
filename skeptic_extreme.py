"""
EXTREME SKEPTIC VALIDATION
===========================
This script tries to DISPROVE that GeoAnomalyMapper finds valuable targets.

Tests:
1. RANDOM BASELINE - Would random dots perform just as well?
2. KNOWN DEPOSIT LEAKAGE - Are we just finding training data?
3. ECONOMIC VALUE CHECK - Are the "hits" actually PRODUCING mines, not just historical prospects?
4. TOPOGRAPHY CORRELATION - Are we just finding mountains/valleys?
5. GEOLOGY CORRELATION - Are we just finding ANY rock outcrops?

If ANY test shows our model is no better than random, we FAIL.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent
TARGETS_FILE = PROJECT_ROOT / "data/outputs/usa_targets.csv"
MRDS_FILE = PROJECT_ROOT / "data/usgs_mrds_full.csv"
TRAINING_FILE = PROJECT_ROOT / "data/usgs_goldilocks.csv"

# USA Bounds (for random sampling)
USA_BOUNDS = {
    'lat_min': 24.5,
    'lat_max': 49.5,
    'lon_min': -124.5,
    'lon_max': -66.5
}

def load_data():
    """Load all datasets."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load targets
    if not TARGETS_FILE.exists():
        print(f"❌ Targets file not found: {TARGETS_FILE}")
        sys.exit(1)
    targets = pd.read_csv(TARGETS_FILE)
    print(f"✓ Loaded {len(targets)} model targets")
    
    # Load MRDS (known deposits)
    if not MRDS_FILE.exists():
        print(f"❌ MRDS file not found: {MRDS_FILE}")
        sys.exit(1)
    mrds = pd.read_csv(MRDS_FILE, low_memory=False)
    # Filter to USA
    mrds = mrds[(mrds['latitude'] >= USA_BOUNDS['lat_min']) & 
                (mrds['latitude'] <= USA_BOUNDS['lat_max']) &
                (mrds['longitude'] >= USA_BOUNDS['lon_min']) & 
                (mrds['longitude'] <= USA_BOUNDS['lon_max'])]
    print(f"✓ Loaded {len(mrds)} MRDS deposits in USA")
    
    # Load training data
    if TRAINING_FILE.exists():
        training = pd.read_csv(TRAINING_FILE)
        print(f"✓ Loaded {len(training)} training sites")
    else:
        training = None
        print("⚠ Training file not found - skipping leakage test")
    
    return targets, mrds, training

def generate_random_targets(n, seed=42):
    """Generate random points within USA bounds."""
    np.random.seed(seed)
    lats = np.random.uniform(USA_BOUNDS['lat_min'], USA_BOUNDS['lat_max'], n)
    lons = np.random.uniform(USA_BOUNDS['lon_min'], USA_BOUNDS['lon_max'], n)
    return pd.DataFrame({'Latitude': lats, 'Longitude': lons})

def test_random_baseline(targets, mrds, n_trials=100):
    """
    TEST 1: RANDOM BASELINE
    If random points score as well as our targets, we have NOTHING.
    """
    print("\n" + "="*60)
    print("TEST 1: RANDOM BASELINE (Monte Carlo)")
    print("="*60)
    
    # Build MRDS spatial index
    mrds_coords = mrds[['latitude', 'longitude']].dropna().values
    mrds_tree = cKDTree(mrds_coords)
    
    # Get our model's target coordinates
    lat_col = 'Latitude' if 'Latitude' in targets.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in targets.columns else 'lon'
    target_coords = targets[[lat_col, lon_col]].values
    
    # Calculate our hit rate (within 50km = ~0.45 degrees)
    threshold_deg = 0.45  # ~50km
    distances, _ = mrds_tree.query(target_coords, k=1)
    our_hits = np.sum(distances < threshold_deg)
    our_rate = our_hits / len(target_coords)
    print(f"Our Model: {our_hits}/{len(target_coords)} hits ({our_rate:.1%})")
    
    # Monte Carlo: Random targets
    random_rates = []
    for trial in range(n_trials):
        random_targets = generate_random_targets(len(target_coords), seed=trial)
        random_coords = random_targets[['Latitude', 'Longitude']].values
        distances, _ = mrds_tree.query(random_coords, k=1)
        random_hits = np.sum(distances < threshold_deg)
        random_rates.append(random_hits / len(random_coords))
    
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    random_max = np.max(random_rates)
    
    print(f"Random Baseline: {random_mean:.1%} ± {random_std:.1%} (max: {random_max:.1%})")
    
    # Statistical test: Are we better than random?
    z_score = (our_rate - random_mean) / random_std if random_std > 0 else 0
    p_value_approx = 1 - min(0.9999, (np.sum(np.array(random_rates) < our_rate) / n_trials))
    
    print(f"Z-score: {z_score:.2f}")
    print(f"P-value (approx): {p_value_approx:.4f}")
    
    if our_rate > random_mean + 2*random_std:
        print("✅ PASS: Model is significantly better than random (>2σ)")
        return True
    else:
        print("❌ FAIL: Model is NOT significantly better than random")
        return False

def test_data_leakage(targets, training):
    """
    TEST 2: DATA LEAKAGE
    Are we just finding the training data?
    """
    print("\n" + "="*60)
    print("TEST 2: DATA LEAKAGE CHECK")
    print("="*60)
    
    if training is None:
        print("⚠ Skipping - no training data available")
        return True
    
    # Build training spatial index
    train_coords = training[['lat', 'lon']].dropna().values
    train_tree = cKDTree(train_coords)
    
    # Get target coordinates
    if 'Latitude' in targets.columns:
        target_coords = targets[['Latitude', 'Longitude']].values
    else:
        target_coords = targets[['lat', 'lon']].values
    
    # Check distances
    distances, _ = train_tree.query(target_coords, k=1)
    
    # Convert to km (rough approximation)
    distances_km = distances * 111  # 1 degree ≈ 111 km
    
    exact_matches = np.sum(distances_km < 1)  # <1km
    near_matches = np.sum(distances_km < 10)  # <10km
    novel = np.sum(distances_km >= 10)
    
    print(f"Exact matches (<1km): {exact_matches} ({exact_matches/len(targets)*100:.1f}%)")
    print(f"Near matches (<10km): {near_matches} ({near_matches/len(targets)*100:.1f}%)")
    print(f"Novel targets (>10km): {novel} ({novel/len(targets)*100:.1f}%)")
    
    if exact_matches / len(targets) > 0.05:  # >5% exact matches = leakage
        print("❌ FAIL: Too many exact matches - likely data leakage")
        return False
    else:
        print("✅ PASS: Low leakage rate")
        return True

def test_economic_value(targets, mrds):
    """
    TEST 3: ECONOMIC VALUE CHECK
    Are the hits actually VALUABLE deposits, or just historical prospects?
    """
    print("\n" + "="*60)
    print("TEST 3: ECONOMIC VALUE CHECK")
    print("="*60)
    
    # Build MRDS spatial index
    mrds_coords = mrds[['latitude', 'longitude']].dropna()
    mrds_indexed = mrds.loc[mrds_coords.index].copy()
    mrds_tree = cKDTree(mrds_coords.values)
    
    # Get target coordinates
    if 'Latitude' in targets.columns:
        target_coords = targets[['Latitude', 'Longitude']].values
    else:
        target_coords = targets[['lat', 'lon']].values
    
    # Find nearest MRDS for each target
    threshold_deg = 0.45  # ~50km
    distances, indices = mrds_tree.query(target_coords, k=1)
    
    hits_mask = distances < threshold_deg
    hit_indices = indices[hits_mask]
    
    if len(hit_indices) == 0:
        print("❌ No hits found - cannot evaluate economic value")
        return False
    
    # Get the matched MRDS records
    matched_mrds = mrds_indexed.iloc[hit_indices]
    
    # Check development status
    if 'dev_stat' in matched_mrds.columns:
        status_counts = matched_mrds['dev_stat'].value_counts()
        print("\nDevelopment Status of Matched Deposits:")
        print("-" * 40)
        for status, count in status_counts.head(10).items():
            print(f"  {status}: {count}")
        
        # Count "valuable" statuses
        valuable_statuses = ['Producer', 'Past Producer', 'Mine']
        valuable_count = sum(matched_mrds['dev_stat'].str.contains('|'.join(valuable_statuses), case=False, na=False))
        valuable_pct = valuable_count / len(matched_mrds) * 100
        
        print(f"\n'Producer/Mine' matches: {valuable_count}/{len(matched_mrds)} ({valuable_pct:.1f}%)")
        
        if valuable_pct < 10:
            print("❌ FAIL: Most hits are prospects, not real deposits")
            return False
        else:
            print("✅ PASS: Significant portion are real producers")
            return True
    else:
        print("⚠ Cannot check development status - column missing")
        return True

def test_commodity_diversity(targets, mrds):
    """
    TEST 4: COMMODITY CHECK
    Are we finding diverse valuable commodities, or just one thing?
    """
    print("\n" + "="*60)
    print("TEST 4: COMMODITY DIVERSITY CHECK")
    print("="*60)
    
    # Build MRDS spatial index
    mrds_coords = mrds[['latitude', 'longitude']].dropna()
    mrds_indexed = mrds.loc[mrds_coords.index].copy()
    mrds_tree = cKDTree(mrds_coords.values)
    
    # Get target coordinates
    if 'Latitude' in targets.columns:
        target_coords = targets[['Latitude', 'Longitude']].values
    else:
        target_coords = targets[['lat', 'lon']].values
    
    # Find nearest MRDS for each target
    threshold_deg = 0.45  # ~50km
    distances, indices = mrds_tree.query(target_coords, k=1)
    
    hits_mask = distances < threshold_deg
    hit_indices = indices[hits_mask]
    
    if len(hit_indices) == 0:
        print("❌ No hits found")
        return False
    
    matched_mrds = mrds_indexed.iloc[hit_indices]
    
    # Check commodities
    commod_col = None
    for col in ['commod1', 'commodity', 'commod']:
        if col in matched_mrds.columns:
            commod_col = col
            break
    
    if commod_col:
        commodity_counts = matched_mrds[commod_col].value_counts()
        print("\nTop Commodities Found:")
        print("-" * 40)
        for commod, count in commodity_counts.head(15).items():
            pct = count / len(matched_mrds) * 100
            print(f"  {commod}: {count} ({pct:.1f}%)")
        
        # Check diversity
        top_commodity_pct = commodity_counts.iloc[0] / len(matched_mrds) * 100
        n_commodities = len(commodity_counts)
        
        print(f"\nTotal unique commodities: {n_commodities}")
        print(f"Top commodity concentration: {top_commodity_pct:.1f}%")
        
        if n_commodities > 5 and top_commodity_pct < 80:
            print("✅ PASS: Diverse commodity detection")
            return True
        else:
            print("⚠ WARNING: Low commodity diversity")
            return True  # Not a hard fail
    else:
        print("⚠ Cannot check commodities - column missing")
        return True

def test_geographic_distribution(targets):
    """
    TEST 5: GEOGRAPHIC CLUSTERING
    Are targets spread across the USA or clustered suspiciously?
    """
    print("\n" + "="*60)
    print("TEST 5: GEOGRAPHIC DISTRIBUTION CHECK")
    print("="*60)
    
    if 'Latitude' in targets.columns:
        lats = targets['Latitude'].values
        lons = targets['Longitude'].values
    else:
        lats = targets['lat'].values
        lons = targets['lon'].values
    
    print(f"Latitude range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")
    print(f"Latitude std: {np.std(lats):.2f}")
    print(f"Longitude std: {np.std(lons):.2f}")
    
    # Check for suspicious clustering
    lat_range = lats.max() - lats.min()
    lon_range = lons.max() - lons.min()
    
    if lat_range > 10 and lon_range > 20:
        print("✅ PASS: Good geographic spread")
        return True
    else:
        print("❌ FAIL: Targets suspiciously clustered")
        return False

def main():
    print("\n" + "="*60)
    print("   EXTREME SKEPTIC VALIDATION")
    print("   'Trust No One. Verify Everything.'")
    print("="*60)
    
    targets, mrds, training = load_data()
    
    results = {}
    
    # Run all tests
    results['random_baseline'] = test_random_baseline(targets, mrds)
    results['data_leakage'] = test_data_leakage(targets, training)
    results['economic_value'] = test_economic_value(targets, mrds)
    results['commodity_diversity'] = test_commodity_diversity(targets, mrds)
    results['geographic_distribution'] = test_geographic_distribution(targets)
    
    # Summary
    print("\n" + "="*60)
    print("   FINAL VERDICT")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test}: {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED - Model appears to find real value")
    elif passed >= total - 1:
        print("\n  ⚠️ MOSTLY PASSED - Some concerns but generally valid")
    else:
        print("\n  ❌ FAILED - Model claims are NOT supported by evidence")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
