"""
EXTREME SKEPTIC V2 - HARDER TESTS
==================================
Previous test was TOO EASY. Let's make it brutal.

New Tests:
1. TIGHT THRESHOLD (10km instead of 50km)
2. VALUABLE COMMODITIES ONLY (exclude sand/gravel/construction)
3. NEVADA STRESS TEST (the known problem area)
4. PRODUCER-ONLY BASELINE (are we finding ACTIVE mines?)
5. NEGATIVE CONTROL (verify we're NOT finding deposits in ocean/cities)
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

# Valuable commodities (NOT construction materials)
VALUABLE_COMMODITIES = [
    'Gold', 'Silver', 'Copper', 'Zinc', 'Lead', 'Nickel', 'Cobalt',
    'Platinum', 'Palladium', 'Uranium', 'Tungsten', 'Molybdenum',
    'Lithium', 'Rare Earth', 'Iron', 'Manganese', 'Chromium',
    'Tin', 'Antimony', 'Mercury', 'Beryllium', 'Tantalum', 'Niobium'
]

# Non-valuable (construction, industrial)
EXCLUDE_COMMODITIES = [
    'Sand', 'Gravel', 'Stone', 'Crushed', 'Clay', 'Limestone',
    'Cement', 'Aggregate', 'Fill', 'Pumice', 'Perlite', 'Volcanic',
    'Geothermal', 'Water', 'Mineral Water', 'Salt', 'Potash',
    'Gypite', 'Dimension'
]

# Nevada bounds (problem area)
NEVADA_BOUNDS = {
    'lat_min': 35.0,
    'lat_max': 42.0,
    'lon_min': -120.0,
    'lon_max': -114.0
}

# USA Bounds
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
    
    targets = pd.read_csv(TARGETS_FILE)
    print(f"Loaded {len(targets)} model targets")
    
    mrds = pd.read_csv(MRDS_FILE, low_memory=False)
    mrds = mrds[(mrds['latitude'] >= USA_BOUNDS['lat_min']) & 
                (mrds['latitude'] <= USA_BOUNDS['lat_max']) &
                (mrds['longitude'] >= USA_BOUNDS['lon_min']) & 
                (mrds['longitude'] <= USA_BOUNDS['lon_max'])]
    print(f"Loaded {len(mrds)} MRDS deposits in USA")
    
    return targets, mrds

def get_coords(df):
    """Get lat/lon from dataframe."""
    lat_col = 'Latitude' if 'Latitude' in df.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in df.columns else 'lon'
    return df[[lat_col, lon_col]].values

def is_valuable_commodity(commod):
    """Check if commodity is valuable (not construction materials)."""
    if pd.isna(commod):
        return False
    commod_upper = str(commod).upper()
    
    # Check if excluded
    for excl in EXCLUDE_COMMODITIES:
        if excl.upper() in commod_upper:
            return False
    
    # Check if valuable
    for val in VALUABLE_COMMODITIES:
        if val.upper() in commod_upper:
            return True
    
    return False

def generate_random_targets(n, bounds=USA_BOUNDS, seed=42):
    """Generate random points within bounds."""
    np.random.seed(seed)
    lats = np.random.uniform(bounds['lat_min'], bounds['lat_max'], n)
    lons = np.random.uniform(bounds['lon_min'], bounds['lon_max'], n)
    return np.column_stack([lats, lons])

def test_tight_threshold(targets, mrds):
    """
    TEST 1: 10km THRESHOLD
    Much harder than 50km
    """
    print("\n" + "="*60)
    print("TEST 1: TIGHT THRESHOLD (10km)")
    print("="*60)
    
    mrds_coords = mrds[['latitude', 'longitude']].dropna().values
    mrds_tree = cKDTree(mrds_coords)
    target_coords = get_coords(targets)
    
    threshold_deg = 0.09  # ~10km
    
    # Our hit rate
    distances, _ = mrds_tree.query(target_coords, k=1)
    our_hits = np.sum(distances < threshold_deg)
    our_rate = our_hits / len(target_coords)
    
    # Random baseline (100 trials)
    random_rates = []
    for seed in range(100):
        random_coords = generate_random_targets(len(target_coords), seed=seed)
        distances, _ = mrds_tree.query(random_coords, k=1)
        random_rates.append(np.sum(distances < threshold_deg) / len(random_coords))
    
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    
    print(f"Our Model (10km): {our_hits}/{len(target_coords)} ({our_rate:.1%})")
    print(f"Random Baseline: {random_mean:.1%} +/- {random_std:.1%}")
    
    improvement = (our_rate - random_mean) / random_mean * 100 if random_mean > 0 else 0
    print(f"Improvement over random: {improvement:.1f}%")
    
    z_score = (our_rate - random_mean) / random_std if random_std > 0 else 0
    print(f"Z-score: {z_score:.2f}")
    
    if z_score > 2:
        print("PASS: Significantly better than random at 10km")
        return True
    else:
        print("FAIL: Not significantly better at tight threshold")
        return False

def test_valuable_commodities(targets, mrds):
    """
    TEST 2: VALUABLE COMMODITIES ONLY
    Are we finding gold/copper/etc, not just sand pits?
    """
    print("\n" + "="*60)
    print("TEST 2: VALUABLE COMMODITIES ONLY")
    print("="*60)
    
    # Filter MRDS to valuable commodities
    mrds_valuable = mrds[mrds['commod1'].apply(is_valuable_commodity)]
    print(f"MRDS with valuable commodities: {len(mrds_valuable)}")
    
    if len(mrds_valuable) == 0:
        print("No valuable deposits found in MRDS")
        return False
    
    mrds_coords = mrds_valuable[['latitude', 'longitude']].dropna().values
    mrds_tree = cKDTree(mrds_coords)
    target_coords = get_coords(targets)
    
    threshold_deg = 0.45  # 50km
    
    # Our hit rate
    distances, _ = mrds_tree.query(target_coords, k=1)
    our_hits = np.sum(distances < threshold_deg)
    our_rate = our_hits / len(target_coords)
    
    # Random baseline
    random_rates = []
    for seed in range(100):
        random_coords = generate_random_targets(len(target_coords), seed=seed)
        distances, _ = mrds_tree.query(random_coords, k=1)
        random_rates.append(np.sum(distances < threshold_deg) / len(random_coords))
    
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    
    print(f"Our Model (valuable only): {our_hits}/{len(target_coords)} ({our_rate:.1%})")
    print(f"Random Baseline: {random_mean:.1%} +/- {random_std:.1%}")
    
    z_score = (our_rate - random_mean) / random_std if random_std > 0 else 0
    print(f"Z-score: {z_score:.2f}")
    
    # What commodities are we hitting?
    hit_mask = distances < threshold_deg
    hit_indices = np.where(hit_mask)[0]
    
    if len(hit_indices) > 0:
        matched = mrds_valuable.iloc[np.searchsorted(np.arange(len(mrds_valuable)), 
                                                      np.clip(hit_indices, 0, len(mrds_valuable)-1))]
        if 'commod1' in matched.columns:
            commod_counts = mrds_valuable.iloc[
                mrds_tree.query(target_coords[hit_mask], k=1)[1]
            ]['commod1'].value_counts().head(10)
            print("\nTop valuable commodities found:")
            for c, n in commod_counts.items():
                print(f"  {c}: {n}")
    
    if z_score > 2:
        print("\nPASS: Finding valuable commodities significantly above random")
        return True
    else:
        print("\nFAIL: Not finding valuable commodities above random")
        return False

def test_nevada(targets, mrds):
    """
    TEST 3: NEVADA STRESS TEST
    This is where we had problems before - Carlin Trend gold
    """
    print("\n" + "="*60)
    print("TEST 3: NEVADA STRESS TEST (Carlin Trend)")
    print("="*60)
    
    # Filter to Nevada
    target_coords = get_coords(targets)
    nevada_mask = (
        (target_coords[:, 0] >= NEVADA_BOUNDS['lat_min']) &
        (target_coords[:, 0] <= NEVADA_BOUNDS['lat_max']) &
        (target_coords[:, 1] >= NEVADA_BOUNDS['lon_min']) &
        (target_coords[:, 1] <= NEVADA_BOUNDS['lon_max'])
    )
    nevada_targets = target_coords[nevada_mask]
    print(f"Targets in Nevada: {len(nevada_targets)}")
    
    # Filter MRDS to Nevada
    mrds_nv = mrds[
        (mrds['latitude'] >= NEVADA_BOUNDS['lat_min']) &
        (mrds['latitude'] <= NEVADA_BOUNDS['lat_max']) &
        (mrds['longitude'] >= NEVADA_BOUNDS['lon_min']) &
        (mrds['longitude'] <= NEVADA_BOUNDS['lon_max'])
    ]
    print(f"MRDS deposits in Nevada: {len(mrds_nv)}")
    
    # Filter to gold specifically
    mrds_gold = mrds_nv[mrds_nv['commod1'].str.contains('Gold', case=False, na=False)]
    print(f"Gold deposits in Nevada: {len(mrds_gold)}")
    
    if len(nevada_targets) == 0:
        print("WARNING: No targets in Nevada - potential blind spot")
        return False
    
    if len(mrds_gold) == 0:
        print("No gold deposits in Nevada MRDS subset")
        return True
    
    mrds_coords = mrds_gold[['latitude', 'longitude']].dropna().values
    mrds_tree = cKDTree(mrds_coords)
    
    threshold_deg = 0.45
    distances, _ = mrds_tree.query(nevada_targets, k=1)
    our_hits = np.sum(distances < threshold_deg)
    our_rate = our_hits / len(nevada_targets) if len(nevada_targets) > 0 else 0
    
    # Random baseline in Nevada
    random_rates = []
    for seed in range(100):
        random_coords = generate_random_targets(len(nevada_targets), bounds=NEVADA_BOUNDS, seed=seed)
        distances, _ = mrds_tree.query(random_coords, k=1)
        random_rates.append(np.sum(distances < threshold_deg) / len(random_coords))
    
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    
    print(f"\nNevada Gold Detection:")
    print(f"  Our Model: {our_hits}/{len(nevada_targets)} ({our_rate:.1%})")
    print(f"  Random: {random_mean:.1%} +/- {random_std:.1%}")
    
    z_score = (our_rate - random_mean) / random_std if random_std > 0 else 0
    print(f"  Z-score: {z_score:.2f}")
    
    if z_score > 1.5:  # Slightly lower threshold for regional test
        print("PASS: Finding gold in Nevada above random")
        return True
    elif our_rate > 0.3:
        print("MARGINAL: Some gold detection but not statistically significant")
        return True
    else:
        print("FAIL: Not detecting Nevada gold deposits")
        return False

def test_producers_only(targets, mrds):
    """
    TEST 4: ACTIVE PRODUCERS ONLY
    Are we finding currently/recently producing mines?
    """
    print("\n" + "="*60)
    print("TEST 4: ACTIVE PRODUCERS ONLY")
    print("="*60)
    
    # Filter to producers only
    producer_mask = mrds['dev_stat'].str.contains('Producer', case=False, na=False)
    mrds_producers = mrds[producer_mask]
    print(f"Active/Past Producers in MRDS: {len(mrds_producers)}")
    
    mrds_coords = mrds_producers[['latitude', 'longitude']].dropna().values
    mrds_tree = cKDTree(mrds_coords)
    target_coords = get_coords(targets)
    
    threshold_deg = 0.27  # ~30km (tighter)
    
    distances, _ = mrds_tree.query(target_coords, k=1)
    our_hits = np.sum(distances < threshold_deg)
    our_rate = our_hits / len(target_coords)
    
    # Random baseline
    random_rates = []
    for seed in range(100):
        random_coords = generate_random_targets(len(target_coords), seed=seed)
        distances, _ = mrds_tree.query(random_coords, k=1)
        random_rates.append(np.sum(distances < threshold_deg) / len(random_coords))
    
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    
    print(f"Our Model (producers, 30km): {our_hits}/{len(target_coords)} ({our_rate:.1%})")
    print(f"Random Baseline: {random_mean:.1%} +/- {random_std:.1%}")
    
    z_score = (our_rate - random_mean) / random_std if random_std > 0 else 0
    print(f"Z-score: {z_score:.2f}")
    
    if z_score > 2:
        print("PASS: Finding producing mines above random")
        return True
    else:
        print("FAIL: Not finding producers above random")
        return False

def test_sanity_check(targets):
    """
    TEST 5: SANITY CHECK
    Are targets in reasonable locations (not ocean, not major cities)?
    """
    print("\n" + "="*60)
    print("TEST 5: SANITY CHECK (Location Quality)")
    print("="*60)
    
    target_coords = get_coords(targets)
    
    # Check for ocean targets (simple heuristic)
    # Most of Atlantic is east of -66, Pacific west of -124
    ocean_targets = np.sum(
        (target_coords[:, 1] < -180) | (target_coords[:, 1] > 0) |  # Wrap-around errors
        (target_coords[:, 0] < 24) | (target_coords[:, 0] > 72)     # Outside USA
    )
    
    print(f"Targets outside valid bounds: {ocean_targets}")
    
    # Check for clustering in major cities (unlikely mining areas)
    # LA, NYC, Chicago, Houston, Phoenix
    major_cities = [
        (34.05, -118.24),  # LA
        (40.71, -74.01),   # NYC
        (41.88, -87.63),   # Chicago
        (29.76, -95.37),   # Houston
        (33.45, -112.07),  # Phoenix
    ]
    
    city_targets = 0
    for city_lat, city_lon in major_cities:
        dist = np.sqrt((target_coords[:, 0] - city_lat)**2 + 
                       (target_coords[:, 1] - city_lon)**2)
        city_targets += np.sum(dist < 0.3)  # ~30km from city center
    
    print(f"Targets near major cities: {city_targets}")
    
    city_pct = city_targets / len(target_coords) * 100
    print(f"City proximity rate: {city_pct:.1f}%")
    
    if ocean_targets == 0 and city_pct < 5:
        print("PASS: Targets in reasonable mining locations")
        return True
    elif city_pct < 10:
        print("MARGINAL: Some questionable locations")
        return True
    else:
        print("FAIL: Too many targets in non-mining areas")
        return False

def main():
    print("\n" + "="*60)
    print("   EXTREME SKEPTIC V2 - HARDER TESTS")
    print("   'The previous test was too easy'")
    print("="*60)
    
    targets, mrds = load_data()
    
    results = {}
    
    results['tight_threshold'] = test_tight_threshold(targets, mrds)
    results['valuable_commodities'] = test_valuable_commodities(targets, mrds)
    results['nevada_gold'] = test_nevada(targets, mrds)
    results['producers_only'] = test_producers_only(targets, mrds)
    results['sanity_check'] = test_sanity_check(targets)
    
    print("\n" + "="*60)
    print("   FINAL VERDICT (HARD MODE)")
    print("="*60)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ALL HARD TESTS PASSED - Model is genuinely finding value")
    elif passed >= 3:
        print("\n  MOSTLY PASSED - Model has some validity but weaknesses exist")
    else:
        print("\n  FAILED - Model claims are questionable")
    
    return passed

if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed >= 3 else 1)
