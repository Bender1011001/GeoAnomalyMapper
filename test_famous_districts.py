"""
SKEPTIC V3: SPECIFIC DISTRICT TEST
===================================
Can we detect targets near SPECIFIC famous mining districts?
This tests if we're finding real mineral belts, not just random MRDS noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

TARGETS_FILE = Path("data/outputs/usa_targets.csv")

# Famous mining districts with known coordinates
FAMOUS_DISTRICTS = {
    # Nevada Gold
    "Carlin Trend": (40.75, -116.3),
    "Cortez Trend": (40.15, -116.55),
    "Battle Mountain": (40.64, -117.02),
    "Goldfield": (37.71, -117.23),
    
    # Arizona Copper
    "Morenci": (33.08, -109.35),
    "Bagdad": (34.59, -113.21),
    "Ray Mine": (33.15, -110.97),
    "Miami-Globe": (33.40, -110.87),
    
    # Utah
    "Bingham Canyon": (40.52, -112.15),
    "Kennecott": (40.52, -112.15),
    
    # Montana
    "Butte": (46.00, -112.53),
    
    # Colorado
    "Cripple Creek": (38.75, -105.18),
    "Leadville": (39.25, -106.29),
    
    # Alaska
    "Pogo": (64.46, -144.92),
    "Fort Knox": (64.78, -147.42),
    "Red Dog": (68.07, -162.87),
    
    # California
    "Bodie": (38.21, -119.01),
    "Mother Lode": (38.35, -120.77),
    
    # South Carolina
    "Haile Gold": (34.45, -80.52),
}

def main():
    print("\n" + "="*60)
    print("   FAMOUS MINING DISTRICT TEST")
    print("   'Are we finding real mineral belts?'")
    print("="*60)
    
    targets = pd.read_csv(TARGETS_FILE)
    lat_col = 'Latitude' if 'Latitude' in targets.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in targets.columns else 'lon'
    target_coords = targets[[lat_col, lon_col]].values
    target_tree = cKDTree(target_coords)
    
    print(f"\nLoaded {len(targets)} targets")
    print(f"Testing {len(FAMOUS_DISTRICTS)} famous mining districts")
    
    # For each famous district, check if we have a target nearby
    threshold_deg = 0.45  # ~50km
    
    hits = []
    misses = []
    
    print("\nDistrict Detection:")
    print("-" * 50)
    
    for name, (lat, lon) in FAMOUS_DISTRICTS.items():
        distance, _ = target_tree.query([lat, lon], k=1)
        distance_km = distance * 111
        
        if distance < threshold_deg:
            hits.append(name)
            print(f"  HIT: {name} ({distance_km:.0f} km)")
        else:
            misses.append(name)
            print(f"  MISS: {name} ({distance_km:.0f} km)")
    
    hit_rate = len(hits) / len(FAMOUS_DISTRICTS) * 100
    
    print("\n" + "-" * 50)
    print(f"Famous Districts Detected: {len(hits)}/{len(FAMOUS_DISTRICTS)} ({hit_rate:.0f}%)")
    
    if misses:
        print(f"\nMissed: {', '.join(misses)}")
    
    # Random baseline: What's the chance of hitting these districts randomly?
    print("\n" + "="*60)
    print("   RANDOM BASELINE COMPARISON")
    print("="*60)
    
    random_hit_rates = []
    for seed in range(100):
        np.random.seed(seed)
        random_lats = np.random.uniform(24.5, 69.5, len(targets))
        random_lons = np.random.uniform(-165, -66, len(targets))
        random_coords = np.column_stack([random_lats, random_lons])
        random_tree = cKDTree(random_coords)
        
        hits_count = 0
        for name, (lat, lon) in FAMOUS_DISTRICTS.items():
            distance, _ = random_tree.query([lat, lon], k=1)
            if distance < threshold_deg:
                hits_count += 1
        
        random_hit_rates.append(hits_count / len(FAMOUS_DISTRICTS) * 100)
    
    random_mean = np.mean(random_hit_rates)
    random_std = np.std(random_hit_rates)
    
    print(f"Our Model: {hit_rate:.0f}%")
    print(f"Random Baseline: {random_mean:.0f}% +/- {random_std:.0f}%")
    
    z_score = (hit_rate - random_mean) / random_std if random_std > 0 else 0
    print(f"Z-score: {z_score:.2f}")
    
    if hit_rate >= 80:
        print("\nVERDICT: EXCELLENT - Detecting most famous districts")
    elif hit_rate >= 60:
        print("\nVERDICT: GOOD - Detecting majority of famous districts")
    elif hit_rate >= 40:
        print("\nVERDICT: MARGINAL - Some district detection")
    else:
        print("\nVERDICT: POOR - Missing most famous districts")

if __name__ == "__main__":
    main()
