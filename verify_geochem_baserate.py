
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import sys

def verify_baserate():
    print("--- GEOCHEM BASE RATE VERIFICATION (Monte Carlo) ---")
    
    # 1. Load Data
    try:
        targets = pd.read_csv('data/outputs/usa_targets.csv')
        anomalies = pd.read_csv('data/outputs/all_geochem_anomalies.csv')
        print(f"Loaded {len(targets)} Targets.")
        print(f"Loaded {len(anomalies)} Geochemical Anomalies.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Fix longitude sign if needed (sanity check)
    if targets['Longitude'].max() > 0:
        targets['Longitude'] = -1 * np.abs(targets['Longitude'])

    # 2. Define Bounding Box (All Targets)
    lat_min, lat_max = targets['Latitude'].min(), targets['Latitude'].max()
    lon_min, lon_max = targets['Longitude'].min(), targets['Longitude'].max()
    
    print(f"Target Bounding Box:")
    print(f"  Lat: {lat_min:.4f} to {lat_max:.4f}")
    print(f"  Lon: {lon_min:.4f} to {lon_max:.4f}")

    # 3. Build Tree for Anomalies
    anom_coords = anomalies[['latitude', 'longitude']].values
    tree = cKDTree(anom_coords)
    radius_deg = 5 / 111.0  # 5km in degrees (approx)

    # 4. Monte Carlo Simulation
    n_targets = len(targets)
    n_iterations = 100
    hit_rates = []

    print(f"\nRunning {n_iterations} iterations of {n_targets} random points...")
    
    for i in range(n_iterations):
        # Generate random points in bbox
        rand_lats = np.random.uniform(lat_min, lat_max, n_targets)
        rand_lons = np.random.uniform(lon_min, lon_max, n_targets)
        rand_coords = np.column_stack((rand_lats, rand_lons))
        
        # Query tree
        matches = tree.query_ball_point(rand_coords, r=radius_deg)
        
        # Count hits
        hits = sum([1 for m in matches if len(m) > 0])
        rate = (hits / n_targets) * 100
        hit_rates.append(rate)
        
        if (i+1) % 10 == 0:
            sys.stdout.write(f".")
            sys.stdout.flush()
            
    print("\n\n--- RESULTS ---")
    stats = pd.Series(hit_rates).describe()
    print(f"Mean Random Hit Rate:   {stats['mean']:.2f}%")
    print(f"Std Dev:                {stats['std']:.2f}%")
    print(f"Min / Max:              {stats['min']:.2f}% / {stats['max']:.2f}%")
    
    # 5. Conclusion
    model_rate = 7.8 
    print("-" * 30)
    print(f"Model Rate:             {model_rate}%")
    print(f"Random Baseline:        {stats['mean']:.2f}%")
    
    if model_rate > (stats['mean'] + 2*stats['std']):
        print("CONCLUSION: Model performance is STATISTICALLY SIGNIFICANT (> 2 sigma).")
    elif model_rate > stats['mean']:
        print("CONCLUSION: Model is slightly better than random, but weak.")
    else:
        print("CONCLUSION: Model is indistinguishable from (or worse than) random noise.")

if __name__ == "__main__":
    verify_baserate()
