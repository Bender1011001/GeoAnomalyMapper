import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def run_fix_diagnostic():
    print("Running Diagnostic (With Sign Correction)...")
    
    try:
        deposits = pd.read_csv('data/usgs_goldilocks.csv')
        targets = pd.read_csv('data/outputs/usa_targets.csv')
    except Exception as e:
        print(f"Error: {e}")
        return

    # Force Longitude to be negative (Western Hemisphere)
    # The data showed +103, which is wrong.
    targets['Longitude'] = -1 * np.abs(targets['Longitude'])
    
    # Coordinate Check after fix
    print(f"Fix applied. Mean Lon: {targets['Longitude'].mean():.4f}")
    
    # Tree Matching
    d_lat = 'latitude' if 'latitude' in deposits.columns else 'lat'
    d_lon = 'longitude' if 'longitude' in deposits.columns else 'lon'
    
    dep_coords = deposits[[d_lat, d_lon]].values
    tar_coords = targets[['Latitude', 'Longitude']].values
    
    tree = cKDTree(tar_coords)
    distances, indices = tree.query(dep_coords, k=1)
    dist_km = distances * 111.0
    
    with open('sensitivity_fixed.txt', 'w') as f:
        print("\n--- SENSITIVITY (FIXED) ---")
        f.write("--- SENSITIVITY (FIXED) ---\n")
        
        for radius in [2.0, 5.0, 10.0]:
            matched = np.sum(dist_km <= radius)
            sens = matched / len(deposits)
            line = f"Deposits with target within {radius}km: {matched} ({sens:.1%})"
            print(line)
            f.write(line + "\n")
            
        # Also Precision
        print("\n--- PRECISION (FIXED) ---")
        dep_tree = cKDTree(dep_coords)
        t_dists, t_inds = dep_tree.query(tar_coords, k=1)
        t_dists_km = t_dists * 111.0
        
        tp_targets = np.sum(t_dists_km <= 2.0)
        precision = tp_targets / len(targets)
        p_line = f"Targets within 2km of a known mine (TP): {tp_targets}\nReal Precision: {precision:.1%}"
        print(p_line)
        f.write(p_line + "\n")

if __name__ == "__main__":
    run_fix_diagnostic()
