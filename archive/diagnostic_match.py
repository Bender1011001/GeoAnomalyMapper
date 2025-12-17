import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def run_diagnostic():
    print("Running Diagnostic Match...")
    
    # 1. Load Data
    try:
        deposits = pd.read_csv('data/usgs_goldilocks.csv')
        targets = pd.read_csv('data/outputs/usa_targets.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Known Deposits: {len(deposits)}")
    print(f"Predicted Targets: {len(targets)}")
    
    # Check Target Columns
    print("\nTarget Columns:", targets.columns.tolist())
    
    # 2. Check Is_Undiscovered
    if 'Is_Undiscovered' in targets.columns:
        print("\nIs_Undiscovered Distribution:")
        print(targets['Is_Undiscovered'].value_counts())
    else:
        print("\n'Is_Undiscovered' column MISSING. This explains the 0% precision report.")

    if 'Dist_to_Known_km' in targets.columns:
        print("\nDist_to_Known_km stats:")
        print(targets['Dist_to_Known_km'].describe())
    
    # 3. Perform Manual Spatial Matching (cKDTree)
    print("\n--- RECALCULATING MATCHES ---")
    
    # Ensure coords
    d_lat = 'latitude' if 'latitude' in deposits.columns else 'lat'
    d_lon = 'longitude' if 'longitude' in deposits.columns else 'lon'
    
    t_lat = 'Latitude'
    t_lon = 'Longitude'
    
    dep_coords = deposits[[d_lat, d_lon]].values
    tar_coords = targets[[t_lat, t_lon]].values
    
    # Build Tree on TARGETS
    tree = cKDTree(tar_coords)
    
    # Query DEPOSITS against Tree
    # finding closest target for each deposit
    # k=1 gives distance to nearest neighbor
    distances, indices = tree.query(dep_coords, k=1)
    
    # Distances are in degrees. convert to km approx
    # 1 deg ~ 111km
    dist_km = distances * 111.0
    
    with open('sensitivity_result.txt', 'w') as f:
        for radius in [2.0, 5.0, 10.0]:
            matched = np.sum(dist_km <= radius)
            sens = matched / len(deposits)
            line = f"Deposits with target within {radius}km: {matched} ({sens:.1%})"
            print(line)
            f.write(line + "\n")
            
        print("\n--- PRECISION CHECK ---")
        # Precision = TP / Total Targets
        # A target is TP if it is close to a deposit.
        # We need to query TARGETS against DEPOSITS tree
        dep_tree = cKDTree(dep_coords)
        t_dists, t_inds = dep_tree.query(tar_coords, k=1)
        t_dists_km = t_dists * 111.0
        
        tp_targets = np.sum(t_dists_km <= 2.0)
        precision = tp_targets / len(targets)
        p_line = f"Targets within 2km of a known mine (TP): {tp_targets}\nReal Precision: {precision:.1%}"
        print(p_line)
        f.write(p_line + "\n")

if __name__ == "__main__":
    run_diagnostic()
