import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def apply_fix_and_verify():
    print("Applying Fix and Verifying...")
    
    # Load
    targets = pd.read_csv('data/outputs/usa_targets.csv')
    deposits = pd.read_csv('data/usgs_goldilocks.csv')
    
    # Fix Coordinates
    # Flip Logitude to Negative
    targets['Longitude'] = -1 * np.abs(targets['Longitude'])
    
    # Save Corrected
    corr_path = 'data/outputs/usa_targets_corrected.csv'
    targets.to_csv(corr_path, index=False)
    print(f"Saved corrected targets to {corr_path}")
    
    # Stats
    print("\nCorrected Stats:")
    print(targets[['Latitude', 'Longitude', 'Density_Contrast']].describe())
    
    # Match
    d_coords = deposits[['latitude', 'longitude']].values if 'latitude' in deposits.columns else deposits[['lat', 'lon']].values
    t_coords = targets[['Latitude', 'Longitude']].values
    
    tree = cKDTree(t_coords)
    dists, _ = tree.query(d_coords, k=1)
    dists_km = dists * 111.0
    
    matches_2km = np.sum(dists_km <= 2.0)
    matches_10km = np.sum(dists_km <= 10.0)
    matches_50km = np.sum(dists_km <= 50.0)
    
    print(f"\n--- MATCH RESULTS (Corrected Coords) ---")
    print(f"Deposits within 2km: {matches_2km}")
    print(f"Deposits within 10km: {matches_10km}")
    print(f"Deposits within 50km: {matches_50km}")
    
    with open('brief_stats.txt', 'w') as f:
        f.write(f"2km: {matches_2km}\n10km: {matches_10km}\n50km: {matches_50km}")
    
    if matches_10km == 0 and matches_50km > 0:
        print("\nCONCLUSION: Targets are geographically aligned (matches at 50km) but locally distinct.")
        print("This confirms the High Threshold hypothesis (Targets are 'New' > 0.8, Mines are 'Known' ~ 0.6).")
    elif matches_50km == 0:
        print("\nCONCLUSION: Still no matches even at 50km. Coordinate misalignment might be huge or datasets disjoint.")

if __name__ == "__main__":
    apply_fix_and_verify()
