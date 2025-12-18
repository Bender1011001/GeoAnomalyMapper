
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def validate_geochem():
    print("--- GEOCHEMICAL VALIDATION START (COVERAGE CORRECTED) ---")
    
    # 1. Load Targets
    target_path = 'data/outputs/usa_targets.csv'
    try:
        targets = pd.read_csv(target_path)
    except FileNotFoundError:
        print(f"Error: {target_path} not found.")
        return
        
    print(f"Loaded {len(targets)} Gravity/Mag Targets.")
    
    # 2. Load NURE Data (Optimized)
    nure_path = 'data/validation/nuresed.csv'
    nure_cols = ['latitude', 'longitude', 'cu_ppm', 'au_ppm', 'zn_ppm', 'pb_ppm']
    
    print("Loading NURE Geochemistry (this may take a moment)...")
    try:
        nure = pd.read_csv(nure_path, usecols=nure_cols, low_memory=False)
        print(f"Loaded {len(nure)} Sediment Samples.")
        
        # CLEAN DATA IMMEDIATELY
        nure = nure.dropna(subset=['latitude', 'longitude'])
        nure = nure[(nure['latitude'] != 0) & (nure['longitude'] != 0)]
        print(f"Cleaned Samples (Valid Coords): {len(nure)}")
        
        # 3. Define Coverage (All Samples)
        # We need this to check if a target has ANY sample nearby
        all_samples = nure[['latitude', 'longitude']].values
        
    except Exception as e:
        print(f"Error loading NURE: {e}")
        return

    # 4. Filter for Anomalies (Robust Stat)
    print("Calculating Geochemical Anomalies...")
    anomalies = []
    elements = ['cu_ppm', 'au_ppm', 'zn_ppm', 'pb_ppm']
    
    for el in elements:
        nure[el] = pd.to_numeric(nure[el], errors='coerce')
        valid_el = nure.dropna(subset=[el])
        
        if len(valid_el) < 100:
             continue

        # Rank-based Top 5%
        percentiles = valid_el[el].rank(pct=True, method='min')
        is_anomalous = percentiles > 0.95
        
        count = is_anomalous.sum()
        if count == 0: continue
        
        val_thresh = valid_el.loc[is_anomalous, el].min()
        print(f"  {el.upper()} > {val_thresh:.4f} ppm: {count} samples")

        anomalous_samples = valid_el[is_anomalous][['latitude', 'longitude']].copy()
        anomalies.append(anomalous_samples)
        
    all_anomalies = pd.concat(anomalies).drop_duplicates()
    print(f"Total Geochemical Anomalies Points: {len(all_anomalies)}")
    
    # 5. Spatial Match
    print("Matching Targets to Geochem...")
    
    if 'Longitude' in targets.columns and targets['Longitude'].max() > 0:
        targets['Longitude'] = -1 * np.abs(targets['Longitude'])
        
    target_coords = targets[['Latitude', 'Longitude']].values
    
    # Build Trees
    anom_tree = cKDTree(all_anomalies[['latitude', 'longitude']].values)
    coverage_tree = cKDTree(all_samples)
    
    radius_deg = 5 / 111.0 # 5km
    
    # A. Check Coverage (Denominator)
    print("Calculating Sampling Coverage...")
    coverage_matches = coverage_tree.query_ball_point(target_coords, r=radius_deg)
    
    covered_indices = [i for i, m in enumerate(coverage_matches) if len(m) > 0]
    covered_targets = targets.iloc[covered_indices]
    
    print("-" * 30)
    print(f"COVERAGE ANALYSIS:")
    print(f"Total Targets: {len(targets)}")
    print(f"Targets with nearby Soil Samples (5km): {len(covered_targets)} ({len(covered_targets)/len(targets):.1%})")
    print(f"Targets in Sampling Voids: {len(targets) - len(covered_targets)}")
    print("-" * 30)
    
    if len(covered_targets) == 0:
        print("No targets have geochemical coverage.")
        return

    # B. Check Validation (Numerator)
    covered_coords = covered_targets[['Latitude', 'Longitude']].values
    validation_matches = anom_tree.query_ball_point(covered_coords, r=radius_deg)
    
    validated_count = sum(1 for m in validation_matches if len(m) > 0)
    
    print(f"VALIDATION RESULT (Sampled Areas Only):")
    print(f"Confirmed by Geochem: {validated_count}")
    print(f"Validation Rate: {(validated_count/len(covered_targets))*100:.1f}%")
    print("-" * 30)
    
    # Save
    val_indices_in_subset = [i for i, m in enumerate(validation_matches) if len(m) > 0]
    final_validated = covered_targets.iloc[val_indices_in_subset].copy()
    final_validated['Validation_Source'] = 'Geochemistry'
    final_validated.to_csv('data/outputs/usa_targets_geochem_validated.csv', index=False)
    print("Saved validated list.")

if __name__ == "__main__":
    validate_geochem()
