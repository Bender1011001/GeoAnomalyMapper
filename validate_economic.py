
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def validate_economic():
    print("--- ECONOMIC VALIDATION PHASE 2 (THE PRISTINE SEARCH) ---")
    
    # 1. Load Geochem-Confirmed Targets (The 141)
    target_path = 'data/outputs/usa_targets_geochem_validated.csv'
    try:
        targets = pd.read_csv(target_path)
    except FileNotFoundError:
        print(f"Error: {target_path} not found.")
        return
    
    if 'Longitude' in targets.columns and targets['Longitude'].max() > 0:
        targets['Longitude'] = -1 * np.abs(targets['Longitude'])
    target_coords = targets[['Latitude', 'Longitude']].values
    
    print(f"Loaded {len(targets)} Geochem-Confirmed Targets.")

    # 2. Load MRDS (All Types)
    mrds_path = 'data/usgs_mrds_full.csv'
    cols = ['latitude', 'longitude', 'dev_stat', 'site_name', 'commod1']
    mrds = pd.read_csv(mrds_path, usecols=cols, low_memory=False)
    mrds = mrds.dropna(subset=['latitude', 'longitude'])
    
    # 3. Identify Producers
    # Be robust to casing
    if 'dev_stat' in mrds.columns:
        mrds['dev_stat'] = mrds['dev_stat'].astype(str).str.title()
        
    producers = mrds[mrds['dev_stat'].isin(['Producer', 'Past Producer'])]
    
    # 4. Filter: Targets near Producers (The 60) vs Novel (The 81)
    prod_tree = cKDTree(producers[['latitude', 'longitude']].values)
    radius = 5 / 111.0 # 5km
    
    prod_matches = prod_tree.query_ball_point(target_coords, r=radius)
    
    is_near_producer = [len(m) > 0 for m in prod_matches]
    novel_indices = [i for i, x in enumerate(is_near_producer) if not x]
    
    novel_targets = targets.iloc[novel_indices].copy()
    
    print("-" * 30)
    print(f"Total Targets: {len(targets)}")
    print(f"Near Producers: {len(targets) - len(novel_targets)}")
    print(f"Novel Candidates (Not near Mine): {len(novel_targets)}")
    
    # 5. Check 'PRISTINE' Status
    # Distance to *ANY* MRDS point (Prospect, Occurrence, etc.)
    # If >5km from ANYTHING, it is truly undiscovered/untested.
    
    all_mrds_tree = cKDTree(mrds[['latitude', 'longitude']].values)
    novel_coords = novel_targets[['Latitude', 'Longitude']].values
    
    all_matches = all_mrds_tree.query_ball_point(novel_coords, r=radius)
    
    is_near_anything = [len(m) > 0 for m in all_matches]
    pristine_indices = [i for i, x in enumerate(is_near_anything) if not x]
    
    pristine_targets = novel_targets.iloc[pristine_indices].copy()
    
    print("-" * 30)
    print("Checking Novel Targets against ALL 300k MRDS records...")
    print(f"Novel Candidates: {len(novel_targets)}")
    print(f"Near Prospect/Occurrence: {len(novel_targets) - len(pristine_targets)}")
    print(f"Truly PRISTINE (No MRDS record within 5km): {len(pristine_targets)}")
    print("-" * 30)
    
    if len(pristine_targets) > 0:
        print("Top Pristine Targets:")
        print(pristine_targets[['Region_ID', 'Latitude', 'Longitude']].head(10))
        pristine_targets.to_csv('data/outputs/pristine_targets.csv', index=False)
        print("Saved 'data/outputs/pristine_targets.csv'")

if __name__ == "__main__":
    validate_economic()
