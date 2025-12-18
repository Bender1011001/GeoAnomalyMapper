import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial import cKDTree

def grade_targets(targets_csv, known_deposits_csv, output_dir='outputs'):
    """
    Filters and grades targets based on 5-Factor Scoring Model.
    """
    print(f"Loading targets from {targets_csv}...")
    try:
        targets = pd.read_csv(targets_csv)
    except FileNotFoundError:
        print(f"Error: {targets_csv} not found.")
        return

    print(f"Loading known deposits from {known_deposits_csv}...")
    try:
        known = pd.read_csv(known_deposits_csv)
    except FileNotFoundError:
        print(f"Error: {known_deposits_csv} not found.")
        return
        
    # Ensure known deposits have lat/lon
    if 'latitude' not in known.columns or 'longitude' not in known.columns:
        if 'lat' in known.columns: known.rename(columns={'lat': 'latitude'}, inplace=True)
        if 'lon' in known.columns: known.rename(columns={'lon': 'longitude'}, inplace=True)
        
    known = known.dropna(subset=['latitude', 'longitude'])

    # Build KDTree for fast spatial query
    known_coords = known[['latitude', 'longitude']].values
    tree = cKDTree(known_coords)
    
    # --- SCORING WEIGHTS ---
    W_STATUS   = 30
    W_COMMODITY = 20
    W_TYPE      = 15
    W_STRENGTH  = 20
    W_SIZE      = 15
    
    # 1. Status Points
    def get_status_score(status):
        if not isinstance(status, str): return 5
        s = status.lower()
        if 'past producer' in s or 'producer' in s: return 1.0 # Full 30 pts
        if 'prospect' in s: return 0.5 # 15 pts
        if 'occurrence' in s: return 0.16 # 5 pts
        return 0.16 # Default to occurrence

    # 2. Commodity Points
    tier1_keywords = ['Gold', 'Au', 'Silver', 'Ag', 'Lithium', 'Li', 'Cobalt', 'Co', 'Uranium', 'U', 'Rare Earth', 'REE', 'Platinum', 'Pt', 'Palladium', 'Pd']
    tier2_keywords = ['Copper', 'Cu', 'Zinc', 'Zn', 'Lead', 'Pb', 'Nickel', 'Ni', 'Molybdenum', 'Mo', 'Tin', 'Sn']
    
    def get_commodity_score(commods):
        if not isinstance(commods, str): return 0.0
        c = commods.lower()
        for k in tier1_keywords:
            if k.lower() in c: return 1.0
        for k in tier2_keywords:
            if k.lower() in c: return 0.5
        return 0.0

    # 3. Type Points
    # Prioritize large systems
    type_high = ['porphyry', 'massive sulfide', 'vms', 'sedex', 'kimberlite', 'iocg', 'skarn', 'pegmatite']
    type_med  = ['vein', 'lode', 'stockwork', 'replacement']
    
    def get_type_score(dep_type):
        if not isinstance(dep_type, str): return 0.0
        t = dep_type.lower()
        for k in type_high:
            if k in t: return 1.0
        for k in type_med:
            if k in t: return 0.6
        return 0.0

    # Normalize Target Metrics for scoring
    # Density Contrast and Area
    if 'Density_Contrast' in targets.columns:
        max_dens = targets['Density_Contrast'].max()
        if max_dens == 0: max_dens = 1
        targets['Norm_Strength'] = targets['Density_Contrast'] / max_dens
    else:
        targets['Norm_Strength'] = 0

    if 'Area_Pixels' in targets.columns:
        max_area = targets['Area_Pixels'].max() # Use quantile to avoid outlier skew?
         # simple max for now
        if max_area == 0: max_area = 1
        targets['Norm_Size'] = targets['Area_Pixels'] / max_area
    else:
        targets['Norm_Size'] = 0

    results = []
    
    print("Grading targets...")
    for idx, row in targets.iterrows():
        t_lat = row['Latitude']
        t_lon = row['Longitude']
        
        # Find nearest SINGLE deposit for "Identity" transfer
        dist, idx_nearest = tree.query([t_lat, t_lon], k=1)
        nearest = known.iloc[idx_nearest]
        
        # Distance check (0.1 deg ~ 11km). 
        # If too far, scoring relies only on intrinsic factors (Size/Strength)
        is_near = dist < 0.1 
        
        # Get Attributes
        n_status = nearest['dev_stat'] if 'dev_stat' in nearest else 'Unknown'
        n_commod = nearest['commod1'] if 'commod1' in nearest else 'Unknown'
        n_type   = nearest['type'] if 'type' in nearest else 'Unknown'
        
        # Calculate Scores
        s_status = get_status_score(n_status) * W_STATUS if is_near else 0
        s_commod = get_commodity_score(n_commod) * W_COMMODITY if is_near else 0
        s_type   = get_type_score(n_type) * W_TYPE if is_near else 0
        
        s_strength = row['Norm_Strength'] * W_STRENGTH
        s_size     = row['Norm_Size'] * W_SIZE
        
        total_score = s_status + s_commod + s_type + s_strength + s_size
        
        # Confidence Tier
        # > 75: Excellent
        # > 50: Good
        # > 25: Fair
        
        results.append({
            **row,
            'Confidence_Score': round(total_score, 1),
            'Nearby_Mine_Status': n_status if is_near else "None",
            'Predicted_Commodity': n_commod if is_near else "Unknown",
            'Predicted_Type': n_type if is_near else "Unknown",
            'Distance_to_Known_km': round(dist * 111, 2)
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by='Confidence_Score', ascending=False)
    
    output_path = Path(output_dir) / "high_value_targets.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved graded targets to {output_path}")
    
    # Sanitized Summary update
    summary_path = Path(output_dir) / "sanitized_summary.md"
    top_tier = df[df['Confidence_Score'] >= 50]
    
    with open(summary_path, 'w') as f:
        f.write("# Exploration Opportunity Summary (Advanced Grading)\n\n")
        f.write("**CONFIDENTIAL**\n\n")
        f.write(f"**Total Targets:** {len(df)}\n")
        f.write(f"**High Confidence Targets (>50/100):** {len(top_tier)}\n\n")
        
        f.write("## Top Prospects\n")
        for i, r in top_tier.head(10).iterrows():
             f.write(f"- **ID {r['Region_ID']}**: Score **{r['Confidence_Score']}**. ")
             f.write(f"Potential: {r['Predicted_Commodity']} ({r['Predicted_Type']}). ")
             f.write(f"Status Linked: {r['Nearby_Mine_Status']}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", help="Path to extracted targets CSV")
    parser.add_argument("--known", default="data/usgs_goldilocks.csv", help="Path to known deposits CSV")
    parser.add_argument("--output-dir", default="data/outputs", help="Directory to save results")
    args = parser.parse_args()
    
    grade_targets(args.targets, args.known, output_dir=args.output_dir)
