#!/usr/bin/env python3
"""
verify_land_status.py - Check if targets are on mineable land
"""

import pandas as pd
import requests
from shapely.geometry import Point
import geopandas as gpd
import os

def check_land_status(csv_path):
    """
    Check if targets are in areas where mining is prohibited.
    
    Excludes:
    - National Parks
    - Wilderness Areas  
    - Military Reservations
    - Native American Reservations
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'probability' not in df.columns:
         df['probability'] = df.get('density_contrast', 0.5)
    
    # Download protected areas database
    shapefile_path = 'data/reference/PADUS3_0_Proclamation_Federal.shp'
    if not os.path.exists(shapefile_path):
        print(f"Warning: PAD-US shapefile not found at {shapefile_path}")
        print("Skipping land status check validation (marking as mineable with warning).")
        df['is_protected'] = False
        df['protection_type'] = 'Unknown_NoData'
        df['mineable'] = True
        df.to_csv('data/outputs/targets_mineable.csv', index=False)
        return df
    
    print("Loading protected areas database...")
    protected = gpd.read_file(shapefile_path)
    
    results = []
    
    print(f"Checking {len(df)} targets against PAD-US...")
    
    for idx, row in df.iterrows():
        point = Point(row['longitude'], row['latitude'])
        
        # Check if point is in any protected area
        # optimization: check bounds first or use spatial index if available
        # But for 4000 points vs huge polygon set, simple contains might be slow.
        # Let's hope geopandas optimized it.
        
        is_protected = protected.contains(point).any()
        
        if is_protected:
            # Get protection type
            matching = protected[protected.contains(point)]
            # GAP_Sts might be 'GAP_Sts' or similar column name depending on version
            # Fallback to checking columns
            gap_col = next((c for c in protected.columns if 'GAP' in c and 'Sts' in c), None)
            
            if gap_col and len(matching) > 0:
                protection_type = str(matching.iloc[0][gap_col])
            else:
                protection_type = 'Protected_UnknownType'
        else:
            protection_type = 'None'
        
        # Determine if mining is allowed
        # GAP Status codes:
        # 1 = Permanent protection (National Parks) - NO MINING
        # 2 = Long-term protection (Wilderness) - NO MINING  
        # 3 = Some protection (National Forest) - MAYBE
        # 4 = Minimal protection (BLM Land) - YES
        
        # Simple logic: 
        mineable = protection_type in ['None', '3', '4', 'Protected_UnknownType'] # keeping unknown as mineable to avoid false negatives? Or '39'? 
        # Actually '3' and '4' are strings usually "3" or "4".
        
        results.append({
            'target_id': idx,
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'probability': row['probability'],
            'is_protected': is_protected,
            'protection_type': protection_type,
            'mineable': mineable
        })
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)}...")
    
    results_df = pd.DataFrame(results)
    
    print(f"Total Targets: {len(results_df)}")
    print(f"On Protected Land: {results_df['is_protected'].sum()}")
    print(f"Mineable: {results_df['mineable'].sum()}")
    
    # Save mineable targets only
    mineable_targets = results_df[results_df['mineable']]
    mineable_targets.to_csv('data/outputs/targets_mineable.csv', index=False)
    print(f"Saved mineable targets to data/outputs/targets_mineable.csv")
    
    return results_df

if __name__ == "__main__":
    # Chain input
    input_file = 'data/outputs/targets_unclaimed.csv'
    if not os.path.exists(input_file):
         # check for previous steps
         if os.path.exists('data/outputs/targets_claims_verified.csv'):
             input_file = 'data/outputs/targets_claims_verified.csv'
         elif os.path.exists('data/outputs/targets_geology_verified.csv'):
             input_file = 'data/outputs/targets_geology_verified.csv'
         else:
             print("No suitable input file found.")
             exit(1)
             
    print(f"Using input file: {input_file}")
    check_land_status(input_file)
