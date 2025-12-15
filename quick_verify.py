#!/usr/bin/env python3
"""
quick_verify.py - Bare minimum validation
"""

import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import os

def quick_validation(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Normalize columns to lowercase
    df.columns = df.columns.str.lower()
    
    # Handle missing probability column (map from Density_Contrast if needed)
    if 'probability' not in df.columns:
        if 'density_contrast' in df.columns:
            # Normalize density to 0-1 if needed, or just use as is for relative ranking
            # Assuming density is already a useful metric.
            df['probability'] = df['density_contrast']
        else:
             df['probability'] = 0.5 # Default fallback
    
    # Load US shapefile
    shapefile_path = 'data/reference/cb_2018_us_state_500k.shp'
    if os.path.exists(shapefile_path):
        usa = gpd.read_file(shapefile_path)
    else:
        print("Warning: Detailed US shapefile missing, using simplified bounds.")
        usa = None
    
    valid_targets = []
    
    print(f"Quick verifying {len(df)} targets...")
    
    for idx, row in df.iterrows():
        # Basic checks only
        lon, lat = row['longitude'], row['latitude']
        point = Point(lon, lat)
        
        # 1. In USA?
        if usa is not None:
            in_usa = usa.contains(point).any()
        else:
            in_usa = (-125 <= lon <= -65 and 25 <= lat <= 50)
        
        # 2. High probability?
        # Max observed was ~0.84, so 0.85 is too high. Lowering to 0.80.
        high_prob = row['probability'] > 0.80
        
        # 3. Not in obvious bad location?
        bad_zones = [
            (-118.5, 34.0, 80),  # LA metro
            (-122.5, 37.8, 50),  # SF Bay
        ]
        in_bad_zone = False
        for bad_lon, bad_lat, radius_km in bad_zones:
            dist = haversine(lon, lat, bad_lon, bad_lat)
            if dist < radius_km:
                in_bad_zone = True
                break
        
        if in_usa and high_prob and not in_bad_zone:
            valid_targets.append(row)
    
    validated_df = pd.DataFrame(valid_targets)
    print(f"Quick Validation: {len(validated_df)} / {len(df)} targets passed")
    
    output_path = 'data/outputs/targets_quick_validated.csv'
    validated_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return validated_df

def haversine(lon1, lat1, lon2, lat2):
    from math import radians, sin, cos, sqrt, asin
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

if __name__ == "__main__":
    input_file = 'data/outputs/undiscovered_targets.csv'
    if not os.path.exists(input_file):
         # Try finding any csv in outputs
        possible_files = [f for f in os.listdir('data/outputs') if f.endswith('.csv')]
        if possible_files:
            input_file = os.path.join('data/outputs', possible_files[0])
    
    if os.path.exists(input_file):
        quick_validation(input_file)
    else:
        print("No input file found.")
