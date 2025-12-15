#!/usr/bin/env python3
"""
verify_geography.py - Basic sanity checks on coordinates
"""

import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import os

def validate_geography(csv_path):
    """
    Check if targets are in valid US locations.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'probability' not in df.columns:
        df['probability'] = df.get('density_contrast', 0.5)
    
    # Load US boundaries
    shapefile_path = 'data/reference/cb_2018_us_state_500k.shp'
    if not os.path.exists(shapefile_path):
        print(f"Warning: Detail US shapefile not found at {shapefile_path}.")
        print("Creating simplified bounding box check instead...")
        # Fallback to simple bounds check if shapefile missing
        usa = None
    else:
        usa = gpd.read_file(shapefile_path)
    
    results = []
    
    for idx, row in df.iterrows():
        point = Point(row['longitude'], row['latitude'])
        
        # Check 1: In continental US bounds (approximate)
        in_conus = (-125 <= row['longitude'] <= -65 and 
                    25 <= row['latitude'] <= 50)
        
        # Check 2: In a US state (exact check if shapefile exists)
        if usa is not None:
            in_state = usa.contains(point).any()
        else:
            in_state = in_conus # Fallback
        
        # Check 3: Not in major cities (unmined zones)
        major_cities = [
            (-118.24, 34.05, 50),   # LA (50km radius)
            (-122.42, 37.77, 30),   # SF
            (-104.99, 39.74, 30),   # Denver
            (-112.07, 33.45, 30),   # Phoenix
        ]
        
        in_city = False
        for city_lon, city_lat, radius_km in major_cities:
            dist_km = haversine(row['longitude'], row['latitude'], 
                               city_lon, city_lat)
            if dist_km < radius_km:
                in_city = True
                break
        
        results.append({
            'target_id': idx,
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'probability': row['probability'],
            'in_conus': in_conus,
            'in_state': in_state,
            'in_city': in_city,
            'valid': in_conus and in_state and not in_city
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print(f"Total Targets: {len(results_df)}")
    print(f"Valid Geography: {results_df['valid'].sum()} ({results_df['valid'].sum()/len(results_df)*100:.1f}%)")
    if usa is not None:
        print(f"In Ocean/Canada/Mexico: {(~results_df['in_state']).sum()}")
    print(f"In Major Cities: {results_df['in_city'].sum()}")
    
    # Save filtered list
    output_path = 'data/outputs/targets_geography_verified.csv'
    valid_targets = results_df[results_df['valid']]
    valid_targets.to_csv(output_path, index=False)
    print(f"Saved {len(valid_targets)} valid targets to {output_path}")
    
    return results_df

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two points."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

if __name__ == "__main__":
    # Use the user's active file if available, otherwise default path
    input_file = 'data/outputs/undiscovered_targets.csv'
    if not os.path.exists(input_file):
        # Try finding any csv in outputs
        possible_files = [f for f in os.listdir('data/outputs') if f.endswith('.csv')]
        if possible_files:
            input_file = os.path.join('data/outputs', possible_files[0])
            print(f"Default input not found, using {input_file}")
    
    results = validate_geography(input_file)
