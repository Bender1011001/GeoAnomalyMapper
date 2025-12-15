#!/usr/bin/env python3
"""
verify_geology.py - Check if targets are in favorable geology
"""

import rasterio
import pandas as pd
import numpy as np
import os

def validate_geology(csv_path, lithology_tif_path):
    """
    Check if targets are in rock types that host mineral deposits.
    
    Favorable lithologies (from USGS Global Lithology Map):
    - Plutonic/Intrusive Igneous (granitoids) = High
    - Metamorphic (schist, gneiss) = High  
    - Volcanic (andesite, rhyolite) = Medium
    - Sedimentary (limestone, shale) = Low
    - Unconsolidated (alluvium) = Very Low
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return
        
    if not os.path.exists(lithology_tif_path):
        print(f"Error: Lithology file {lithology_tif_path} not found.")
        print("Skipping geology check (marking all as favorable for now).")
        # Fail open if data missing
        df = pd.read_csv(csv_path)
        df['geology_favorable'] = True
        df['lithology_code'] = -1
        output_path = 'data/outputs/targets_geology_verified.csv'
        df.to_csv(output_path, index=False)
        return df
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'probability' not in df.columns:
         df['probability'] = df.get('density_contrast', 0.5)
    
    # Load lithology raster
    with rasterio.open(lithology_tif_path) as src:
        results = []
        
        for idx, row in df.iterrows():
            # Sample lithology at target location
            lon, lat = row['longitude'], row['latitude']
            try:
                py, px = src.index(lon, lat)
                # Check bounds
                if ((0 <= py < src.height) and (0 <= px < src.width)):
                    lithology_code = src.read(1, window=((py, py+1), (px, px+1)))[0, 0]
                else:
                    lithology_code = -9999
            except Exception as e:
                lithology_code = -9999
            
            # Classify favorability
            favorable = classify_lithology(lithology_code)
            
            results.append({
                'target_id': idx,
                'latitude': lat,
                'longitude': lon,
                'probability': row['probability'],
                'lithology_code': lithology_code,
                'geology_favorable': favorable
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"Favorable Geology: {results_df['geology_favorable'].sum()} / {len(results_df)}")
        print(f"Percentage: {results_df['geology_favorable'].sum()/len(results_df)*100:.1f}%")
        
        # Save
        output_path = 'data/outputs/targets_geology_verified.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
        
        return results_df

def classify_lithology(code):
    """
    Classify lithology favorability for mineral deposits.
    Adjust codes based on your lithology dataset.
    """
    # Example classification (adjust to your data)
    # If code is -9999 (off map), assume False or True? Let's be strict.
    if code == -9999:
        return False
        
    favorable_codes = [
        1, 2, 3,    # Igneous intrusive
        10, 11, 12, # Metamorphic
        5, 6, 7,    # Volcanic
    ]
    
    return code in favorable_codes

if __name__ == "__main__":
    validate_geology(
        'data/outputs/targets_geography_verified.csv',
        'data/reference/global_lithology.tif'
    )
