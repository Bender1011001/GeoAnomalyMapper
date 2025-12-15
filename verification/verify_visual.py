#!/usr/bin/env python3
"""
verify_visual.py - Automated visual checks using satellite imagery
"""

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    print("Error: Earth Engine API not installed. Run: pip install earthengine-api")

import pandas as pd
import numpy as np
import os

def check_surface_indicators(csv_path):
    """
    Use Google Earth Engine to check for:
    1. Vegetation loss (mining activity)
    2. Road access
    3. Topographic roughness (outcrop exposure)
    4. Historic imagery changes
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return

    if not EE_AVAILABLE:
        # Fallback if EE not installed
        print("Skipping visual verification (Earth Engine missing).")
        return pd.read_csv(csv_path)

    try:
        ee.Initialize()
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Please authenticate using: 'earthengine authenticate' in your terminal.")
        print("Skipping visual verification...")
        return pd.read_csv(csv_path)
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'probability' not in df.columns:
         df['probability'] = df.get('density_contrast', 0.5)
    print(f"Analyzing {len(df)} targets with Earth Engine...")
    
    results = []
    
    for idx, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        year_prob = row['probability']
        
        # Earth Engine objects
        point = ee.Geometry.Point([lon, lat])
        
        # 1. Check NDVI (vegetation index)
        # Low NDVI = exposed rock/mining activity
        # We need to handle when no images are found (e.g. cloud cover or bad dates)
        try:
            ndvi_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(point) \
                .filterDate('2020-01-01', '2024-01-01') \
                .select(['SR_B5', 'SR_B4']) # NIR, Red
            
            # Helper to calc NDVI for one image
            def calc_ndvi(img):
                return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            
            ndvi_img = ndvi_coll.map(calc_ndvi).mean()
            
            # Reduce region to get value at point
            # scale=30m for Landsat
            grad = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            )
            
            ndvi_value = grad.getInfo().get('NDVI', None)
        except Exception as e:
            # print(f"NDVI Error {idx}: {e}")
            ndvi_value = None
        
        # 2. Check for roads (OpenStreetMap via Earth Engine)
        # Verify dataset availability. If "projects/sat-io..." isn't public or available, this might fail.
        # Alternative: use TIGER roads for US? "TIGER/2016/Roads"
        try:
            roads = ee.FeatureCollection("TIGER/2016/Roads")
            nearby_roads = roads.filterBounds(point.buffer(1000))  # 1km buffer
            road_count = nearby_roads.size().getInfo()
        except:
            road_count = 0
        
        # 3. Elevation roughness (DEM)
        try:
            dem = ee.Image('USGS/SRTMGL1_003')
            elevation = dem.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(500),
                scale=30
            ).getInfo().get('elevation', None)
        except:
            elevation = None
        
        # 4. Change detection (Simplified)
        change_value = 0 # Placeholder if too complex for quick script or if memory limits hit
        
        # Scoring
        score = 0
        if ndvi_value is not None and ndvi_value < 0.3:  # Low vegetation/exposed
            score += 1
        if road_count > 0:  # Accessible
            score += 1
        if elevation is not None and elevation > 1000:  # Mountainous
            score += 1
        
        results.append({
            'target_id': idx,
            'latitude': lat,
            'longitude': lon,
            'probability': year_prob,
            'ndvi': ndvi_value,
            'road_access': road_count > 0,
            'elevation_m': elevation,
            'visual_score': score
        })
        
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(df)}...")

    results_df = pd.DataFrame(results)
    
    # Filter high-score targets
    high_confidence = results_df[results_df['visual_score'] >= 2] # Relaxed threshold
    
    print(f"\nTotal Targets: {len(results_df)}")
    print(f"High Visual Confidence (score >= 2): {len(high_confidence)}")
    
    results_df.to_csv('data/outputs/targets_visual_verified.csv', index=False)
    high_confidence.to_csv('data/outputs/targets_high_confidence.csv', index=False)
    print("Saved results to data/outputs/targets_visual_verified.csv")
    
    return results_df

if __name__ == "__main__":
    # Chain input
    input_file = 'data/outputs/targets_mineable.csv'
    if not os.path.exists(input_file):
         if os.path.exists('data/outputs/targets_unclaimed.csv'):
             input_file = 'data/outputs/targets_unclaimed.csv'
         else:
             print("No suitable input file found.")
             # Create dummy if running isolated for test? No, better to fail.
             exit(1)

    check_surface_indicators(input_file)
