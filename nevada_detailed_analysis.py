# nevada_detailed_analysis.py

import pandas as pd
import numpy as np
from collections import Counter

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

nevada = targets[(targets['Latitude'] > 35) & 
                 (targets['Latitude'] < 42) & 
                 (targets['Longitude'] > -120) & 
                 (targets['Longitude'] < -114)]

print(f"Nevada targets: {len(nevada)}")

if len(nevada) == 0:
    exit()

# Load comprehensive Nevada mine database
try:
    # Corrected path
    mrds_path = 'data/usgs_mrds_full.csv'
    mrds = pd.read_csv(mrds_path, low_memory=False)
    
    nevada_mrds = mrds[(mrds['latitude'] > 35) & 
                       (mrds['latitude'] < 42) & 
                       (mrds['longitude'] > -120) & 
                       (mrds['longitude'] < -114)]
    
    print(f"Known Nevada deposits: {len(nevada_mrds)}")
    
    # For each target, what's it near?
    matches_within_5km = 0
    matches_within_10km = 0
    
    closest_commodities = []
    
    print("\nAnalyzing proximity...")
    for idx, target in nevada.iterrows():
        # Cosine correction
        avg_lat_rad = np.radians(target['Latitude'])
        lon_scale = np.cos(avg_lat_rad)
        
        d_lat = (nevada_mrds['latitude'] - target['Latitude']) * 111
        d_lon = (nevada_mrds['longitude'] - target['Longitude']) * 111 * lon_scale
        
        dists = (d_lat**2 + d_lon**2)**0.5
        
        nearest_dist = dists.min()
        
        if nearest_dist < 5:
            matches_within_5km += 1
        if nearest_dist < 10:
            matches_within_10km += 1
            
        # Collect commodity info for nearby matches
        if nearest_dist < 10:
            nearest_mine = nevada_mrds.iloc[dists.argmin()]
            commod = nearest_mine.get('commod1', 'Unknown')
            if pd.notna(commod):
                closest_commodities.append(commod)

    print(f"\nNevada targets within 5km of known deposit: {matches_within_5km}/{len(nevada)} ({matches_within_5km/len(nevada)*100:.1f}%)")
    print(f"Nevada targets within 10km of known deposit: {matches_within_10km}/{len(nevada)} ({matches_within_10km/len(nevada)*100:.1f}%)")
    
    print("\nCommodities near your Nevada targets (within 10km):")
    for commodity, count in Counter(closest_commodities).most_common(10):
        print(f"  {commodity}: {count}")
        
except Exception as e:
    print(f"MRDS data analysis failed: {e}")
