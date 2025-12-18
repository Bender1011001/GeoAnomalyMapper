# alaska_analysis.py

import pandas as pd
import numpy as np

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

# Get all Alaska targets
alaska = targets[targets['Latitude'] > 55]

print(f"Alaska targets found: {len(alaska)}")
if len(alaska) == 0:
    print("No Alaska targets found.")
    exit()

print("\nTop 10 Alaska targets by density contrast:")
print(alaska.nlargest(10, 'Density_Contrast')[['Latitude', 'Longitude', 'Density_Contrast']])

# Check against all USGS MRDS Alaska entries
try:
    # Corrected path to the file found in data directory
    mrds_path = 'data/usgs_mrds_full.csv'
    mrds = pd.read_csv(mrds_path, low_memory=False)
    
    # Filter for Alaska approximated by lat/lon
    alaska_mrds = mrds[(mrds['latitude'] > 55) & (mrds['latitude'] < 72)]
    
    print(f"\nKnown Alaska deposits in MRDS: {len(alaska_mrds)}")
    
    # For each of your Alaska targets, find nearest MRDS entry
    print("\nChecking Top 10 High-Density Targets against MRDS:")
    for idx, target in alaska.nlargest(10, 'Density_Contrast').iterrows():
        # Cosine correction
        avg_lat_rad = np.radians(target['Latitude'])
        lon_scale = np.cos(avg_lat_rad)
        
        d_lat = (alaska_mrds['latitude'] - target['Latitude']) * 111
        d_lon = (alaska_mrds['longitude'] - target['Longitude']) * 111 * lon_scale
        
        dists = (d_lat**2 + d_lon**2)**0.5
        
        nearest_dist = dists.min()
        nearest_mine = alaska_mrds.iloc[dists.argmin()]
        
        print(f"\nTarget at {target['Latitude']:.2f}, {target['Longitude']:.2f} (Density: {target['Density_Contrast']:.2f})")
        print(f"  Nearest known mine: {nearest_mine.get('site_name', 'N/A')} ({nearest_dist:.1f}km)")
        print(f"  Commodity: {nearest_mine.get('commod1', 'N/A')}")
        print(f"  Dep Type: {nearest_mine.get('dep_type', 'N/A')}")
        
except Exception as e:
    print(f"MRDS data analysis failed: {e}")
