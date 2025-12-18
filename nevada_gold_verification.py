# nevada_gold_verification.py

import pandas as pd
import numpy as np

# Major Nevada gold deposits (NOT in MRDS training set)
# Focus on post-2000 discoveries or recent expansions

nevada_modern = [
    {"name": "Long Canyon", "lat": 41.18, "lon": -116.51, "discovered": 2006},
    {"name": "Goldrush", "lat": 40.75, "lon": -116.42, "discovered": 2015},
    {"name": "Fourmile", "lat": 40.95, "lon": -117.23, "discovered": 2017},
]

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

nevada_targets = targets[(targets['Latitude'] > 35) & 
                         (targets['Latitude'] < 42) & 
                         (targets['Longitude'] > -120) & 
                         (targets['Longitude'] < -114)]

print(f"\n--- Nevada Gold Test ---")
print(f"Total Nevada targets: {len(nevada_targets)}")

if len(nevada_targets) == 0:
    print("No targets found in Nevada box.")
    exit()

for deposit in nevada_modern:
    avg_lat_rad = np.radians(deposit['lat'])
    lon_scale = np.cos(avg_lat_rad)
    
    d_lat = (nevada_targets['Latitude'] - deposit['lat']) * 111
    d_lon = (nevada_targets['Longitude'] - deposit['lon']) * 111 * lon_scale
    
    distances = (d_lat**2 + d_lon**2)**0.5
    
    nearest = distances.min()
    nearest_idx = distances.argmin()
    nearest_target = nevada_targets.iloc[nearest_idx]
    
    print(f"\n{deposit['name']} (discovered {deposit['discovered']}):")
    print(f"  Nearest target: {nearest:.1f}km away")
    # Use .get() locally or just access columns safely if they exist
    print(f"  Target density: {nearest_target.get('Density_Contrast', 'N/A')}")
    print(f"  Geochem validated: {nearest_target.get('Geochem_Validated', False)}")
