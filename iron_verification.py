# iron_verification.py

import pandas as pd
import numpy as np

# Major USA iron deposits (should be EASY to detect)
iron_deposits = [
    {"name": "Mesabi Range (MN)", "lat": 47.4, "lon": -92.5, "type": "Banded Iron"},
    {"name": "Marquette Range (MI)", "lat": 46.5, "lon": -87.6, "type": "Banded Iron"},
    {"name": "Eagle Mountain (CA)", "lat": 33.9, "lon": -115.4, "type": "Skarn"},
    {"name": "Iron Mountain (MO)", "lat": 37.6, "lon": -90.6, "type": "IOCG"},
]

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
    print(f"Loaded {len(targets)} targets.")
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

print("\nIRON DEPOSIT VERIFICATION")
print("="*60)

for deposit in iron_deposits:
    # Use cosine correction for longitude distance
    avg_lat_rad = np.radians(deposit['lat'])
    lon_scale = np.cos(avg_lat_rad)
    
    d_lat = (targets['Latitude'] - deposit['lat']) * 111
    d_lon = (targets['Longitude'] - deposit['lon']) * 111 * lon_scale
    
    distances = (d_lat**2 + d_lon**2)**0.5
    
    nearest_dist = distances.min()
    nearest_idx = distances.argmin()
    nearest_target = targets.iloc[nearest_idx]
    
    status = "✓ FOUND" if nearest_dist < 10 else "✗ MISSED"
    
    print(f"\n{deposit['name']}")
    print(f"  {status}: Nearest target {nearest_dist:.1f}km away")
    if nearest_dist < 20:
        print(f"  Target density: {nearest_target.get('Density_Contrast', 'N/A')}")
        print(f"  Coords: {nearest_target['Latitude']:.3f}, {nearest_target['Longitude']:.3f}")
