# alaska_verification.py

import pandas as pd

# Known major Alaska deposits
alaska_majors = [
    {"name": "Red Dog", "lat": 68.08, "lon": -162.83, "discovered": 1968},
    {"name": "Pebble", "lat": 59.50, "lon": -156.25, "discovered": 1987},
    {"name": "Donlin Gold", "lat": 61.98, "lon": -158.16, "discovered": 1995},
    {"name": "Kensington", "lat": 58.89, "lon": -135.04, "discovered": 1988},
    {"name": "Fort Knox", "lat": 64.99, "lon": -147.42, "discovered": 1992},
]

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
    print(f"Loaded {len(targets)} targets.")
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

print("\n--- Alaska Verification Test ---")
# Check: Did your model flag these independently?
for deposit in alaska_majors:
    # Calculate distance to nearest target
    # 1 degree lat ~ 111 km. Longitude varies but we'll use a simplified metric approximation or haversine if we want to be fancy.
    # The user provided a simplified Euclidean approximation which is decent for short distances but errors grow at high latitudes for longitude.
    # Let's stick closer to the user's logic but maybe slightly robustify the filter.
    
    alaska_targets = targets[(targets['Latitude'] > 50)]  # Alaska only
    
    if len(alaska_targets) == 0:
        print("No targets found in Alaska region.")
        break
        
    # deg to km approximation: 1 deg lat = 111 km. 1 deg lon = 111 * cos(lat).
    # At 60 deg N, cos(60) = 0.5. So 1 deg lon is ~55km.
    # The user's formula: ((dLat)**2 + (dLon)**2)**0.5 * 111
    # This assumes 1 deg lon = 111 km, which artificially inflates longitude distance at high latitudes.
    # Be more precise: adjust longitude difference by cosine of latitude.
    import numpy as np
    
    avg_lat_rad = np.radians(deposit['lat'])
    lon_scale = np.cos(avg_lat_rad)
    
    d_lat = (alaska_targets['Latitude'] - deposit['lat']) * 111
    d_lon = (alaska_targets['Longitude'] - deposit['lon']) * 111 * lon_scale
    
    distances = (d_lat**2 + d_lon**2)**0.5
    
    if len(distances) > 0:
        nearest = distances.min()
        
        if nearest < 10:
            print(f"✓ {deposit['name']}: Target within {nearest:.1f}km (discovered {deposit['discovered']})")
        else:
            print(f"✗ {deposit['name']}: Missed (nearest {nearest:.1f}km)")
    else:
        print(f"✗ {deposit['name']}: No targets to compare")
