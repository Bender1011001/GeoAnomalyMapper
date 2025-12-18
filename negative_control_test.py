# negative_control_test.py

import pandas as pd

# Pick regions with NO known mineralization
# (e.g., Mississippi Delta, Florida peninsula, Kansas plains)

negative_regions = [
    {"name": "Mississippi Delta", "lat_range": (32, 34), "lon_range": (-91, -89)},
    {"name": "Florida Peninsula", "lat_range": (25, 28), "lon_range": (-82, -80)},
    {"name": "Kansas Plains", "lat_range": (38, 40), "lon_range": (-99, -97)},
]

try:
    targets = pd.read_csv('data/outputs/usa_targets.csv')
except Exception as e:
    print(f"Error loading targets: {e}")
    exit(1)

print("\n--- Negative Control Test ---")

for region in negative_regions:
    region_targets = targets[
        (targets['Latitude'] > region['lat_range'][0]) &
        (targets['Latitude'] < region['lat_range'][1]) &
        (targets['Longitude'] > region['lon_range'][0]) &
        (targets['Longitude'] < region['lon_range'][1])
    ]
    
    count = len(region_targets)
    print(f"{region['name']}: {count} targets")
    print(f"  Expected: ~0 (no known mineralization)")
    print(f"  Result: {'PASS' if count < 5 else 'FAIL - overflagging'}")
