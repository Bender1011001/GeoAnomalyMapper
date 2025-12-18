# deposit_type_analysis.py

import pandas as pd

# Your 137 geochem-validated targets
try:
    validated = pd.read_csv('data/outputs/usa_targets_geochem_validated.csv')
    print(f"Loaded {len(validated)} validated targets.")
except Exception as e:
    print(f"Error loading validated targets: {e}")
    # If file doesn't exist, we can't run this test.
    print("Ensure 'data/outputs/usa_targets_geochem_validated.csv' exists.")
    exit(1)

print("\n--- Deposit Type Analysis ---")

# Analyze Cu vs Zn signatures
# Assuming columns 'Cu_ppm' and 'Zn_ppm' exist. If not, we might need to be flexible or check columns first.
# The user prompt implies these columns exist in the 'validated' dataset.

cols = validated.columns
has_cu = 'Cu_ppm' in cols
has_zn = 'Zn_ppm' in cols

if not has_cu or not has_zn:
    print(f"Warning: Cu_ppm or Zn_ppm columns missing. Available: {cols}")
    # Try to proceed if at least one exists or exit? Let's be robust.
    
if has_cu:
    high_cu = validated[validated['Cu_ppm'] > 100]
    print(f"High Cu targets (>100ppm): {len(high_cu)}")
    
    # Check if they're in the right geological settings
    # High Cu should be in: 
    # - Southwest (Arizona, New Mexico) → Porphyry copper
    # - Nevada → Sediment-hosted copper
    
    cu_sw = high_cu[(high_cu['Latitude'] < 37) & (high_cu['Longitude'] > -115)]
    sw_ratio = len(cu_sw) / len(high_cu) if len(high_cu) > 0 else 0
    print(f"Cu targets in SW (expected for porphyry): {len(cu_sw)}/{len(high_cu)} ({sw_ratio:.1%})")
    
    if sw_ratio > 0.6:
        print("✓ PASS: Majority of Cu targets are in the Southwest.")
    else:
        print("? NOTE: Cu targets scattered outside SW.")

if has_zn:
    high_zn = validated[validated['Zn_ppm'] > 300]
    print(f"High Zn targets (>300ppm): {len(high_zn)}")
    
    # High Zn should be in:
    # - Alaska → VMS
    # - Tennessee/Missouri → MVT
    
    zn_alaska = high_zn[high_zn['Latitude'] > 50]
    zn_midest = high_zn[(high_zn['Latitude'] > 35) & (high_zn['Latitude'] < 38) & 
                        (high_zn['Longitude'] > -92) & (high_zn['Longitude'] < -82)] # Rough TN/MO box
    
    print(f"Zn targets in Alaska: {len(zn_alaska)}")
    print(f"Zn targets in TN/MO (MVT belt): {len(zn_midest)}")

# General check
print("Deposit type verification complete.")
