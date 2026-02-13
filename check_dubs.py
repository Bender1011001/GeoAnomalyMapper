"""
DUB Detection Analysis - Check void signals at known DUB locations
"""
import numpy as np
import pandas as pd
import rasterio

# Load known DUBs
dubs = pd.read_csv('data/known_dubs.csv')
usa_dubs = dubs[(dubs['lat'] >= 25) & (dubs['lat'] <= 49) & (dubs['lon'] >= -124) & (dubs['lon'] <= -67)]

print(f"USA DUBs: {len(usa_dubs)}")

# Check void signal at each DUB
print("\nVoid signal at each DUB location:")
with rasterio.open('data/outputs/dub_detection/pinn_void_density.tif') as src:
    data = src.read(1)
    height, width = data.shape
    
    # Get value distribution in continental USA
    print(f"\nData shape: {data.shape}")
    print(f"Value range: {np.nanmin(data):.1f} to {np.nanmax(data):.1f}")
    
    # Sample some points in USA to understand distribution
    valid_vals = []
    for _, row in usa_dubs.iterrows():
        r, c = src.index(row['lon'], row['lat'])
        if 0 <= r < height and 0 <= c < width:
            val = data[r, c]
            void = -val  # Negative density = void
            valid_vals.append(void)
            print(f"  {row['name'][:30]:30s} density={val:8.1f} void_signal={void:8.1f}")

print(f"\nVoid signals at DUBs: min={min(valid_vals):.1f}, max={max(valid_vals):.1f}, mean={np.mean(valid_vals):.1f}")

# What percentile are the DUB void signals?
# Sample continental USA
usa_samples = []
for lat in np.linspace(26, 48, 50):
    for lon in np.linspace(-122, -70, 100):
        r, c = src.index(lon, lat)
        if 0 <= r < height and 0 <= c < width:
            val = data[r, c]
            if not np.isnan(val) and abs(val) < 700:  # Exclude nodata and edge artifacts
                usa_samples.append(-val)

usa_samples = np.array(usa_samples)
print(f"\nUSA sample distribution (n={len(usa_samples)}):")
print(f"  p50: {np.percentile(usa_samples, 50):.1f}")
print(f"  p75: {np.percentile(usa_samples, 75):.1f}")
print(f"  p90: {np.percentile(usa_samples, 90):.1f}")
print(f"  p95: {np.percentile(usa_samples, 95):.1f}")
print(f"  p99: {np.percentile(usa_samples, 99):.1f}")

# Percentile of DUB void signals
for void in valid_vals:
    pct = (usa_samples < void).mean() * 100
    print(f"  DUB void_signal {void:.1f} is at percentile {pct:.1f}")
