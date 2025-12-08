import rasterio
import numpy as np

f = "outputs/mountain_pass_rare_earths_magnetic.tif"
try:
    with rasterio.open(f) as src:
        data = src.read(1)
        print(f"File: {f}")
        print(f"  Shape: {data.shape}")
        print(f"  Min: {np.nanmin(data)}")
        print(f"  Max: {np.nanmax(data)}")
        print(f"  NaN count: {np.isnan(data).sum()}")
        print(f"  Total pixels: {data.size}")
        print(f"  Unique values: {len(np.unique(data))}")
except Exception as e:
    print(f"Error reading {f}: {e}")