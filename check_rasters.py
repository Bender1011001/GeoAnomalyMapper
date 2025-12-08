import rasterio
import numpy as np
import sys

files = [
    "outputs/mountain_pass_rare_earths_gravity_residual.tif",
    "outputs/mountain_pass_rare_earths_fused_belief_reinforced.tif",
    "outputs/mountain_pass_rare_earths_poisson_correlation.tif",
    "outputs/mountain_pass_rare_earths_structural_artificiality.tif",
    "outputs/mountain_pass_rare_earths_density_model.tif"
]

for f in files:
    try:
        with rasterio.open(f) as src:
            data = src.read(1)
            print(f"File: {f}")
            print(f"  Shape: {data.shape}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Min: {np.nanmin(data)}")
            print(f"  Max: {np.nanmax(data)}")
            print(f"  NaN count: {np.isnan(data).sum()}")
            print(f"  Total pixels: {data.size}")
            print("-" * 20)
    except Exception as e:
        print(f"Error reading {f}: {e}")