import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
import argparse
from pathlib import Path

def align_raster(src_path, ref_path, dst_path):
    print(f"Aligning {src_path} to match {ref_path}...")
    
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_profile = ref.profile.copy()

    with rasterio.open(src_path) as src:
        dst_profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'count': 1 # Enforce single band for features
        })

        with rasterio.open(dst_path, 'w', **dst_profile) as dst:
            for i in range(1, 2): # Just band 1
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
    print(f"Saved aligned raster to {dst_path}")

if __name__ == "__main__":
    # Reference is Gravity
    ref = "data/outputs/usa_supervised/usa_gravity_mosaic.tif"
    
    # Align Magnetic
    align_raster(
        "data/outputs/usa_supervised/usa_magnetic_mosaic.tif",
        ref,
        "data/outputs/usa_supervised/usa_magnetic_aligned.tif"
    )
    
    # Align Density (Just in case, though likely matches Gravity)
    # Check if density exists
    density = "data/outputs/usa_density_model.tif"
    if Path(density).exists():
        align_raster(
            density,
            ref,
            "data/outputs/usa_density_aligned.tif"
        )
