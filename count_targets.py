import rasterio
import numpy as np
from scipy.ndimage import label

def count_targets(tif_path, threshold=1.5):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        # Handle nodata
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)
            
        # Threshold
        binary_mask = (data >= threshold)
        
        # Label connected components
        # structure=np.ones((3,3)) defines connectivity (diagonals included)
        labeled_array, num_features = label(binary_mask, structure=np.ones((3,3)))
        
        # Calculate Area Percentage
        total_valid_pixels = np.sum(~np.isnan(data))
        flagged_pixels = np.sum(binary_mask)
        area_percent = (flagged_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
        print(f"Total Targets Marked: {num_features}")
        print(f"Total Area Flagged: {area_percent:.2f}%")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()
    count_targets("data/outputs/usa_density_model.tif", threshold=args.threshold)
