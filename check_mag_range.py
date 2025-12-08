import rasterio
import numpy as np
from pathlib import Path
import project_paths

def check_mag_range():
    path = project_paths.RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff_Float.tif"
    if not path.exists():
        print(f"File not found: {path}")
        return

    with rasterio.open(path) as src:
        data = src.read(1)
        # Mask nodata if present
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)
        
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        
        print(f"File: {path.name}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        
        if 0 <= min_val and max_val <= 255:
            print("WARNING: Data range suggests 8-bit image data (0-255). Likely uncalibrated proxy.")
        else:
            print("Data range looks like physical units (nT).")

if __name__ == "__main__":
    check_mag_range()