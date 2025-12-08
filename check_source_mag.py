import rasterio
import numpy as np
from pathlib import Path
import project_paths

mag_path = project_paths.RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
print(f"Checking source: {mag_path}")

if not mag_path.exists():
    print("Source file does not exist!")
else:
    try:
        with rasterio.open(mag_path) as src:
            print(f"  Shape: {src.shape}")
            print(f"  Count: {src.count}")
            print(f"  Dtype: {src.dtypes}")
            print(f"  Nodata: {src.nodata}")
            print(f"  Bounds: {src.bounds}")
            print(f"  CRS: {src.crs}")
            
            # Read a small window around the target region
            # Region: -116.0, 35.0, -115.0, 36.0
            window = rasterio.windows.from_bounds(-116.0, 35.0, -115.0, 36.0, src.transform)
            data = src.read(1, window=window)
            print(f"  Window data shape: {data.shape}")
            print(f"  Window Min: {np.nanmin(data)}")
            print(f"  Window Max: {np.nanmax(data)}")
            print(f"  Window Mean: {np.nanmean(data)}")
            
    except Exception as e:
        print(f"Error reading source: {e}")