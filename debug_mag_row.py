import rasterio
import numpy as np

path = 'data/raw/magnetic/EMAG2_V3_20170530_Sealevel.tif'

with rasterio.open(path) as src:
    print(f"Checking Row 1500 (approx 40N)...")
    
    # Read row 1500
    row_data = src.read(1, window=((1500, 1501), (0, src.width)))
    row_data = row_data[0, :]
    
    valid_mask = row_data > -10000
    
    if np.any(valid_mask):
        cols = np.where(valid_mask)[0]
        print(f"Valid Cols at Row 1500: {cols.min()} to {cols.max()}")
        
        # Check specific points
        for c in [0, 5400, 7800, 10000]:
            val = row_data[c] if c < len(row_data) else -9999
            lon, lat = src.xy(1500, c)
            print(f"Col {c} (Lon {lon:.2f}): {val}")
    else:
        print("Row 1500 is entirely empty.")
