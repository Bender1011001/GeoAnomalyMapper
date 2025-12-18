import rasterio
import numpy as np

path = 'data/raw/magnetic/EMAG2_V3_20170530_Sealevel.tif'

with rasterio.open(path) as src:
    print(f"Shape: {src.shape}")
    print(f"Bounds: {src.bounds}")
    
    # Read in chunks to avoid memory kill if huge
    # actually 200MB is fine
    data = src.read(1)
    
    # Check for valid data ( > -10000 usually, since min is -3e38)
    valid_mask = data > -10000
    
    if np.any(valid_mask):
        print("FOUND VALID DATA!")
        rows, cols = np.where(valid_mask)
        print(f"Valid Rows: {rows.min()} to {rows.max()}")
        print(f"Valid Cols: {cols.min()} to {cols.max()}")
        
        # Sample a valid value
        r, c = rows[0], cols[0]
        val = data[r, c]
        # Get coord
        lon, lat = src.xy(r, c)
        print(f"Sample Valid Pixel: ({r}, {c}) -> ({lat}, {lon}) = {val}")
    else:
        print("NO VALID DATA FOUND IN ENTIRE FILE.")
