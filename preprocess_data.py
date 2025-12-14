import os
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def preprocess_mountain_pass():
    input_dir = "data/mountain_pass"
    grav_gxf = os.path.join(input_dir, "602202_Fourier_GDD_2p67_final.gxf")
    mag_csv = os.path.join(input_dir, "Magnetic_Data.csv")
    
    out_grav = os.path.join(input_dir, "gravity_grad.tif")
    out_mag = os.path.join(input_dir, "magnetic_rmi.tif")
    
    print("1. Processing Gravity...")
    with rasterio.open(grav_gxf) as src:
        grav_data = src.read(1)
        grav_meta = src.profile
        bounds = src.bounds
        res = src.res
        print(f" Loaded Gravity: {grav_data.shape}, Res: {res}")
        
        # Replace NoData with NaN
        if src.nodata:
            grav_data[grav_data == src.nodata] = np.nan
            
        # Update driver to GTiff
        grav_meta.update(driver='GTiff')
        
        # Save as TIF
        with rasterio.open(out_grav, 'w', **grav_meta) as dst:
            dst.write(grav_data, 1)
        print(f" Saved Gravity to {out_grav}")
        
    print("2. Processing Magnetics...")
    # Read CSV
    try:
        df = pd.read_csv(mag_csv)
        print(f" Loaded CSV: {len(df)} rows")
        
        # Identify columns
        cols = df.columns
        col_x = next((c for c in cols if 'Easting' in c), None)
        col_y = next((c for c in cols if 'Northing' in c), None)
        # Prefer RMI, then TMI, then raw
        col_mag = next((c for c in cols if 'mag_RMI' in c or 'Residual' in c), None)
        if not col_mag:
            col_mag = next((c for c in cols if 'mag_tmi' in c or 'TMI' in c), None)
            
        if not (col_x and col_y and col_mag):
            print(f" ERROR: Could not find X, Y, or Mag columns. Found: {cols}")
            return
            
        print(f" Using columns: X={col_x}, Y={col_y}, Mag={col_mag}")
        
        # Create Target Grid coordinates
        # Rasterio convention: y is from top to bottom (max to min), x is left to right
        height, width = grav_data.shape
        trans = grav_meta['transform']
        
        # Create meshgrid of target coordinates
        # Center of pixels
        xs = np.linspace(bounds.left + res[0]/2, bounds.right - res[0]/2, width)
        ys = np.linspace(bounds.top - res[1]/2, bounds.bottom + res[1]/2, height)
        grid_x, grid_y = np.meshgrid(xs, ys)
        
        # Scipy griddata expects points as (N, 2)
        points = df[[col_x, col_y]].values
        values = df[col_mag].values
        
        print(" Gridding (this may take a moment)...")
        # 'linear' is good but leaves NaNs outside hull. 'nearest' fills everything.
        # Let's use linear and fill with 0 or nearest?
        # Geological structures need continuous data.
        # Let's try linear first.
        mag_grid = griddata(points, values, (grid_x, grid_y), method='linear')
        
        # Fill NaNs with nearest to avoid holes at edges if any
        if np.isnan(mag_grid).any():
             print(" Filling NaNs with nearest neighbor...")
             # Mask of valid values
             mask = ~np.isnan(mag_grid)
             # Coordinates of valid values
             valid_coords = np.argwhere(mask)
             valid_values = mag_grid[mask]
             # Coordinates of NaNs
             nan_coords = np.argwhere(~mask)
             
             # Use NearestNDInterpolator or just griddata again for the NaNs
             # Simpler: just run griddata with 'nearest' for the whole grid to get a base
             # then overwrite with linear where valid.
             mag_grid_near = griddata(points, values, (grid_x, grid_y), method='nearest')
             mag_grid[np.isnan(mag_grid)] = mag_grid_near[np.isnan(mag_grid)]
             
        # Save Magnetic Grid
        with rasterio.open(out_mag, 'w', **grav_meta) as dst:
            dst.write(mag_grid.astype(np.float32), 1)
        print(f" Saved Magnetics to {out_mag}")
        
    except Exception as e:
        print(f" Error processing magnetics: {e}")

if __name__ == "__main__":
    preprocess_mountain_pass()
