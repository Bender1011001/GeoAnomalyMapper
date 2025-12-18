
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import rasterio

def analyze_full_run():
    print("--- FULL USA RUN ANALYSIS ---")
    
    # 1. Load Targets
    target_path = 'data/outputs/usa_targets.csv'
    try:
        targets = pd.read_csv(target_path)
        print(f"Targets Found: {len(targets)}")
    except Exception as e:
        print(f"Error loading targets: {e}")
        return

    # 2. Load Known Deposits (Ground Truth)
    known_path = 'data/usgs_goldilocks.csv'
    try:
        known = pd.read_csv(known_path)
        print(f"Known Deposits: {len(known)}")
    except Exception as e:
        print(f"Error loading known deposits: {e}")
        return

    # Count New vs Known
    if 'Is_Undiscovered' in targets.columns:
        new_targets = targets[targets['Is_Undiscovered'] == True]
        print(f"Undiscovered Targets: {len(new_targets)} ({(len(new_targets)/len(targets))*100:.1f}%)")

    # 3. Calculate Sensitivity (Recall)
    if len(targets) > 0 and len(known) > 0:
        target_coords = targets[['Longitude', 'Latitude']].values
        # FIX: usgs_goldilocks uses 'lon', 'lat'
        known_coords = known[['lon', 'lat']].values
        
        # Build Tree on Targets
        tree = cKDTree(target_coords)
        
        thresholds_km = [5, 10, 25, 50]
        
        print("\n--- SENSITIVITY (Recall) ---")
        for km in thresholds_km:
            deg = km / 111.0
            matches = tree.query_ball_point(known_coords, r=deg)
            detected_count = sum([1 for m in matches if len(m) > 0])
            sensitivity = (detected_count / len(known)) * 100
            print(f"Within {km}km: {detected_count}/{len(known)} Detected ({sensitivity:.1f}%)")

    # 4. Calculate Flagged Area
    # Sum of 'Area_Pixels' in targets / Total Land Pixels
    # We need to know Total Land Pixels.
    # We can estimate it from the mask file if available, or load the supervised prob file.
    
    mask_path = 'data/outputs/usa_land_mask.tif'
    prob_path = 'data/outputs/usa_supervised_probability.tif'
    
    total_pixels = 0
    valid_pixels = 0
    
    try:
        # Try mask first
        r_path = mask_path if  pd.io.common.file_exists(mask_path) else prob_path
        
        with rasterio.open(r_path) as src:
            # We can't read the whole thing into memory likely?
            # It's huge. 
            # Read subsampled
            data = src.read(1, out_shape=(src.height // 10, src.width // 10))
            # Rough estimate of validity percentage
            valid_ratio = np.count_nonzero(data != src.nodata) / data.size
            
            # Real total pixels
            total_pixels = src.width * src.height
            valid_pixels = total_pixels * valid_ratio # approx
            
            # Pixel area
            res_x, res_y = src.res
            # degrees to km2 approx
            # Area of 1 pixel in km2 approx: (0.03 * 111) * (0.03 * 111) approx 3.3 * 3.3 = 10.89 km2
            # Let's be more precise: avg lat 40.
            pixel_area_km2 = (res_x * 111) * (res_y * 111 * 0.76) # Roughly 3.3 km x 2.5 km -> 8.25 km2
            
            flagged_pixels = targets['Area_Pixels'].sum()
            flagged_area_km2 = flagged_pixels * pixel_area_km2
            total_land_area_km2 = valid_pixels * pixel_area_km2
            
            flagged_percent = (flagged_pixels / valid_pixels) * 100
            
            print("\n--- FLAGGED AREA ---")
            print(f"Total Land Area (Approx): {total_land_area_km2:,.0f} km2")
            print(f"Flagged Target Area: {flagged_area_km2:,.0f} km2")
            print(f"Flagged Percentage: {flagged_percent:.4f}%")
            
    except Exception as e:
        print(f"Could not calculate area: {e}")

if __name__ == "__main__":
    analyze_full_run()
