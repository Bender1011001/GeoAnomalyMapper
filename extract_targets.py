import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, label, center_of_mass
import argparse
from pathlib import Path

def extract_targets(tif_path, threshold=50.0, min_distance=5):
    """
    Extracts targets (local maxima) from a density map GeoTIFF.
    
    Args:
        tif_path (str): Path to the GeoTIFF file.
        threshold (float): Minimum density contrast to consider a target.
        min_distance (int): Minimum distance (in pixels) between peaks.
        
    Returns:
        pd.DataFrame: DataFrame containing Lat, Lon, and Value of targets.
    """
    print(f"Processing {tif_path}...")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        
        # Handle nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
            
        # Print Statistics
        valid_data = data[~np.isnan(data)]
        print(f"Data Statistics:")
        print(f"  Min: {np.min(valid_data):.4f}")
        print(f"  Max: {np.max(valid_data):.4f}")
        print(f"  Mean: {np.mean(valid_data):.4f}")
        print(f"  Std: {np.std(valid_data):.4f}")
        
        # 1. Thresholding
        # If threshold is default 50, but data range is different, warn user
        if threshold == 50.0 and np.max(valid_data) < 50.0:
            print("Warning: Max value is below threshold 50.0")
            
        mask = data > threshold
        
        # Handle NaN in mask (False)
        mask = np.nan_to_num(mask, nan=False)
        
        if not np.any(mask):
            print("No targets found above threshold.")
            return pd.DataFrame()
            
        # 2. Find Distinct Anomalous Regions (Connected Components)
        labeled_array, num_features = label(mask)
        print(f"Found {num_features} distinct anomalous regions. Optimizing extraction...")
        
        from scipy.ndimage import find_objects
        objects = find_objects(labeled_array)
        
        target_list = []
        
        for i, sl in enumerate(objects, start=1):
            if sl is None: continue
            
            # Slice the valid data/mask using bounding box
            # sl is a tuple of slices (slice(row_min, row_max), slice(col_min, col_max))
            
            # Extract local component mask and data
            local_mask = (labeled_array[sl] == i)
            local_data = data[sl]
            
            # Apply mask to data (get values for this component only)
            # Make sure to handle NaNs or other region spillover
            # local_data[~local_mask] = -np.inf # Ignore other regions in box
            
            comp_vals = local_data[local_mask]
            if comp_vals.size == 0: continue
            
            max_val = np.max(comp_vals)
            
            # Find location of max value relative to slice
            # argmax returns flat index, unravel gives local (r, c)
            max_idx_flat = np.argmax(local_data * local_mask) # masking ensures we pick from correct region
            # Wait, local_data * local_mask zero-out others? 
            # If max_val is > 0, yes. If data can be negative? Density contrast usually positive here.
            # Safer: 
            local_coords = np.argwhere(local_mask & (local_data == max_val))
            
            if len(local_coords) > 0:
                local_r, local_c = local_coords[0]
                
                # Convert to global (r, c)
                # slice.start gives offset
                global_r = sl[0].start + local_r
                global_c = sl[1].start + local_c
                
                # Convert to native CRS coordinates
                x_native, y_native = rasterio.transform.xy(transform, global_r, global_c, offset='center')
                
                # Transform to Lat/Lon (EPSG:4326) if needed
                if src.crs and src.crs != 'EPSG:4326':
                    from rasterio.warp import transform as transform_coords
                    xs, ys = transform_coords(src.crs, 'EPSG:4326', [x_native], [y_native])
                    lon, lat = xs[0], ys[0]
                else:
                    lon, lat = x_native, y_native
                
                target_list.append({
                    'Region_ID': i,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Density_Contrast': max_val,
                    'Area_Pixels': np.sum(local_mask)
                })
        
        # Create DataFrame
        targets = pd.DataFrame(target_list)
        
        # Sort by value descending
        if not targets.empty:
            targets = targets.sort_values('Density_Contrast', ascending=False).reset_index(drop=True)
        
        return targets

def main():
    parser = argparse.ArgumentParser(description="Extract targets from Density Map")
    parser.add_argument("input_tif", help="Path to input GeoTIFF")
    parser.add_argument("--threshold", type=float, default=50.0, help="Density threshold (default: 50.0)")
    parser.add_argument("--output", default="extracted_targets.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    if not Path(args.input_tif).exists():
        print(f"Error: File {args.input_tif} not found.")
        return

    targets = extract_targets(args.input_tif, args.threshold)
    
    if not targets.empty:
        # Load known deposits to filter
        usgs_csv = Path('data/usgs_goldilocks.csv')
        known_df = None
        if usgs_csv.exists():
            try:
                known_df = pd.read_csv(usgs_csv)
                # Ensure lat/lon columns
                if 'latitude' not in known_df.columns and 'lat' in known_df.columns:
                    known_df.rename(columns={'lat': 'latitude'}, inplace=True)
                if 'longitude' not in known_df.columns and 'lon' in known_df.columns:
                    known_df.rename(columns={'lon': 'longitude'}, inplace=True)
            except Exception as e:
                print(f"Warning: Could not load known deposits for filtering: {e}")

        if known_df is not None:
            print(f"\nFiltering against {len(known_df)} known USGS deposits...")
            # Simple distance filter (e.g. 1km ~ 0.01 degrees)
            # If a target is within X distance of a known deposit, mark it as "Known"
            
            # Vectorized distance check would be better, but loop is fine for extraction list
            is_new = []
            nearest_known = []
            dist_to_known = []
            
            known_coords = known_df[['latitude', 'longitude']].values
            
            for idx, row in targets.iterrows():
                target_lat = row['Latitude']
                target_lon = row['Longitude']
                
                # Euclidean distance (approx degrees)
                # dy = target_lat - known_coords[:, 0]
                # dx = (target_lon - known_coords[:, 1]) * np.cos(np.deg2rad(target_lat))
                # dists = np.sqrt(dy**2 + dx**2)
                
                # Simple degree distance
                dists = np.sqrt((target_lat - known_coords[:,0])**2 + (target_lon - known_coords[:,1])**2)
                
                min_dist = np.min(dists)
                min_idx = np.argmin(dists)
                
                # Threshold: 0.02 degrees (approx 2.2km)
                # If further than this, it's "New"
                threshold_deg = 0.02
                
                is_new.append(min_dist > threshold_deg)
                nearest_known.append(f"Deposit_{min_idx}") # Placeholder name if 'name' missing
                dist_to_known.append(min_dist * 111.0) # Approx km
                
            targets['Is_Undiscovered'] = is_new
            targets['Dist_to_Known_km'] = dist_to_known
            
            # Filter
            undiscovered = targets[targets['Is_Undiscovered']]
            print(f"Found {len(undiscovered)} POTENTIALLY NEW targets (>{0.02*111:.1f} km from known sites).")
            
            if not undiscovered.empty:
                print(f"Top 20 Undiscovered Targets:")
                print(undiscovered.head(20).to_string(index=False))
                
                # Save just the new ones
                new_output = Path(args.output).with_name("undiscovered_targets.csv")
                undiscovered.to_csv(new_output, index=False)
                print(f"\nUndiscovered list saved to {new_output}")
                
        else:
            print(f"\nFound {len(targets)} targets (No known deposit filtering applied).")
            print(f"Top 20 Targets:")
            print(targets.head(20).to_string(index=False))
        
        targets.to_csv(args.output, index=False)
        print(f"Full list saved to {args.output}")
    else:
        print("No targets found.")

if __name__ == "__main__":
    main()
