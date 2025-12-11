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
        print(f"Found {num_features} distinct anomalous regions.")
        
        target_list = []
        
        for i in range(1, num_features + 1):
            # Get indices for this component
            component_mask = labeled_array == i
            
            # Find max value in this component
            component_data = data[component_mask]
            max_val = np.max(component_data)
            
            # Find location of max value
            # We need the index within the full array
            # np.where returns indices for the masked array, but we want indices in 'data'
            # So we find where (mask is true) AND (data == max_val)
            # This might find multiple pixels if max is repeated, take the first
            coords = np.argwhere(component_mask & (data == max_val))
            
            if len(coords) > 0:
                r, c = coords[0]
                
                # Convert to Lat/Lon
                lon, lat = rasterio.transform.xy(transform, r, c, offset='center')
                
                target_list.append({
                    'Region_ID': i,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Density_Contrast': max_val,
                    'Area_Pixels': np.sum(component_mask)
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
        print(f"\nFound {len(targets)} targets.")
        print(f"Top 20 Targets:")
        print(targets.head(20).to_string(index=False))
        
        targets.to_csv(args.output, index=False)
        print(f"\nFull list saved to {args.output}")
    else:
        print("No targets found.")

if __name__ == "__main__":
    main()
