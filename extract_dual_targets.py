#!/usr/bin/env python3
"""
Dual-Pipeline Target Extraction: Mass Excess AND Mass Deficit

Based on geophysical research:
- POSITIVE anomalies (highs) → VMS, IOCG, Ni-Cu, Skarns
- NEGATIVE anomalies (lows)  → Carlin Gold, Kimberlites, Epithermal Au

This script extracts BOTH types from the density model.
"""

import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects, minimum_filter, maximum_filter
import argparse
from pathlib import Path


def extract_dual_targets(tif_path, high_threshold=0.5, low_threshold=-0.5, min_area=1):
    """
    Extracts both HIGH (dense) and LOW (light) density anomalies.
    
    Args:
        tif_path: Path to density contrast GeoTIFF
        high_threshold: Minimum value for mass-excess targets (default 0.5)
        low_threshold: Maximum value for mass-deficit targets (default -0.5)
        min_area: Minimum area in pixels to consider a target
        
    Returns:
        Tuple of (high_targets_df, low_targets_df)
    """
    print(f"Processing {tif_path} for DUAL-PIPELINE extraction...")
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        
        # Handle nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
            
        # Print statistics
        valid_data = data[~np.isnan(data)]
        print(f"\nData Statistics:")
        print(f"  Min: {np.min(valid_data):.4f}")
        print(f"  Max: {np.max(valid_data):.4f}")
        print(f"  Mean: {np.mean(valid_data):.4f}")
        print(f"  Std: {np.std(valid_data):.4f}")
        
        # Calculate percentiles for adaptive thresholding
        p05 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        print(f"  5th percentile: {p05:.4f}")
        print(f"  95th percentile: {p95:.4f}")
        
        # =====================
        # PIPELINE A: MASS EXCESS (High Density → VMS, IOCG, Ni-Cu)
        # =====================
        print(f"\n{'='*60}")
        print("PIPELINE A: MASS EXCESS (VMS, IOCG, Ni-Cu, Skarn targets)")
        print('='*60)
        
        # Use adaptive threshold if user threshold seems wrong
        if high_threshold > np.max(valid_data):
            high_threshold = p95
            print(f"  Adjusted high_threshold to 95th percentile: {high_threshold:.4f}")
        
        high_mask = data > high_threshold
        high_mask = np.nan_to_num(high_mask, nan=False).astype(bool)
        
        high_targets = _extract_from_mask(data, high_mask, transform, src.crs, 
                                          target_type='HIGH', min_area=min_area)
        print(f"  Found {len(high_targets)} MASS EXCESS targets")
        
        # =====================
        # PIPELINE B: MASS DEFICIT (Low Density → Carlin, Kimberlite, Epithermal)
        # =====================
        print(f"\n{'='*60}")
        print("PIPELINE B: MASS DEFICIT (Carlin Gold, Kimberlite, Epithermal)")
        print('='*60)
        
        # For deficit, we look for values BELOW a negative threshold
        if low_threshold < np.min(valid_data):
            low_threshold = p05
            print(f"  Adjusted low_threshold to 5th percentile: {low_threshold:.4f}")
        
        low_mask = data < low_threshold
        low_mask = np.nan_to_num(low_mask, nan=False).astype(bool)
        
        low_targets = _extract_from_mask(data, low_mask, transform, src.crs,
                                         target_type='LOW', min_area=min_area,
                                         find_minima=True)
        print(f"  Found {len(low_targets)} MASS DEFICIT targets")
        
        return high_targets, low_targets


def _extract_from_mask(data, mask, transform, crs, target_type='HIGH', min_area=1, find_minima=False):
    """
    Extract targets from a binary mask.
    
    For HIGH targets: find the maximum value in each region.
    For LOW targets: find the minimum value in each region.
    """
    if not np.any(mask):
        return pd.DataFrame()
    
    labeled_array, num_features = label(mask)
    objects = find_objects(labeled_array)
    
    target_list = []
    
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
            
        local_mask = (labeled_array[sl] == i)
        local_data = data[sl].copy()
        
        # Get area
        area = np.sum(local_mask)
        if area < min_area:
            continue
        
        # Get values for this component
        comp_vals = local_data[local_mask]
        if comp_vals.size == 0:
            continue
        
        # Find extremum
        if find_minima:
            extremum_val = np.min(comp_vals)
            local_coords = np.argwhere(local_mask & (local_data == extremum_val))
        else:
            extremum_val = np.max(comp_vals)
            local_coords = np.argwhere(local_mask & (local_data == extremum_val))
        
        if len(local_coords) == 0:
            continue
            
        local_r, local_c = local_coords[0]
        global_r = sl[0].start + local_r
        global_c = sl[1].start + local_c
        
        # Convert to coordinates
        x_native, y_native = rasterio.transform.xy(transform, global_r, global_c, offset='center')
        
        # Transform to Lat/Lon if needed
        if crs and str(crs) != 'EPSG:4326':
            from rasterio.warp import transform as transform_coords
            xs, ys = transform_coords(crs, 'EPSG:4326', [x_native], [y_native])
            lon, lat = xs[0], ys[0]
        else:
            lon, lat = x_native, y_native
        
        target_list.append({
            'Region_ID': i,
            'Latitude': lat,
            'Longitude': lon,
            'Density_Contrast': extremum_val,
            'Area_Pixels': area,
            'Pixel_X': global_c,
            'Pixel_Y': global_r,
            'Target_Type': target_type,
            'Deposit_Class': _classify_deposit(target_type, extremum_val)
        })
    
    targets = pd.DataFrame(target_list)
    
    if not targets.empty:
        # Sort by absolute value of contrast (most extreme first)
        targets['Abs_Contrast'] = targets['Density_Contrast'].abs()
        targets = targets.sort_values('Abs_Contrast', ascending=False).reset_index(drop=True)
        targets = targets.drop('Abs_Contrast', axis=1)
    
    return targets


def _classify_deposit(target_type, value):
    """
    Classify probable deposit type based on gravity signature.
    """
    if target_type == 'HIGH':
        if value > 100:
            return 'IOCG_or_Massive_Sulfide'
        elif value > 50:
            return 'VMS_or_Skarn'
        elif value > 10:
            return 'Ni_Cu_or_Mafic_Intrusion'
        else:
            return 'Weak_Excess'
    else:  # LOW
        if value < -100:
            return 'Kimberlite_or_Major_Alteration'
        elif value < -50:
            return 'Carlin_Style_or_Epithermal'
        elif value < -10:
            return 'Alteration_Halo_or_Silicification'
        else:
            return 'Weak_Deficit'


def main():
    parser = argparse.ArgumentParser(description="Dual-Pipeline Target Extraction")
    parser.add_argument("input_tif", help="Path to density contrast GeoTIFF")
    parser.add_argument("--high-threshold", type=float, default=0.5,
                        help="Threshold for mass-excess targets (default: 0.5)")
    parser.add_argument("--low-threshold", type=float, default=-0.5,
                        help="Threshold for mass-deficit targets (default: -0.5)")
    parser.add_argument("--output-prefix", default="dual_targets",
                        help="Prefix for output CSV files")
    parser.add_argument("--output-dir", default="data/outputs",
                        help="Output directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_tif)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract both types
    high_targets, low_targets = extract_dual_targets(
        args.input_tif,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold
    )
    
    # Save results
    if not high_targets.empty:
        high_path = output_dir / f"{args.output_prefix}_high.csv"
        high_targets.to_csv(high_path, index=False)
        print(f"\nMASS EXCESS targets saved to {high_path}")
        print(f"Top 10 HIGH targets:")
        print(high_targets.head(10)[['Latitude', 'Longitude', 'Density_Contrast', 'Deposit_Class']].to_string(index=False))
    
    if not low_targets.empty:
        low_path = output_dir / f"{args.output_prefix}_low.csv"
        low_targets.to_csv(low_path, index=False)
        print(f"\nMASS DEFICIT targets saved to {low_path}")
        print(f"Top 10 LOW targets:")
        print(low_targets.head(10)[['Latitude', 'Longitude', 'Density_Contrast', 'Deposit_Class']].to_string(index=False))
    
    # Combined summary
    print(f"\n{'='*60}")
    print("DUAL-PIPELINE SUMMARY")
    print('='*60)
    print(f"  Mass Excess (VMS/IOCG/NiCu): {len(high_targets)} targets")
    print(f"  Mass Deficit (Carlin/Kimberlite): {len(low_targets)} targets")
    print(f"  TOTAL EXPLORATION TARGETS: {len(high_targets) + len(low_targets)}")
    
    # Combine and save master list
    if not high_targets.empty or not low_targets.empty:
        all_targets = pd.concat([high_targets, low_targets], ignore_index=True)
        all_path = output_dir / f"{args.output_prefix}_all.csv"
        all_targets.to_csv(all_path, index=False)
        print(f"\nCombined target list saved to {all_path}")


if __name__ == "__main__":
    main()
