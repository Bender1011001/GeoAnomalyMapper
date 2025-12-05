import sys
import os
from pathlib import Path
import rasterio
import numpy as np
import time

# Ensure current directory is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from insar_features import extract_insar_features
except ImportError as e:
    print(f"ImportError: Could not import 'extract_insar_features'. Details: {e}")
    sys.exit(1)

def main():
    # --- Setup Paths ---
    insar_dir = (project_root / "data/raw/insar/sentinel1").resolve()
    output_dir = (project_root / "data/processed").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- GeoAnomalyMapper InSAR Pipeline ---")
    print(f"System: {os.cpu_count()} logical cores detected.")

    # --- Find Files ---
    print(f"Searching for data in: {insar_dir}")
    safe_dirs = sorted(list(insar_dir.glob("*.SAFE")))
    coh_paths = []
    
    # Strategy 1: Look inside .SAFE
    if safe_dirs:
        for safe_dir in safe_dirs:
            measurement_dir = safe_dir / "measurement"
            if measurement_dir.exists():
                # Get VV polarization (standard for consistency)
                tiffs = list(measurement_dir.glob("*iw-slc-vv*.tiff"))
                if not tiffs:
                    tiffs = list(measurement_dir.glob("*.tiff"))
                coh_paths.extend(tiffs)
    
    # Strategy 2: Flat files
    if not coh_paths:
        coh_paths = sorted(list(insar_dir.glob("*.tif"))) + sorted(list(insar_dir.glob("*.tiff")))

    # Strategy 3: Check processed folder
    if len(coh_paths) < 2:
        processed_insar_dir = output_dir / "insar"
        if processed_insar_dir.exists():
             coh_paths.extend(sorted(list(processed_insar_dir.glob("*.tif"))))

    if not coh_paths:
        print("Error: No InSAR files found.")
        sys.exit(1)
    
    # Handle single file case
    if len(coh_paths) == 1:
        print("Warning: Only 1 InSAR file found. Duplicating for compatibility.")
        coh_paths.append(coh_paths[0])
    
    print(f"Found {len(coh_paths)} files to process.")

    # --- Compute Incremental Mean ---
    coh_mean_path = output_dir / "coh_mean.tif"
    print("\nStarting Incremental Mean Calculation (RAM Safe Mode)...")
    
    valid_coh_paths = []
    output_profile = None
    
    # Accumulator for the mean
    running_sum = None
    count = 0
    
    t_start = time.time()

    try:
        # Initialize with first file
        print(f"[1/{len(coh_paths)}] Initializing with {coh_paths[0].name}...")
        with rasterio.open(coh_paths[0]) as src:
            output_profile = src.profile.copy()
            ref_shape = src.shape
            # Update profile for float32 output
            output_profile.update(dtype=rasterio.float32, count=1, driver='GTiff')
            
            # Allocate memory for the accumulator (One full scene frame)
            # On 64GB RAM, this single allocation is perfectly fine.
            running_sum = np.zeros(ref_shape, dtype=np.float32)

        # Loop through files
        for i, p in enumerate(coh_paths):
            iter_start = time.time()
            print(f"[{i+1}/{len(coh_paths)}] Processing: {p.name}")
            
            try:
                with rasterio.open(p) as src:
                    if src.shape != ref_shape:
                        print(f"    Warning: Shape mismatch. Expected {ref_shape}, got {src.shape}. Skipping.")
                        continue
                    
                    # Read data
                    data = src.read(1, masked=True)
                    
                    # Convert to valid float array
                    arr = data.filled(np.nan)
                    
                    # If Complex (SLC), get Magnitude
                    if np.iscomplexobj(arr):
                        arr = np.abs(arr)
                    
                    # Add to running sum
                    # nan_to_num ensures NaNs don't poison the sum (treat as 0)
                    np.nan_to_num(arr, copy=False, nan=0.0)
                    running_sum += arr
                    count += 1
                    
                    # Explicit cleanup
                    del data
                    del arr
                
            except Exception as e:
                print(f"    Failed to read {p.name}: {e}")
                continue
            
            print(f"    Processed in {time.time() - iter_start:.2f}s")

        if count == 0:
            print("Error: No valid data processed.")
            sys.exit(1)

        # Calculate Final Mean
        print(f"Calculating average of {count} files...")
        running_sum /= count
        
        # Write Output
        print(f"Writing result to {coh_mean_path}...")
        with rasterio.open(coh_mean_path, "w", **output_profile) as dst:
            dst.write(running_sum.astype(np.float32), 1)
        
        # Free memory
        del running_sum
        print(f"Mean coherence generation complete in {time.time() - t_start:.2f}s")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Run External Features ---
    print(f"\nRunning InSAR feature extraction...")
    cwd = os.getcwd()
    try:
        # Prepare absolute paths
        abs_coh_paths = [str(p.resolve()) for p in valid_coh_paths]
        abs_coh_mean = str(coh_mean_path.resolve())
        
        # Switch to output dir so artifacts save there
        os.chdir(output_dir)
        
        # Call the actual feature extractor
        extract_insar_features(abs_coh_paths, abs_coh_mean)
        print("InSAR features generated successfully.")
        
    except Exception as e:
        print(f"Error extracting InSAR features: {e}")
        sys.exit(1)
    finally:
        os.chdir(cwd)

if __name__ == "__main__":
    main()