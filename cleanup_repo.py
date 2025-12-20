"""
Cleanup Script for GeoAnomalyMapper.

Removes large intermediate files from failed/superseded experiments (V1, V2, V4).
Retains V3 (Best Model) and all data inputs.
"""

import os
import glob
from pathlib import Path

def cleanup():
    print("Starting repository cleanup...")
    
    # Files to delete (large GeoTIFFs and checkpoints)
    patterns = [
        "data/outputs/usa_density_model.tif",      # V1
        "data/outputs/usa_density_model_v2.tif",   # V2 (broken)
        "data/outputs/usa_density_model_v4.tif",   # V4 (valid but worse MRDS hit rate)
        "usa_pinn_model.pth",                      # V1 model
        "usa_pinn_model_v4.pth",                   # V4 model
        "analyze_gold_belts.py",                   # Intermediate script
        "compute_residual_gravity.py",             # Keep this? It's useful tool. Let's keep it.
        # Keep usa_density_model_v3.tif as the MAIN result
    ]
    
    deleted_size = 0
    
    for p in patterns:
        files = glob.glob(p)
        for f in files:
            try:
                size = os.path.getsize(f)
                os.remove(f)
                deleted_size += size
                print(f"Deleted: {f} ({size/1024/1024:.1f} MB)")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
                
    print(f"Cleanup complete. Reclaimed {deleted_size/1024/1024:.1f} MB.")
    print("Kept: V3 Model (Best Performer) and Residual Gravity inputs.")

if __name__ == "__main__":
    cleanup()
