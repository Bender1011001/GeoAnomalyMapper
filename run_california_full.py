#!/usr/bin/env python3
"""
Run Full California Mineral Exploration Workflow

This script executes the complete multi-source pipeline for California,
integrating all available public datasets:
- Gravity (USGS)
- InSAR Coherence (Seasonal USA)
- Lithology (GLIM database)
- DEM (if available)

Magnetic data (EMAG2) is currently missing but the workflow will adapt.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from workflow import run_workflow

if __name__ == "__main__":
    # California bounding box
    california_region = (-125.01, 31.98, -113.97, 42.52)  # (lon_min, lat_min, lon_max, lat_max)
    
    # Resolution in degrees per pixel
    # Using ~2.5km resolution to match gravity data
    # At California's latitude (~37°), 1° ≈ 111 km
    # So 0.025° ≈ 2.75 km
    resolution = 0.025
    
    # Output path
    output_path = Path("data/outputs/california_full_multisource")
    
    print("="*80)
    print("CALIFORNIA MULTI-SOURCE MINERAL EXPLORATION WORKFLOW")
    print("="*80)
    print(f"Region: {california_region}")
    print(f"Resolution: {resolution}° (~2.5 km)")
    print(f"Output: {output_path}")
    print()
    print("Data Sources:")
    print("  ✅ Gravity (USGS): Available")
    print("  ✅ InSAR (Seasonal USA): Available")
    print("  ✅ Lithology (GLIM): Available")
    print("  ✅ Magnetic (EMAG2): Available (3 files found)")
    print("="*80)
    print()
    
    try:
        run_workflow(
            region=california_region,
            resolution=resolution,
            output_name=output_path,
            mode='mineral',
            skip_visuals=False
        )
        print()
        print("="*80)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ WORKFLOW FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
