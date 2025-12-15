#!/usr/bin/env python3
"""
setup_environment.py - Check dependencies and setup directories
"""

import os
import sys
import subprocess
import pkg_resources

def check_dependencies():
    """Check and install missing dependencies."""
    required = {
        'pandas', 
        'numpy', 
        'shapely', 
        'geopandas', 
        'rasterio', 
        'requests', 
        'earthengine-api'
    }
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing...")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        print("Dependencies installed.")
    else:
        print("All dependencies installed.")

def setup_directories():
    """Create data directories."""
    dirs = [
        'data/reference',
        'data/outputs',
        'data/inputs' 
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Verified directory: {d}")

def print_data_instructions():
    """Print instructions for downloading reference data."""
    print("\n" + "="*50)
    print("DATA DOWNLOAD INSTRUCTIONS")
    print("="*50)
    print("Please download the following files to 'data/reference/':")
    print("\n1. US State Boundaries (cb_2018_us_state_500k.shp)")
    print("   Source: https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")
    print("   Action: Unzip and place contents in data/reference/")
    
    print("\n2. Global Lithology (global_lithology.tif)")
    print("   Source: https://www.glim.org/ (or use USGS State Geologic Map Compilation)")
    print("   Note: For this demo, ensure you have a TIF file named 'global_lithology.tif'")
    
    print("\n3. PAD-US Protected Areas (PADUS3_0_Proclamation_Federal.shp)")
    print("   Source: https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download")
    print("   Action: Download and extract the Shapefile to data/reference/")
    print("="*50 + "\n")

if __name__ == "__main__":
    setup_directories()
    check_dependencies()
    print_data_instructions()
