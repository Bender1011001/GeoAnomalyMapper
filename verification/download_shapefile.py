#!/usr/bin/env python3
"""
download_shapefile.py - Download US State boundaries for ocean masking
"""

import requests
import zipfile
import io
import os

def download_us_shapefile():
    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip"
    target_dir = "data/reference"
    
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        
        print("Extracting...")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(target_dir)
        print(f"Successfully extracted to {target_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if not os.path.exists("data/reference"):
        os.makedirs("data/reference")
    download_us_shapefile()
