import os
import requests
import json
import time
from urllib.parse import urlparse

def fetch_earth_mri_data(sb_item_id, output_dir='./data/mountain_pass'):
    """
    Programmatically retrieves geophysical data from a USGS ScienceBase Item ID.
    
    Args:
        sb_item_id (str): The alphanumeric ScienceBase ID (e.g., '5f8f45a482ce06b040efca6f').
        output_dir (str): Destination directory for the downloaded files.
    """
    # ScienceBase Item URL endpoint
    sb_url = f"https://www.sciencebase.gov/catalog/item/{sb_item_id}?format=json"
    
    print(f"[INFO] Connecting to USGS ScienceBase for Item: {sb_item_id}")
    
    try:
        # Request metadata with a timeout to prevent hanging
        response = requests.get(sb_url, timeout=30)
        response.raise_for_status()
        item_metadata = response.json()
    except requests.exceptions.RequestException as e:
        print(f" Failed to retrieve metadata: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Dataset Title: {item_metadata.get('title', 'Unknown')}")
    
    # Parse file listings
    files = item_metadata.get('files', [])
    if not files:
        print(" No direct file downloads found in this Item.")
        return

    print(f"[INFO] Found {len(files)} available files.")
    
    # Define priority extensions for geophysical grids
    #.gxf = Grid Exchange Format (Standard for Gravity/Mag)
    #.grd = Geosoft Grid
    #.csv /.xyz = Point Data
    target_extensions = ['.gxf', '.grd', '.csv', '.xyz', '.tif']
    
    for file_obj in files:
        file_url = file_obj.get('url')
        file_name = file_obj.get('name')
        
        # Check extension
        _, ext = os.path.splitext(file_name)
        if ext.lower() in target_extensions:
            local_path = os.path.join(output_dir, file_name)
            
            # Skip if already exists
            if os.path.exists(local_path):
                print(f" File already exists: {file_name}")
                continue
                
            print(f" Retrieving {file_name}...")
            try:
                # Stream download for large geophysical grids (often >100MB)
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f" Saved to {local_path}")
                # Be polite to the server
                time.sleep(1) 
            except Exception as e:
                print(f" Failed to download {file_name}: {e}")

# Execution Block for Mountain Pass
if __name__ == "__main__":
    # ScienceBase ID for Mountain Pass Gravity/Mag/Rad Data
    # Source: USGS Data Release DOI:10.5066/P9SQV3SB
    MTN_PASS_ID = "5f8f45a482ce06b040efca6f"
    
    fetch_earth_mri_data(MTN_PASS_ID)
