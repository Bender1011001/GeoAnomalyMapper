import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Bounding Box for Contiguous USA (Approximate)
LAT_MIN, LAT_MAX = 24, 50   # 24N to 50N
LON_MIN, LON_MAX = -125, -66 # 125W to 66W

# Output Directory (Adjust as needed)
# Use environment variable if set, otherwise default to data/
data_root = os.environ.get('GEOANOMALYMAPPER_DATA_DIR', 'data')
OUTPUT_DIR = Path(data_root) / "raw/insar/seasonal_usa"

# AWS Bucket Details (Public, No Login Needed)
BUCKET_NAME = "sentinel-1-global-coherence-earthbigdata"
PREFIX = "data/tiles" 

# Metrics to download
# COH12 = 12-day coherence (Best for general structure detection)
# AMP   = Backscatter Amplitude (Optional, good for visual context)
METRICS = ["COH12"] 
SEASONS = ["winter", "spring", "summer", "fall"]
POLARIZATION = "vv" # VV is standard for land features

def get_tile_id(lat, lon):
    """
    Generates Tile ID based on Upper Left Corner (Format: N48W090).
    Dataset uses Upper Left coordinate naming.
    """
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"

def download_usa_dataset(region=None, resolution=None, output_dir=None):
    # Configure anonymous S3 access (No credentials needed)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    if output_dir:
        out_dir = Path(output_dir) / "raw/insar/seasonal_usa"
    else:
        out_dir = OUTPUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    
    lat_min, lat_max = LAT_MIN, LAT_MAX
    lon_min, lon_max = LON_MIN, LON_MAX

    if region:
        # region is (lon_min, lat_min, lon_max, lat_max)
        # We need to find integer bounds that cover this region
        import math
        lon_min_req, lat_min_req, lon_max_req, lat_max_req = region
        
        # Floor min, Ceil max to ensure coverage
        lat_min = max(LAT_MIN, int(math.floor(lat_min_req)))
        lat_max = min(LAT_MAX, int(math.ceil(lat_max_req)))
        lon_min = max(LON_MIN, int(math.floor(lon_min_req)))
        lon_max = min(LON_MAX, int(math.ceil(lon_max_req)))

    print(f"Generating tile list for region ({lat_min}N-{lat_max}N, {lon_min}W-{lon_max}W)...")
    
    tasks = []
    # Loop through Lat/Lon grid (1-degree steps)
    # Range ends are exclusive, so we add +1 to max
    for lat in range(lat_max, lat_min, -1): # Top-down
        for lon in range(lon_min, lon_max):  # Left-right
            tile_id = get_tile_id(lat, lon)
            
            # Construct S3 Key for each season/metric
            for season in SEASONS:
                for metric in METRICS:
                    # File format: N48W090_winter_vv_COH12.tif
                    filename = f"{tile_id}_{season}_{POLARIZATION}_{metric}.tif"
                    key = f"{PREFIX}/{tile_id}/{filename}"
                    local_path = out_dir / tile_id / filename
                    tasks.append((key, local_path))

    print(f"Found {len(tasks)} potential files to download.")
    print("Starting download (this is ~100GB of data, ensure disk space!)...")

    success_count = 0
    skip_count = 0
    fail_count = 0

    pbar = tqdm(tasks, unit="file")
    for s3_key, local_path in pbar:
        if local_path.exists():
            skip_count += 1
            continue
        
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET_NAME, s3_key, str(local_path))
            success_count += 1
        except Exception as e:
            # 404 is common over ocean tiles or edges
            fail_count += 1
            # Optional: delete empty parents if failed
            if local_path.parent.exists() and not any(local_path.parent.iterdir()):
                local_path.parent.rmdir()

    print("\n--- Download Complete ---")
    print(f"Downloaded: {success_count}")
    print(f"Skipped (Exists): {skip_count}")
    print(f"Failed/Missing (Ocean): {fail_count}")
    print(f"Data saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    download_usa_dataset()