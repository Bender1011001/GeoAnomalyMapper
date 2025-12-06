import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
import os

# 1. Define USA Bounding Box (Approximate Conterminous US)
# Split into chunks if 'Main USA' is too large for one query
usa_bbox = "POLYGON((-125 25, -66 25, -66 49, -125 49, -125 25))"

print("Searching for Seasonal Coherence tiles over USA...")

# 2. Search for the specific 'S1_COHERENCE' collection
# We want 'Coherence' polarization (usually VV is best for structures)
results = asf.geo_search(
    intersectsWith=usa_bbox,
    dataset='S1_COHERENCE_12_DAY',  # 12-day repeat coherence is standard
    # You can also look for 'Seasonal' aggregates if available via API, 
    # but 12-day pairs are standard products.
    maxResults=500  # Adjust based on storage! USA is huge.
)

print(f"Found {len(results)} tiles.")

# 3. Download
# WARNING: This is lots of data. Ensure GEOANOMALYMAPPER_DATA_DIR has space.
data_dir = os.environ.get('GEOANOMALYMAPPER_DATA_DIR', 'data') + '/raw/insar/seasonal_global'
os.makedirs(data_dir, exist_ok=True)

print(f"Downloading to {data_dir}...")
# Requires NASA Earthdata Login. It will prompt you or look for .netrc
results.download(path=data_dir, processes=4)