import rasterio
import numpy as np
from rasterio.transform import from_bounds
from pathlib import Path

# Target region: -116.0, 35.0, -115.0, 36.0
# Resolution: 0.001
minx, miny, maxx, maxy = -116.0, 35.0, -115.0, 36.0
res = 0.001
width = int((maxx - minx) / res)
height = int((maxy - miny) / res)
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Create mask
data = np.zeros((height, width), dtype=np.uint8)

# Mountain Pass Mine location (approx)
mine_lon = -115.53
mine_lat = 35.48

# Convert to pixel coords
r, c = rasterio.transform.rowcol(transform, mine_lon, mine_lat)

# Draw a circle (radius ~2km = 20 pixels)
y, x = np.ogrid[:height, :width]
dist = np.sqrt((x - c)**2 + (y - r)**2)
data[dist <= 20] = 1

output_dir = Path("outputs/validation")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "known_mineral_deposits_mask.tif"

profile = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': 1,
    'dtype': 'uint8',
    'crs': 'EPSG:4326',
    'transform': transform,
    'nodata': 0,
    'compress': 'DEFLATE'
}

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(data, 1)

print(f"Created validation mask at {output_path}")