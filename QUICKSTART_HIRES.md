# Quick Start: High-Resolution Anomaly Detection

This guide provides ready-to-run examples for achieving maximum resolution with free data.

## Prerequisites

```bash
# Install dependencies
pip install rasterio scipy numpy matplotlib tqdm requests python-dotenv

# Set up credentials (for Sentinel-1 downloads)
cp .env.example .env
# Edit .env and add your Copernicus credentials
```

---

## Example 1: Basic High-Resolution Processing (Using Existing Data)

**Scenario:** You already have gravity and magnetic data, want to fuse at 100m resolution.

```bash
# Unix/Linux/Mac:
python multi_resolution_fusion.py \
  --region "-105.0,32.0,-104.0,33.0" \
  --resolution 0.001 \
  --output my_fusion_100m

# Windows PowerShell:
python multi_resolution_fusion.py `
  --region "-105.0,32.0,-104.0,33.0" `
  --resolution 0.001 `
  --output my_fusion_100m

# Or on one line (works everywhere):
python multi_resolution_fusion.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.001 --output my_fusion_100m

# Output: data/outputs/multi_resolution/my_fusion_100m.tif
```

**What happens:**
1. Loads available data (gravity, magnetic)
2. Resamples to 0.001° (≈100m) using adaptive methods
3. Weights by uncertainty
4. Outputs fused GeoTIFF + statistics report

---

## Example 2: Download + Process Sentinel-1 InSAR

**Scenario:** Need maximum resolution (10-20m) including surface deformation.

### Step 1: Download Sentinel-1 Data

```bash
python download_geodata.py
```

**When prompted:**
- Number of tiles: `10` (for testing)
- Parallel workers: `4`
- Region: Automatically uses USA_LOWER48 preset

**Output:** Raw SLC files in `data/raw/insar/sentinel1/`

### Step 2: Process InSAR (Requires External Tools)

The raw SLC files need interferometric processing. Two options:

#### Option A: Using SNAP (ESA Sentinel Application Platform)

```bash
# Download SNAP: https://step.esa.int/main/download/snap-download/

# Process interferometric pair (example)
gpt graph.xml \
  -Pmaster=S1A_..._20250101_....zip \
  -Pslave=S1A_..._20250113_....zip \
  -Poutput=velocity.tif
```

#### Option B: Use Pre-Processed EGMS Data (Europe Only)

```python
# For European regions, download pre-processed velocities
# Visit: https://egms.land.copernicus.eu/

# Download velocity raster for your region
# Save to: data/raw/insar/egms_velocity.tif
```

### Step 3: Fuse InSAR with Gravity/Magnetic

```bash
# Once you have velocity.tif, run fusion:
# Windows PowerShell:
python multi_resolution_fusion.py `
  --region "-105.0,32.0,-104.0,33.0" `
  --resolution 0.0001 `
  --output hires_with_insar

# Unix/bash (one line):
python multi_resolution_fusion.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.0001 --output hires_with_insar

# Now includes InSAR at 10m resolution!
```

---

## Example 3: Download Higher Resolution Gravity (XGM2019e)

**Scenario:** Want better than EGM2008's 11km resolution.

### Manual Download (One-time Setup)

```bash
# Visit: http://icgem.gfz-potsdam.de/calcgrid

# Fill form:
# - Model: XGM2019e_2159
# - Area: Custom (your region)
# - From latitude: 32.0
# - To latitude: 33.0
# - From longitude: -105.0
# - To longitude: -104.0
# - Grid step: 0.025 (degrees, ≈2.8km)
# - Functional: gravity_disturbance
# - Height: 0.0 km
# - Output format: GeoTIFF

# Download and save to:
mkdir -p data/raw/gravity
mv ~/Downloads/xgm2019e_*.tif data/raw/gravity/xgm2019e_gravity.tif
```

### Run Fusion

```bash
# Windows PowerShell:
python multi_resolution_fusion.py `
  --region "-105.0,32.0,-104.0,33.0" `
  --resolution 0.001 `
  --output fusion_xgm2019e

# Unix/bash (one line):
python multi_resolution_fusion.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.001 --output fusion_xgm2019e

# Automatically detects and uses XGM2019e
# Resolution improved from 11km → 2.8km for gravity
```

---

## Example 4: Regional High-Resolution Gravity (USA)

**Scenario:** Analyzing a site in the USA, want sub-kilometer gravity resolution.

### Download USGS Gravity Data

```bash
# Visit: https://mrdata.usgs.gov/gravity/
# Select your state (e.g., New Mexico)
# Download XYZ point data

wget https://mrdata.usgs.gov/magnetic/nmag-download.zip
unzip nmag-download.zip

# Convert XYZ to GeoTIFF (using gdal_grid)
gdal_grid -a invdist:power=2:smoothing=1.0 \
  -txe -105.0 -104.0 \
  -tye 32.0 33.0 \
  -outsize 1000 1000 \
  -of GTiff \
  -ot Float32 \
  nmag.xyz \
  data/raw/gravity_regional/nm_gravity.tif
```

### Fuse with All Data

```bash
# Windows PowerShell or one-line (all platforms):
python multi_resolution_fusion.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.001 --output complete_fusion
```

**Result:** Combines:
- Regional gravity (1km)
- XGM2019e (2.8km)
- Magnetic (3.7km)
- InSAR if available (10m)

---

## Example 5: Void Detection with High-Resolution Data

**Scenario:** Detect underground voids at Carlsbad Caverns using all available data.

```bash
# Step 1: Ensure data is available (from previous examples)
# - InSAR velocity map: data/raw/insar/sentinel1_velocity.tif
# - XGM2019e gravity: data/raw/gravity/xgm2019e_gravity.tif
# - EMAG2 magnetic: data/raw/emag2/EMAG2_V3_SeaLevel_DataTiff.tif

# Step 2: Run high-resolution fusion (Windows PowerShell):
python multi_resolution_fusion.py --region "-104.5,32.1,-104.0,32.3" --resolution 0.0001 --output carlsbad_hires

# Step 3: Detect voids using fused data
python detect_voids.py --region "-104.5,32.1,-104.0,32.3" --resolution 0.0001 --output carlsbad_voids

# Outputs:
# - data/outputs/void_detection/carlsbad_voids.tif (probability map)
# - data/outputs/void_detection/carlsbad_voids.png (visualization)
# - data/outputs/void_detection/carlsbad_voids_report.txt (statistics)
```

### Interpret Results

```python
import rasterio
import matplotlib.pyplot as plt

# Load probability map
with rasterio.open('data/outputs/void_detection/carlsbad_voids.tif') as src:
    probability = src.read(1)
    extent = [src.bounds.left, src.bounds.right, 
              src.bounds.bottom, src.bounds.top]

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(probability, extent=extent, cmap='hot', vmin=0, vmax=1)
plt.colorbar(label='Void Probability')
plt.title('Underground Void Detection - Carlsbad Caverns')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# High probability areas (>0.7) are likely voids
high_prob = probability > 0.7
print(f"High-probability void pixels: {high_prob.sum()}")
```

---

## Example 6: Time Series Analysis (Advanced)

**Scenario:** Track progressive void development over time.

```bash
# Download multiple Sentinel-1 acquisitions (20+ scenes over 1 year)
python download_geodata.py
# Select 50 tiles, covering full year

# Process time series (requires mintpy or similar)
# This creates a velocity + acceleration dataset

# Then analyze acceleration patterns
python -c "
import numpy as np
import rasterio

# Load velocity time series
velocities = []  # Load your processed time series here
dates = []       # Acquisition dates

# Calculate acceleration
acceleration = np.gradient(velocities, axis=0)

# Pixels accelerating > 2 mm/year² indicate active voids
active_voids = acceleration[-1] > 2.0
print(f'Active void zones: {active_voids.sum()} pixels')
"
```

---

## Example 7: Custom Resolution Pyramid

**Scenario:** Generate multiple resolutions for web mapping.

```bash
# Generate resolution pyramid: 10m, 100m, 1km, 10km
for res in 0.0001 0.001 0.01 0.1; do
  python multi_resolution_fusion.py \
    --region "-105.0,32.0,-104.0,33.0" \
    --resolution $res \
    --output "pyramid_${res}"
done

# Create overview pyramid for fast web viewing
gdal_translate -of GTiff \
  -co TILED=YES \
  -co COMPRESS=DEFLATE \
  data/outputs/multi_resolution/pyramid_0.0001.tif \
  pyramid_web.tif

gdaladdo -r average pyramid_web.tif 2 4 8 16 32 64
```

---

## Performance Tips

### Memory Usage

For large regions or high resolution, use tiling:

```python
# Process in 1° x 1° tiles
for lat in range(32, 34):
    for lon in range(-106, -103):
        process_multi_resolution(
            bounds=(lon, lat, lon+1, lat+1),
            target_resolution=0.0001,
            output_name=f"tile_{lat}_{lon}"
        )

# Merge tiles
gdalbuildvrt merged.vrt tile_*.tif
```

### Download Speed

Optimize parallel downloads:

```python
# Edit download_geodata.py to increase workers
# Line ~XXX: max_workers = 10  # Increase from 4 to 10

# Or use aria2c for faster downloads
aria2c -x 16 -s 16 <sentinel_url>
```

### Processing Speed

Use cloud-optimized formats:

```bash
# Convert to COG for faster access
rio cogeo create input.tif output_cog.tif \
  --overview-level 5 \
  --overview-resampling average
```

---

## Troubleshooting

### "No data layers available"

**Solution:** Check data paths are correct
```bash
ls -lh data/raw/gravity/
ls -lh data/raw/emag2/
# Ensure files exist
```

### "Out of memory" errors

**Solution:** Reduce resolution or region size
```bash
# Instead of 0.0001° (10m), use 0.001° (100m)
python multi_resolution_fusion.py \
  --region "-105.0,32.0,-104.0,33.0" \
  --resolution 0.001  # Increased from 0.0001
```

### Sentinel-1 download fails

**Solution:** Check credentials and network
```bash
# Verify .env file
cat .env | grep CDSE_

# Test connection
curl -I https://identity.dataspace.copernicus.eu
```

---

## Next Steps

1. **Read the full guide:** [HIGH_RESOLUTION_DATA_GUIDE.md](HIGH_RESOLUTION_DATA_GUIDE.md)
2. **Explore fusion parameters:** Adjust weights, spectral transitions
3. **Validate results:** Compare with known cave/void locations
4. **Contribute:** Share your findings and improvements!

---

## Expected Results by Resolution

| Resolution | Processing Time | Memory | Best For |
|-----------|----------------|--------|----------|
| **10m** (0.0001°) | 10-30 min | 8-16 GB | Individual voids, caves |
| **100m** (0.001°) | 1-5 min | 2-4 GB | Void clusters, structures |
| **1km** (0.01°) | <1 min | <1 GB | Regional patterns |
| **10km** (0.1°) | <10 sec | <500 MB | Continental scale |

Choose resolution based on:
- **Target depth:** Deeper features need coarser resolution
- **Data availability:** Higher resolution needs more data sources
- **Computational resources:** Finer grids need more RAM/time

---

For questions, see [README.md](README.md) or [HIGH_RESOLUTION_DATA_GUIDE.md](HIGH_RESOLUTION_DATA_GUIDE.md)