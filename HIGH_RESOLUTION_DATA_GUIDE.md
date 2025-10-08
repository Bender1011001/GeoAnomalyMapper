# High-Resolution Free Data Sources Guide

## Overview

This guide explains how to access and use high-resolution free geophysical data sources to achieve **deeper penetration** and **finer spatial resolution** for subsurface anomaly detection.

## Resolution Comparison

| Data Source | Resolution | Coverage | Depth Sensitivity | Best For |
|-------------|-----------|----------|-------------------|----------|
| **Sentinel-1 InSAR** | 5-20m | Global | Surface-50m | Surface deformation, shallow voids |
| **XGM2019e Gravity** | ~4km | Global | 100m-10km | Mid-depth density anomalies |
| **EIGEN-6C4 Gravity** | ~11km | Global | 1-50km | Regional structures |
| **EMAG2v3 Magnetic** | ~3.7km | Global | Surface-100km | Magnetic anomalies, geology |
| **Regional Gravity** | 100m-1km | Regional | 20m-5km | Local voids, structures |
| **SRTM DEM** | 30m | Global | Surface | Topographic corrections |
| **EGMS InSAR** | 100m | Europe | Surface-50m | Pre-processed velocities |

---

## 1. Sentinel-1 InSAR (5-20m Resolution)

### What It Provides
- **Surface deformation velocity** (mm/year precision)
- **Coherence maps** (measurement quality indicator)
- **5-20m spatial resolution** depending on processing
- **12-day repeat** for change detection

### How It Detects Deeper Features
Subsidence over voids propagates to the surface through:
- **Compaction zones** (months to years)
- **Sinkhole precursors** (weeks to months)
- **Gradual settlement** (continuous)

Typical detection depths: **20-100 feet (6-30m)** for active processes.

### Access Methods

#### Option 1: Copernicus Data Space (Our Download Script)
```bash
cd GeoAnomalyMapper
python download_geodata.py
# Select region and number of tiles
# Automatically downloads Sentinel-1 SLC products
```

**Advantages:**
- Fully automated
- Includes token refresh
- Parallel downloads
- Auto-resume on failure

**Outputs:**
- Raw SLC (Single Look Complex) files
- Requires InSAR processing (interferometry)

#### Option 2: ASF Data Search (Pre-processed)
```bash
# Alaska Satellite Facility provides processed InSAR products
# Visit: https://search.asf.alaska.edu/

# Filter by:
# - Product Type: InSAR
# - Mission: Sentinel-1
# - Processing Level: RTC (Radiometrically Terrain Corrected)
```

**Advantages:**
- Pre-geocoded
- Radiometric corrections applied
- Easier to use

#### Option 3: EGMS (Europe Only, 100m Resolution)
European Ground Motion Service provides **pre-processed velocity maps**:
- **100m resolution** (lower than raw Sentinel-1)
- **mm/year precision**
- **2015-present coverage**
- **Free download**

```python
# Already implemented in download_geodata.py
# Automatic access via EGMS viewer:
# https://egms.land.copernicus.eu/
```

### Processing InSAR for Anomaly Detection

```python
from GeoAnomalyMapper.multi_resolution_fusion import process_multi_resolution

# After downloading Sentinel-1 data, process interferometric pairs
# (Requires SNAP or ISCE2 software for interferometry)

# Once you have velocity maps, use our fusion pipeline:
process_multi_resolution(
    bounds=(-105.0, 32.0, -104.0, 33.0),  # lon_min, lat_min, lon_max, lat_max
    target_resolution=0.0001,  # 0.0001° ≈ 10m
    output_name="carlsbad_hires"
)
```

---

## 2. XGM2019e Gravity Model (~4km Resolution)

### What It Provides
- **Gravity disturbances** (mGal)
- **~4km resolution** (vs. 11km for EGM2008)
- **Improved accuracy** from GOCE satellite
- **Better lithospheric structure mapping**

### How It Detects Deeper Features
Gravity anomalies indicate density contrasts:
- **Negative anomalies** → voids, low-density zones
- **Positive anomalies** → dense structures, ore bodies

Detection depth: **100m to 10km** depending on size.

### Download Instructions

```bash
# Visit ICGEM (International Centre for Global Earth Models)
# http://icgem.gfz-potsdam.de/

# 1. Select Model: XGM2019e_2159
# 2. Choose "Grid" output
# 3. Set parameters:
#    - Region: Custom (your area of interest)
#    - Resolution: 0.05° (5.5 km) or finer
#    - Gravity Field: Gravity Disturbance
#    - Height: 0 km (surface)
# 4. Format: GeoTIFF

# Example using curl:
curl -X POST "http://icgem.gfz-potsdam.de/calcgrid" \
  -d "model=XGM2019e_2159" \
  -d "lat_from=32" \
  -d "lat_to=33" \
  -d "lon_from=-105" \
  -d "lon_to=-104" \
  -d "grid_step=0.05" \
  -d "functional=gravity_disturbance" \
  -d "height=0" \
  -o xgm2019e_gravity.tif
```

**Recommended Settings:**
- **Resolution:** 0.025° (2.8 km) for best results
- **Height:** 0 km (surface level)
- **Output:** GeoTIFF format

### Integration with Pipeline

```python
# Save downloaded file to:
# data/raw/gravity/xgm2019e_gravity.tif

# The fusion pipeline will automatically detect and use it:
# - Higher weight than EGM2008
# - Preserves fine-scale features via spectral fusion
```

---

## 3. Regional Airborne Gravity Surveys (100m-1km)

### Available Sources

#### USGS Gravity Database (North America)
**Coverage:** United States, parts of Canada/Mexico  
**Resolution:** Variable, often 1-5 km spacing  
**Access:** https://mrdata.usgs.gov/gravity/

```bash
# Download via web interface or API
# Example for southwestern US:
wget https://mrdata.usgs.gov/gravity/gravity-download.zip

# Extract to: data/raw/gravity_regional/usgs_gravity.xyz
```

#### BGS Gravity (United Kingdom)
**Resolution:** 500m-2km  
**Access:** https://www.bgs.ac.uk/datasets/gb-gravity/

#### Geoscience Australia
**Resolution:** 800m onshore, 1.5km offshore  
**Access:** https://ecat.ga.gov.au/

### Processing Point Data

```python
# Convert XYZ point data to raster grid
import pandas as pd
from scipy.interpolate import griddata

# Load point data
data = pd.read_csv('usgs_gravity.xyz', sep='\s+', 
                   names=['lon', 'lat', 'gravity', 'uncertainty'])

# Grid the data
from GeoAnomalyMapper.multi_resolution_fusion import process_multi_resolution

# Grid to 0.001° (100m) resolution
lon_grid = np.arange(lon_min, lon_max, 0.001)
lat_grid = np.arange(lat_min, lat_max, 0.001)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

points = data[['lon', 'lat']].values
values = data['gravity'].values

gravity_grid = griddata(points, values, (LON, LAT), method='cubic')

# Save as GeoTIFF
# (Use rasterio to write grid with proper georeferencing)
```

---

## 4. Complete Workflow Example

### Example: Carlsbad Caverns Void Detection

```bash
# Step 1: Download high-resolution data
cd GeoAnomalyMapper

# 1a. Sentinel-1 InSAR
python download_geodata.py
# Enter region: 50 tiles covering New Mexico
# This gets you 5-20m resolution deformation data

# 1b. High-resolution gravity (manual)
# Visit http://icgem.gfz-potsdam.de/
# Download XGM2019e for region: -106°W to -103°W, 31°N to 34°N
# Save to: data/raw/gravity/xgm2019e_gravity.tif

# 1c. Regional gravity (manual)
# Visit https://mrdata.usgs.gov/gravity/
# Download point data for New Mexico
# Save to: data/raw/gravity_regional/nm_gravity.xyz

# Step 2: Process InSAR interferometry
# (Requires SNAP or ISCE2 - separate workflow)
# Output: Velocity map at data/raw/insar/sentinel1_velocity.tif

# Step 3: Run multi-resolution fusion
python multi_resolution_fusion.py \
  --region "-105.0,32.0,-104.0,33.0" \
  --resolution 0.0001 \
  --output carlsbad_hires

# Step 4: Detect voids with enhanced resolution
python detect_voids.py \
  --region "-105.0,32.0,-104.0,33.0" \
  --resolution 0.0001 \
  --output carlsbad_voids
```

### Expected Results

**Without High-Resolution Data:**
- Resolution: ~11km (EGM2008 gravity only)
- Detection depth: 1-10km
- Suitable for: Regional structures

**With High-Resolution Fusion:**
- Resolution: **10-100m** (combined InSAR + XGM2019e)
- Detection depth: **20-300 feet (6-100m)**
- Suitable for: **Individual caves, voids, sinkholes**

---

## 5. Data Fusion Strategy

The [`multi_resolution_fusion.py`](multi_resolution_fusion.py) script uses several advanced techniques:

### Spectral Fusion
Combines high-frequency details from fine-resolution data with stable background from coarse data:

```
Final = HighPass(InSAR, 10px) + LowPass(Gravity, 10px)
```

**Benefits:**
- Preserves **sharp boundaries** from InSAR
- Maintains **stable anomaly trends** from gravity
- Reduces noise in high-resolution data

### Uncertainty Weighting
Each data source weighted by inverse uncertainty:

```
Weight = 1 / (Uncertainty² + ε)
Fused = Σ(Data × Weight) / Σ(Weight)
```

**Data Quality Ranking:**
1. **InSAR (Sentinel-1):** ±5mm/year → High weight
2. **XGM2019e Gravity:** ±10% → Medium weight
3. **EGM2008 Gravity:** ±15% → Lower weight
4. **Magnetic:** ±8% → Medium weight

### Adaptive Resampling
Automatically selects optimal interpolation:
- **Upsampling (4x+):** Cubic spline
- **Moderate (1.5-3x):** Cubic
- **Downsampling:** Averaging with anti-aliasing

---

## 6. Advanced Techniques

### InSAR Time Series Analysis
For detecting **progressive void development**:

```python
# Process multi-temporal InSAR stack
# Requires: 20+ Sentinel-1 acquisitions over 1+ year

from mintpy import view, timeseries

# Load processed velocity time series
ts = timeseries.load('velocity_stack.h5')

# Identify accelerating subsidence (void growth)
acceleration = np.gradient(ts.velocity, ts.dates)

# Pixels with acceleration > 2 mm/year² are active voids
active_voids = acceleration > 2.0
```

### Gravity Gradient Analysis
Enhances shallow features by computing spatial derivatives:

```python
# Compute horizontal gradients
grad_y, grad_x = np.gradient(gravity_data)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Maxima indicate density boundaries (void edges)
from scipy.ndimage import maximum_filter
edges = (gradient_magnitude == maximum_filter(gradient_magnitude, size=5))
```

### Multi-Frequency Gravity Inversion
Separate shallow and deep signals:

```python
# Upward continuation to different heights
from scipy.fft import fft2, ifft2

def upward_continue(data, height_km):
    """Continue gravity field upward to suppress shallow sources."""
    kx, ky = np.meshgrid(np.fft.fftfreq(data.shape[1]),
                          np.fft.fftfreq(data.shape[0]))
    k = 2 * np.pi * np.sqrt(kx**2 + ky**2)
    
    F = fft2(data)
    F_continued = F * np.exp(-k * height_km)
    return np.real(ifft2(F_continued))

# Shallow features (0-1 km)
shallow = data - upward_continue(data, 1.0)

# Deep features (>1 km)  
deep = upward_continue(data, 1.0)
```

---

## 7. Troubleshooting

### "No products found" for Sentinel-1

**Solution:** Adjust date range or region
```python
# In download_geodata.py, modify:
INSAR_DATE_RANGE_DAYS = 365  # Expand from 90 to 365 days
```

### Gravity download fails from ICGEM

**Solution:** Use alternative format
```python
# Try NetCDF instead of GeoTIFF:
# Format: netCDF → convert with:
gdalwarp -of GTiff gravity.nc gravity.tif
```

### OutOfMemory errors with high-resolution

**Solution:** Process in tiles
```python
# Use smaller regions or limit resolution:
process_multi_resolution(
    bounds=(lon_min, lat_min, lon_max, lat_max),
    target_resolution=0.001,  # Limit to 100m instead of 10m
    max_resolution=0.001
)
```

---

## 8. Citation and Acknowledgments

When using these data sources, please cite:

**Sentinel-1:**
- ESA Copernicus Sentinel-1 data (Year)

**XGM2019e:**
- Zingerle et al. (2020). "The combined global gravity field model XGM2019e." Journal of Geodesy, 94(7), 1-12.

**EMAG2:**
- Meyer et al. (2017). "EMAG2v3: Earth Magnetic Anomaly Grid (2-arc-minute resolution)." NOAA/NCEI.

**EGMS:**
- European Ground Motion Service (EGMS), Copernicus Land Monitoring Service

---

## 9. Summary: Achieving Maximum Resolution

| Goal | Recommended Combination | Expected Resolution |
|------|------------------------|---------------------|
| **Shallow voids (0-50m)** | InSAR + Regional gravity | **10-100m** |
| **Mid-depth (50-300m)** | XGM2019e + InSAR + Magnetic | **100-500m** |
| **Deep structures (>300m)** | XGM2019e + EGM2008 + Seismic | **1-5km** |

**Key Principle:** Resolution is limited by:
1. **Wavelength** of the data source
2. **Depth** of the target (deeper = coarser)
3. **Signal-to-noise** ratio

The multi-resolution fusion pipeline **optimally combines** all available sources to achieve the best possible resolution at each depth range.

---

For questions or issues, see the main [README.md](README.md) or open an issue on GitHub.