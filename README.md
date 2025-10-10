# GeoAnomalyMapper - Multi-Resolution Geophysical Analysis

**Achieve 10-100m resolution using free data** - from global coverage to individual voids.

## 🚀 New: High-Resolution Data Fusion

**See deeper with higher resolution** using free data sources:

| Resolution | Data Source | What You Can Detect |
|-----------|-------------|---------------------|
| **10-20m** | Sentinel-1 InSAR | Individual caves, sinkholes, active subsidence |
| **100m-1km** | Regional gravity + InSAR | Void clusters, karst zones, abandoned mines |
| **~4km** | XGM2019e gravity model | Mid-depth density anomalies, salt domes |
| **~11km** | EGM2008 (baseline) | Regional structures, lithospheric features |

**👉 Quick Start:** [QUICKSTART_HIRES.md](QUICKSTART_HIRES.md)
**📚 Complete Guide:** [HIGH_RESOLUTION_DATA_GUIDE.md](HIGH_RESOLUTION_DATA_GUIDE.md)

### What's New

✨ **Multi-Resolution Fusion Pipeline** ([`multi_resolution_fusion.py`](multi_resolution_fusion.py))
- Adaptive resampling based on data characteristics
- Uncertainty-weighted layer combination
- Spectral fusion preserving fine-scale features
- Automatic detection of available data sources

✨ **High-Resolution Data Access**
- Automated Sentinel-1 InSAR downloads (5-20m resolution)
- XGM2019e gravity model integration (~4km vs 11km)
- Regional airborne gravity survey support (100m-1km)
- EGMS pre-processed InSAR for Europe (100m)

✨ **Advanced Void Detection** ([`detect_voids.py`](detect_voids.py))
- Multi-layer probability mapping
- Optimized for 20-300 foot (6-100m) depth
- Combines gravity, InSAR, lithology, and seismic data
- Identifies high-probability void clusters

---

## What This Does

Processes global and regional geophysical datasets to create:
- **High-resolution fusion maps** (10m-100m with InSAR)
- **648 tiled Cloud-Optimized GeoTIFFs** (10°×10° global coverage)
- **Void probability maps** with uncertainty quantification
- Interactive Google Earth KMZ overlay
- Web-based Cesium.js globe viewer
- Statistical analysis and visualizations

## Quick Start

### 1. Install Dependencies

**Basic (gravity/magnetic processing):**
```bash
pip install numpy rasterio affine tqdm matplotlib simplekml scipy
```

**Optional enhancements:**
```bash
# Better COG generation
pip install rio-cogeo

# InSAR processing (advanced)
pip install isce2 mintpy

# Or install everything:
pip install -e ".[all]"
```
### 2. Configure Credentials (Required for Sentinel-1 Downloads)

**⚠️ IMPORTANT: Never commit credentials to git!**

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Register for free Copernicus account:**
   - Visit: https://dataspace.copernicus.eu/
   - Click "Register" and verify your email

3. **Add your credentials to `.env`:**
   ```bash
   CDSE_USERNAME=your_email@example.com
   CDSE_PASSWORD=your_password
   ```

See [`SECURITY.md`](SECURITY.md) for detailed credential management best practices.


### 3. Download Data

Create data directory and download source files:

```bash
mkdir -p ../data/raw/emag2
mkdir -p ../data/raw/gravity
```

**Required files:**
- **Magnetic:** EMAG2_V3_SeaLevel_DataTiff.tif → `../data/raw/emag2/`
  - Mirrors: `https://www.ngdc.noaa.gov/geomag/EMag2/EMAG2_V3_20170530.tif.gz`, `https://www.ngdc.noaa.gov/mgg/global/EMAG2_V3_20170530.tif.gz`
  - The helper automatically decompresses the gzip archive into the expected filename.

- **Gravity:** EGM2008_Free_Air_Anomaly.tif → `../data/raw/gravity/`
  - Mirrors: `https://topex.ucsd.edu/gravity/EGM2008/EGM2008_Free_Air_Anomaly.tif.gz`, `https://download.csr.utexas.edu/outgoing/legendre/EGM2008/EGM2008_Free_Air_Anomaly.tif.gz`
  - If the mirrors are offline, request the grid from https://icgem.gfz-potsdam.de/

**Automated missing-data helper:**

If the Final Project Report or processing log lists missing baseline datasets,
use the helper to download (or at least locate) them automatically:

```bash
python download_missing_data.py --report data/outputs/processing.log
```

The script parses the report, fetches recognised datasets when possible, and
prints manual follow-up instructions if automated downloads are not available.

### 4. Run Pipeline

**Basic processing (global gravity/magnetic):**
```bash
# Process all 648 global tiles (resumable)
python process_global_map.py

# Analyze results and generate statistics
python analyze_results.py

# Create interactive globe viewers
python create_globe_overlay.py
```

**Maximum resolution data downloader (optional):**
```bash
# Download highest resolution free data globally
# Requires Copernicus credentials in .env file
python download_geodata.py

# Interactive prompts will guide you through:
# - Gravity models (XGM2019e - 4km resolution)
# - Magnetic data (EMAG2v3 - 2 arcmin)
# - Sentinel-1 InSAR (5-20m resolution, select regions)
# - EGMS pre-processed InSAR (Europe, 100m)
# - Regional high-res gravity datasets
```

**Advanced void detection (20-300 ft depth):**
```bash
# Detect underground voids using multi-layer analysis
python detect_voids.py --region "-105.0,32.0,-104.0,33.0" --output my_area_voids

# For InSAR data setup, see INSAR_DATA_GUIDE.md
```

### 4. View Results

**Google Earth (Recommended):**
- Open: `../data/outputs/final/fused_anomaly_google_earth.kmz`

**Web Browser:**
- Open: `../data/outputs/final/globe_viewer.html`

## Files

```
GeoAnomalyMapper/
├── process_global_map.py    # Main processing pipeline (standalone)
├── analyze_results.py        # Statistics and visualization
├── create_globe_overlay.py   # Interactive globe generators
├── detect_voids.py           # 🆕 Advanced void detection (multi-layer)
├── INSAR_DATA_GUIDE.md       # 🆕 How to get InSAR subsidence data
├── pyproject.toml            # Dependencies
├── README.md                 # This file
└── LICENSE                   # MIT license

../data/
├── raw/                      # Input datasets
│   ├── emag2/                # Magnetic data
│   ├── gravity/              # Gravity data
│   ├── insar/                # 🆕 InSAR subsidence data (optional)
│   └── LiMW_GIS 2015.gdb     # Lithology database
└── outputs/
    ├── cog/fused/            # 648 processed tiles
    ├── final/                # Final outputs (KMZ, HTML, GeoTIFF)
    └── void_detection/       # 🆕 Void probability maps
```

## System Requirements

- **Python:** 3.9+
- **GDAL:** Command-line tools (gdalbuildvrt, gdal_translate)
- **PMTiles:** Optional, for web tile generation
- **Disk:** ~2GB for outputs
- **RAM:** 4GB minimum, 8GB recommended

## Advanced Void Detection 🆕

For detecting underground voids at **20-300 feet depth** with much better resolution:

### What's New:

**Multi-Layer Probability Mapping:**
- ✅ Gravity anomalies (density deficits = voids)
- ✅ InSAR subsidence (ground sinking above voids)
- ✅ Lithology analysis (karst-prone rocks)
- ✅ Seismic velocity (low velocity = fractures/voids)

**Resolution Improvements:**
- Global data: ~11 km (regional features)
- With InSAR: **5-100 meters** (local voids!)

**Detection Capabilities:**
- Cave systems (20+ ft)
- Sinkholes (early warning via subsidence)
- Abandoned mines
- Salt dome voids
- Karst collapse zones

### Quick Example:

```bash
# 1. Get InSAR data (see INSAR_DATA_GUIDE.md)
# 2. Run void detection for your area:
python detect_voids.py \
    --region "lon_min,lat_min,lon_max,lat_max" \
    --resolution 0.001 \  # ~100m resolution
    --output my_voids

# Outputs:
# - my_voids.tif (probability map: 0=no void, 1=very likely)
# - my_voids.png (visualization)
# - my_voids_report.txt (statistics & hotspot list)
```

### Best Practices:

1. **Start with gravity alone** (already have data)
2. **Add InSAR** for European regions (free EGMS data!)
3. **Add lithology** (already in data/raw/)
4. **Combine all layers** for highest accuracy

See [`INSAR_DATA_GUIDE.md`](INSAR_DATA_GUIDE.md) for detailed InSAR setup instructions.

---

## Notes

- Processing is **resumable** - interrupted runs restart automatically
- Each tile is 100×100 pixels at 0.1° resolution (global data)
- Void detection can achieve **5-100m resolution** with InSAR
- Robust z-score normalization handles outliers
- NaN values preserved for nodata regions
- All utilities embedded in scripts (minimal external dependencies)