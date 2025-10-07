# GeoAnomalyMapper - Minimal Pipeline

**Ultra-lightweight global geophysical anomaly mapper** - just 3 scripts, no framework overhead.

## What This Does

Processes global magnetic (EMAG2) and gravity (EGM2008) datasets to create:
- 648 tiled Cloud-Optimized GeoTIFFs (10°×10° tiles)
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

### 2. Download Data

Create data directory and download source files:

```bash
mkdir -p ../data/raw/emag2
mkdir -p ../data/raw/gravity
```

**Required files:**
- **Magnetic:** EMAG2_V3_SeaLevel_DataTiff.tif → `../data/raw/emag2/`
  - URL: https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif

- **Gravity:** EGM2008 gravity disturbance GeoTIFF → `../data/raw/gravity/`
  - Find from ICGEM or preprocessed sources

### 3. Run Pipeline

**Basic processing (global gravity/magnetic):**
```bash
# Process all 648 global tiles (resumable)
python process_global_map.py

# Analyze results and generate statistics
python analyze_results.py

# Create interactive globe viewers
python create_globe_overlay.py
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