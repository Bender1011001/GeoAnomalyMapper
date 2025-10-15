# GeoAnomalyMapper - Data Download Status

**Last Updated:** 2025-10-11  
**Project:** SAR-project (GeoAnomalyMapper)

---

## ✅ Downloaded Data Summary

### Phase 1: Critical Baseline Data

| Dataset | Status | Location | Size | Resolution | Notes |
|---------|--------|----------|------|------------|-------|
| **EMAG2v3 Magnetic** | ✅ Complete | `data/raw/emag2/` | ~300 MB | 2 arcmin (~4km) | Global magnetic anomaly field |
| **XGM2019e Gravity Model** | ✅ Complete | `data/raw/gravity/` | ~500 MB | Degree 5540 (~2km) | Coefficient file (.gfc) |
| **EGM2008 Gravity** | ✅ Complete | `data/raw/gravity/` | Variable | ~9km | Baseline gravity disturbance |
| **Copernicus DEM 30m** | 🔄 In Progress | `data/raw/elevation/copernicus_dem/` | ~50 GB | 30m | USA Lower 48 coverage |

### Additional Data

| Dataset | Status | Location | Notes |
|---------|--------|----------|-------|
| **Sentinel-1 InSAR** | ✅ Complete | `data/raw/insar/sentinel1/` | Multiple SAR scenes (S1A, S1C) for ground deformation analysis |
| **Lithology Database** | ✅ Complete | `data/raw/LiMW_GIS 2015.gdb/` | GIS geodatabase with lithology information |
| **AWS Open Data Access** | ✅ Documented | `data/raw/aws_open_data/` | Instructions for Landsat, Sentinel-2, NAIP, Terrain Tiles |

---

## 📋 What You Have Now

### Ready to Use
1. **Global Magnetic Anomaly** (EMAG2v3) - Immediate use
2. **High-Resolution Gravity** (XGM2019e) - Needs conversion to GeoTIFF
3. **InSAR Data** (Sentinel-1) - Needs processing with SNAP/ISCE
4. **Lithology/Geology** - GIS database ready

### In Progress
- **Copernicus DEM** - Downloading tiles for USA Lower 48

---

## ⚠️ Manual Steps Required

### 1. XGM2019e Gravity - Convert to GeoTIFF

The XGM2019e model is downloaded as coefficients (`.gfc` file).  
**Convert it to GeoTIFF for your region:**

1. Visit: http://icgem.gfz-potsdam.de/tom_longtime
2. Select these options:
   - Model: **XGM2019e_2159**
   - Grid type: **Grid**
   - Latitude range: **24.5 to 49.5** (USA Lower 48)
   - Longitude range: **-125.0 to -66.95**
   - Grid step: **0.02 degree** (2km resolution)
   - Height: **0m** (sea level)
   - Quantity: **Gravity disturbance**
   - Format: **GeoTIFF**
3. Click "Compute grid"
4. Download and save to: `data/raw/gravity/xgm2019e/xgm2019e_usa.tif`

**Why this matters:** XGM2019e provides ~2km resolution vs ~20km for EGM2008 - that's **10x better resolution** for detecting subsurface anomalies!

---

## 🚀 Next Steps

### To Start Using the Data:

1. **Complete Copernicus DEM download** (in progress)
   
2. **Convert XGM2019e to GeoTIFF** (see manual steps above)

3. **Run data processing:**
   ```bash
   python GeoAnomalyMapper/process_data.py --region "-105.0,32.0,-104.0,33.0"
   ```

4. **Run void detection:**
   ```bash
   python GeoAnomalyMapper/detect_voids.py --region "-105.0,32.0,-104.0,33.0"
   ```

### For Higher Resolution Analysis:

1. **Process InSAR data** with SNAP or ISCE:
   ```bash
   python GeoAnomalyMapper/process_insar_data.py
   ```

2. **Run multi-resolution fusion:**
   ```bash
   python GeoAnomalyMapper/multi_resolution_fusion.py --output usa_hires
   ```

---

## 📊 Data Specifications

### Magnetic Data (EMAG2v3)
- **File:** `EMAG2_V3_SeaLevel_DataTiff.tif`
- **Coverage:** Global
- **Resolution:** 2 arc-minutes (~3.7 km at equator)
- **Units:** nanoTesla (nT)
- **Purpose:** Detect magnetic anomalies from subsurface structures

### Gravity Data (XGM2019e)
- **File:** `XGM2019e_2159.gfc` (coefficients)
- **Max Degree:** 5540 (effective resolution ~2km)
- **Units:** m/s² or mGal (after conversion)
- **Purpose:** Detect mass deficits (voids, caves, karst)

### InSAR Data (Sentinel-1)
- **Sensor:** C-band SAR (5.6 cm wavelength)
- **Resolution:** 5m × 20m (range × azimuth)
- **Purpose:** Detect ground deformation over subsurface voids
- **Scenes Available:** Multiple dates for interferometry

### Elevation Data (Copernicus DEM)
- **Resolution:** 30 meters
- **Vertical Accuracy:** < 4m (90% linear error)
- **Format:** Cloud-Optimized GeoTIFF (COG)
- **Purpose:** Topographic analysis, deformation detection

---

## 💾 Storage Requirements

| Phase | Downloaded | Total Needed |
|-------|-----------|--------------|
| **Current** | ~52 GB | - |
| **Phase 1 Complete** | ~51 GB | 100 GB free space recommended |
| **With Processing** | +10-20 GB | 150 GB free space recommended |
| **Phase 2 (Optional)** | +610 GB | 1 TB+ recommended |

---

## 🔧 Optional Enhancements

### Phase 2: High-Resolution Data (Not Yet Downloaded)

If you want even better results, consider adding:

1. **Sentinel-2 Optical** (~100 GB)
   - 10m multispectral imagery
   - Requires: Copernicus Data Space account (free)
   
2. **USGS 3DEP Lidar** (~500 GB)
   - 1m resolution elevation
   - Coverage: ~60% of USA
   
3. **USGS Aeromagnetic Surveys** (~10 GB)
   - 100m-1km resolution
   - Regional coverage

**To download Phase 2:**
```bash
python GeoAnomalyMapper/download_all_free_data.py --phases 2
```

---

## 📚 Documentation Files

- **Download Guide:** `GeoAnomalyMapper/AUTOMATED_DOWNLOAD_GUIDE.md`
- **High-Res Guide:** `GeoAnomalyMapper/HIGH_RESOLUTION_DATA_GUIDE.md`
- **InSAR Guide:** `GeoAnomalyMapper/INSAR_DATA_GUIDE.md`
- **Quickstart:** `GeoAnomalyMapper/QUICKSTART.md`
- **Data Catalog:** `GeoAnomalyMapper/COMPREHENSIVE_FREE_DATA_CATALOG.md`

---

## ✅ Verification Checklist

- [x] EMAG2v3 magnetic data downloaded
- [x] XGM2019e gravity coefficients downloaded
- [ ] XGM2019e converted to GeoTIFF (MANUAL STEP REQUIRED)
- [x] Sentinel-1 InSAR scenes downloaded
- [ ] Copernicus DEM tiles downloaded (IN PROGRESS)
- [x] Lithology database present
- [ ] Data processing completed
- [ ] Void detection run

---

## 🆘 Troubleshooting

### Copernicus DEM 404 Errors
**This is normal!** Not all lat/lon coordinates have tiles (oceans, outside coverage). The script downloads available tiles automatically.

### Missing XGM2019e GeoTIFF
You need to manually convert the coefficient file. See "Manual Steps Required" section above.

### InSAR Processing Fails
InSAR requires specialized software (SNAP or ISCE). Consider using pre-processed data from COMET LiCSAR instead.

### Out of Disk Space
- Move data to external drive
- Download only specific regions instead of full USA
- Start with Phase 1 only, add Phase 2 later

---

## 📞 Support

For issues or questions:
1. Check documentation in `GeoAnomalyMapper/` directory
2. Review processing guides in `data/processed/`
3. Consult `PROJECT_STATUS.md` for known issues

---

**Status:** ✅ Phase 1 data mostly complete, ready for processing after XGM2019e conversion