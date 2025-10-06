# Data Acquisition Guide for GeoAnomalyMapper

This guide provides detailed instructions for obtaining the required global geophysical datasets for the GeoAnomalyMapper pipeline.

## Required Datasets

The `process_global_map.py` script requires two global datasets in GeoTIFF format:

1. **EMAG2_V3_Sea_Level.tif** - Global magnetic anomaly data
2. **EGM2008_Free_Air_Anomaly.tif** - Global gravity free-air anomaly data

Both files must be placed in: `data/raw/`

---

## 1. Magnetic Data: EMAG2 V3

### Direct Download (Recommended)

**Source:** NOAA National Centers for Environmental Information (NCEI)

**File:** EMAG2 V3 Sea Level GeoTIFF

**URL:** https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif

**Instructions:**
```bash
# Create the directory if it doesn't exist
mkdir -p data/raw

# Download the file (approximately 1.5 GB)
curl -o data/raw/EMAG2_V3_Sea_Level.tif https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif

# Or use wget
wget -O data/raw/EMAG2_V3_Sea_Level.tif https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif
```

**Dataset Details:**
- Resolution: 2 arc-minute (approximately 3.7 km at the equator)
- Coverage: Global
- Units: nanoTesla (nT)
- Coordinate System: WGS84 Geographic (EPSG:4326)

---

## 2. Gravity Data: EGM2008 Free-Air Anomaly

Since there is no single publicly available GeoTIFF file named `EGM2008_Free_Air_Anomaly.tif`, you have several options to obtain this data:

### Option A: ICGEM Calculation Service (Recommended for Custom Grids)

**Source:** International Centre for Global Earth Models (ICGEM)

**URL:** http://icgem.gfz-potsdam.de/calcgrid

**Instructions:**

1. Navigate to the ICGEM calculation service
2. Select Model: **EGM2008**
3. Select Functional: **Gravity disturbance** or **Free-air anomaly**
4. Set Grid Parameters:
   - Latitude: -90 to 90
   - Longitude: -180 to 180
   - Grid step: 0.1° (or desired resolution)
5. Select Output Format: **Grid (netCDF)**
6. Submit the calculation
7. Download the resulting netCDF file

**Convert to GeoTIFF:**
```bash
# Convert netCDF to GeoTIFF using GDAL
gdal_translate -of GTiff -a_srs EPSG:4326 \
  input_gravity.nc data/raw/EGM2008_Free_Air_Anomaly.tif

# Verify the output
gdalinfo data/raw/EGM2008_Free_Air_Anomaly.tif
```

### Option B: BGI (Bureau Gravimétrique International) WGM2012

**Source:** BGI (Observatoire Midi-Pyrénées)

**URL:** https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/

**Instructions:**

1. Visit the BGI website
2. Download the WGM2012 free-air anomaly grid
3. The file is typically in ASCII or binary grid format
4. Convert to GeoTIFF:

```bash
# If the file is in ASCII XYZ format
gdal_translate -of GTiff -a_srs EPSG:4326 -a_ullr -180 90 180 -90 \
  input_freeair.xyz data/raw/EGM2008_Free_Air_Anomaly.tif

# If it's in GMT grid format
gmt grdconvert input.grd data/raw/EGM2008_Free_Air_Anomaly.tif=gd:GTiff
```

### Option C: Compute from Spherical Harmonic Coefficients

**For Advanced Users**

If you need full control over the computation:

1. Download EGM2008 spherical harmonic coefficients from ICGEM or NGA
2. Use software to compute free-air anomaly:
   - Python: `pyshtools`, `harmonica`, or custom code
   - MATLAB: `gravbox` or similar
   - Fortran: NGA's official EGM2008 software

**Example Python workflow:**
```python
import pyshtools as pysh
import numpy as np
from osgeo import gdal, osr

# Load EGM2008 coefficients
coeffs = pysh.SHGravCoeffs.from_file('EGM2008_to360.gfc')

# Compute free-air anomaly on a grid
lats = np.linspace(90, -90, 1800)  # 0.1° resolution
lons = np.linspace(-180, 180, 3600)
gravity_anomaly = coeffs.expand(lat=lats, lon=lons, kind='free_air')

# Save as GeoTIFF
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create('data/raw/EGM2008_Free_Air_Anomaly.tif',
                       3600, 1800, 1, gdal.GDT_Float32)
dataset.SetGeoTransform((-180, 0.1, 0, 90, 0, -0.1))
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
dataset.SetProjection(srs.ExportToWkt())
dataset.GetRasterBand(1).WriteArray(gravity_anomaly)
dataset.FlushCache()
```

### Option D: Alternative Gravity Data Sources

If EGM2008 is difficult to obtain, consider these alternatives:

1. **EIGEN-6C4** (similar to EGM2008)
   - Available from ICGEM
   - Resolution: up to 2190 degree/order
   
2. **WGM2012** (BGI)
   - Global free-air anomaly compilation
   - Combines satellite and terrestrial data

3. **Sandwell & Smith Marine Gravity**
   - Good for ocean areas
   - Available from Scripps Institution of Oceanography

---

## Verification

After downloading or generating both files, verify they are correctly formatted:

```bash
cd data/raw

# Check EMAG2
gdalinfo EMAG2_V3_Sea_Level.tif

# Expected output should include:
# - Driver: GTiff/GeoTIFF
# - Size: 10800 x 5400 (for 2-minute resolution)
# - Coordinate System: WGS 84

# Check Gravity data
gdalinfo EGM2008_Free_Air_Anomaly.tif

# Expected output should include:
# - Driver: GTiff/GeoTIFF
# - Coordinate System: WGS 84
# - Data type: Float32
```

---

## File Size Expectations

| Dataset | Resolution | Approximate Size | Format |
|---------|-----------|------------------|---------|
| EMAG2 V3 | 2 arc-min | ~1.5 GB | GeoTIFF |
| EGM2008 FAA | 2.5 arc-min | ~1.2 GB | GeoTIFF |
| EGM2008 FAA | 0.1 degree | ~500 MB | GeoTIFF |

---

## Troubleshooting

### Missing GDAL Tools

If `gdal_translate` is not found:

**Windows:**
```bash
# Install via conda
conda install -c conda-forge gdal

# Or via OSGeo4W
# Download and run OSGeo4W installer from https://trac.osgeo.org/osgeo4w/
```

**Linux:**
```bash
sudo apt-get install gdal-bin  # Ubuntu/Debian
sudo yum install gdal         # RHEL/CentOS
```

**macOS:**
```bash
brew install gdal
```

### Format Conversion Issues

If you encounter issues converting formats:

1. Check the input file structure: `gdalinfo input_file`
2. Verify coordinate system: ensure WGS84/EPSG:4326
3. Check data type: gravity anomalies should be Float32 or Float64
4. Verify nodata values are properly set

---

## Ready to Process

Once both files are in `data/raw/`:

```bash
cd /path/to/GeoAnomalyMapper
python process_global_map.py
```

The script will:
1. Verify data file existence
2. Process all 648 global 10°×10° tiles
3. Generate Cloud-Optimized GeoTIFFs
4. Create a final PMTiles file for web visualization

---

## Additional Resources

- **ICGEM Service:** http://icgem.gfz-potsdam.de/
- **BGI Data Portal:** https://bgi.obs-mip.fr/
- **NOAA NCEI Geophysics:** https://www.ngdc.noaa.gov/mgg/global/
- **GDAL Documentation:** https://gdal.org/
- **PyGMT/SHTools:** For custom gravity field computations

---

## Support

For questions or issues with data acquisition:
- Check the project documentation
- Review GDAL error messages carefully
- Verify file paths and formats
- Ensure sufficient disk space (>3 GB recommended)