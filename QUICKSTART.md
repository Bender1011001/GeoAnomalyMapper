# GeoAnomalyMapper - Quick Start Guide

## 🚀 Simple 3-Step Workflow

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up Copernicus credentials (optional, for InSAR data)
# Create .env file in GeoAnomalyMapper/ directory:
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```

### Step 1: Download Data
```bash
cd GeoAnomalyMapper

# Example: Carlsbad Caverns area (New Mexico)
python download_data.py --region "-105.0,32.0,-104.0,33.0"

# Skip InSAR for faster download (gravity/magnetic only)
python download_data.py --region "-105.0,32.0,-104.0,33.0" --skip-insar
```

**What it downloads:**
- ✅ Gravity data (XGM2019e model file)
- ✅ Magnetic data (EMAG2v3 global)
- ⚠️  Elevation data (instructions provided)
- ⚠️  InSAR data (optional, needs credentials)
- ⚠️  Lithology data (instructions provided)

### Step 2: Process Data
```bash
# Process all downloaded data for your region
python process_data.py --region "-105.0,32.0,-104.0,33.0"
```

**What it does:**
- Clips global datasets to your region
- Resamples to uniform resolution (~100m)
- Saves processed data in `data/processed/`

**Note:** Some data requires manual steps (see instructions in `data/raw/`)

### Step 3: Detect Voids
```bash
# Run void detection analysis
python detect_voids.py --region "-105.0,32.0,-104.0,33.0"
```

**Output files** (in `data/outputs/void_detection/`):
- `void_probability.tif` - GeoTIFF probability map
- `void_probability.png` - Visualization
- `void_probability_report.txt` - Analysis report

---

## 📊 Understanding the Results

### Probability Map
- **0.0-0.3 (Low)**: Unlikely to contain voids
- **0.3-0.7 (Medium)**: Possible void signatures
- **0.7-1.0 (High)**: Strong void indicators

### Data Layers
The analysis combines multiple data sources:
- **Gravity**: Negative anomalies = mass deficits (potential voids)
- **Magnetic**: Disruptions in magnetic field
- **InSAR**: Ground subsidence over voids
- **Lithology**: Karst-prone rock types (limestone, dolomite)

---

## 🗺️ Example Regions

### Carlsbad Caverns, New Mexico
```bash
python download_data.py --region "-105.0,32.0,-104.0,33.0"
python process_data.py --region "-105.0,32.0,-104.0,33.0"
python detect_voids.py --region "-105.0,32.0,-104.0,33.0"
```

### Mammoth Cave, Kentucky
```bash
python download_data.py --region "-86.2,37.1,-86.0,37.3"
python process_data.py --region "-86.2,37.1,-86.0,37.3"
python detect_voids.py --region "-86.2,37.1,-86.0,37.3"
```

### San Andreas Fault, California
```bash
python download_data.py --region "-122.0,36.0,-121.0,37.0"
python process_data.py --region "-122.0,36.0,-121.0,37.0"
python detect_voids.py --region "-122.0,36.0,-121.0,37.0"
```

---

## 🔧 Troubleshooting

### Issue: No gravity GeoTIFF found
**Solution:** Convert the .gfc file manually:
1. Visit http://icgem.gfz-potsdam.de/calcgrid
2. Select model: XGM2019e_2159
3. Set your region and download as GeoTIFF
4. Save to `data/raw/gravity/`

### Issue: InSAR download fails
**Solution:** InSAR is optional. Either:
- Skip it with `--skip-insar` flag
- Use pre-processed data from COMET LiCSAR
- Set up Copernicus credentials correctly

### Issue: Insufficient data error
**Solution:** You need at least ONE of:
- Gravity data
- Magnetic data

Complete the manual download steps shown in processing output.

---

## 📁 Project Structure

```
SAR-project/
├── GeoAnomalyMapper/
│   ├── download_data.py      # Step 1: Download all data
│   ├── process_data.py        # Step 2: Process data
│   ├── detect_voids.py        # Step 3: Detect voids
│   └── .env                   # Credentials (create this)
├── data/
│   ├── raw/                   # Downloaded data
│   ├── processed/             # Processed data
│   └── outputs/               # Results
│       └── void_detection/
│           ├── void_probability.tif
│           ├── void_probability.png
│           └── void_probability_report.txt
```

---

## 🎯 Advanced Options

### High-Resolution Analysis
```bash
# Use finer resolution (10m instead of 100m)
python process_data.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.0001
python detect_voids.py --region "-105.0,32.0,-104.0,33.0" --resolution 0.0001
```

### Visualize Results
```bash
# Create visualization
python create_visualization.py
```

### Validate Results
```bash
# Compare with known features
python validate_against_known_features.py
```

---

## 📚 Additional Documentation

- **multi_resolution_fusion.py** - Advanced multi-resolution data fusion
- **process_insar_data.py** - Detailed InSAR processing guide
- **SECURITY.md** - Security and credential management
- **README.md** - Full project documentation

---

## ⚠️ Important Notes

1. **Gravity data** requires manual conversion from .gfc to GeoTIFF
2. **InSAR data** is optional but provides better results
3. **Minimum requirement**: Gravity OR Magnetic data
4. **Processing time**: ~5-30 minutes depending on region size
5. **Disk space**: ~500MB - 2GB per region

---

## 🆘 Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Review instructions in `data/raw/` directories
3. Consult processing guides in `data/processed/`
4. Verify your region coordinates format

---

## ✅ Success Checklist

- [ ] Python environment set up
- [ ] Data downloaded (at least gravity OR magnetic)
- [ ] Manual steps completed (if needed)
- [ ] Data processed successfully
- [ ] Void detection completed
- [ ] Results reviewed

---

Happy void hunting! 🕳️🔍