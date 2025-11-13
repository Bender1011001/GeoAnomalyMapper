# InSAR Processing Guide for Anomaly Detection

Downloaded scenes: 5
Location: data\raw\insar\sentinel1

---

## Processing Options

### Option 1: SNAP (ESA Sentinel Application Platform)

**Best for:** Beginners, graphical workflow

**Installation:**
1. Download: https://step.esa.int/main/download/snap-download/
2. Install SNAP Desktop
3. Install Sentinel-1 Toolbox

**Processing Steps:**
```
1. Open SNAP Desktop
2. File > Import > SAR Sensors > Sentinel-1 > SLC
3. Navigate to: data\raw\insar\sentinel1
4. Select two scenes (master + slave) from same track
5. Radar > Interferometric > Products > InSAR Stack Overview
6. Create interferogram:
   - TOPSAR Split (select same sub-swath and burst)
   - Apply Orbit File
   - Back-Geocoding
   - Interferogram Formation
   - TOPSAR Deburst
   - TopoPhase Removal (with SRTM DEM)
   - Goldstein Phase Filtering
   - Coherence Estimation
   - Terrain Correction
7. Export as GeoTIFF
```

### Option 2: ISCE2 (Advanced Processing)

**Best for:** Advanced users, batch processing

**Installation:**
```bash
conda install -c conda-forge isce2
```

**Processing Example:**
```python
from isce.applications import topsApp

# Create topsApp.xml config
# Run: topsApp.py topsApp.xml
```

### Option 3: Cloud Processing (Easiest)

**Best for:** Quick results without local processing

**COMET LiCSAR Portal:**
1. Visit: https://comet.nerc.ac.uk/COMET-LiCS-portal/
2. Search for your region
3. Download pre-processed interferograms
4. Already geocoded and phase-unwrapped!

---

## Expected Outputs

For integration with GeoAnomalyMapper, process to get:

1. **Coherence** (GeoTIFF): Measures surface stability
   - High coherence = stable surface
   - Low coherence = changes/decorrelation

2. **Unwrapped Phase** (GeoTIFF): Ground deformation
   - Converted to displacement (cm or mm)
   - Negative = subsidence (potential voids!)
   - Positive = uplift

3. **Line-of-Sight Displacement** (GeoTIFF):
   - Final deformation map
   - Resolution: ~5-20 meters

---

## Integration with Fusion Pipeline

Once processed, save outputs to:
```
data\processed\insar/
├── coherence.tif          # Surface stability
├── displacement.tif       # Ground deformation (mm/year)
└── processing_metadata.txt
```

Then run fusion:
```powershell
python multi_resolution_fusion.py --include-insar --output with_insar
```

---

## Downloaded Scenes

```json
1. Unknown
2. Unknown
3. Unknown
4. Unknown
5. Unknown
```

---

## Quick Start (Recommended)

**If you don't want to process locally:**

1. Use COMET LiCSAR pre-processed data:
   https://comet.nerc.ac.uk/COMET-LiCS-portal/

2. Download interferograms for your region

3. Place in data/processed/insar/

4. Run fusion pipeline

**Processing time estimates:**
- SNAP GUI: 2-4 hours per interferogram
- ISCE batch: 1-2 hours per pair
- LiCSAR download: 5 minutes
