# Mineral Exploration Mode Guide

This guide details how to use the **Mineral Exploration Mode** in GeoAnomalyMapper. Unlike the default "Void Detection" mode which searches for mass deficits (caves, tunnels), this mode "flips the physics" to detect **Mass Excess** (dense ore bodies, mineral deposits).

## 1. Overview

The Mineral Exploration Mode modifies the underlying physics engines to target high-density anomalies associated with mineral systems:

*   **Gravity Inversion:** Penalizes negative density contrasts, favoring positive mass solutions.
*   **Poisson Analysis:** Looks for positive correlations between Gravity and Magnetic fields (common in iron-rich ore bodies).
*   **Fusion Logic:** Prioritizes high-density, high-magnetic-susceptibility targets.

## 2. Data Requirements

To run this mode effectively, you must provide authentic geophysical data. The pipeline expects the following files in the `data/raw/` directory:

### A. Gravity Data
*   **File:** `data/raw/gravity/gravity_anomaly.tif`
*   **Format:** GeoTIFF (Float32)
*   **Source:** Bouguer Gravity Anomaly (e.g., USGS Isostatic Gravity or WGM2012).
*   **Note:** Do not use "worms" or gradient maps; the physics engine requires the raw field values (mGal).

### B. Magnetic Data
*   **File:** `data/raw/magnetic/EMAG2_V3_SeaLevel_DataTiff.tif`
*   **Format:** GeoTIFF (Float32)
*   **Source:** EMAG2v3 Total Magnetic Intensity (TMI).
*   **Note:** Ensure you have the data version (nT values), not the RGB visualization.

### C. Lithology (Geology)
*   **File:** `data/raw/lithology/glim_wgs84_0.5deg.tif`
*   **Format:** GeoTIFF (Categorical/Integer)
*   **Source:** GLiM (Global Lithological Map) v1.0.
*   **Setup:** If you downloaded the GLiM data as an ASCII grid (`.asc`), place it in `data/raw/lithology/` and run the setup script:
    ```bash
    python setup_data.py
    ```

### D. InSAR (Surface Deformation)
*   **File:** `data/processed/insar/mosaics/usa_winter_vv_COH12.vrt`
*   **Source:** Sentinel-1 Coherence (processed via LiCSAR or similar).

## 3. Running the Workflow

To execute the pipeline in Mineral Exploration Mode, use the `--mode mineral` flag.

**Command:**
```bash
python workflow.py \
  --region "-116.0,35.0,-115.0,36.0" \
  --resolution 0.001 \
  --output-name "outputs/mountain_pass_run" \
  --mode mineral
```

**Arguments:**
*   `--region`: Bounding box in `min_lon,min_lat,max_lon,max_lat` format.
*   `--resolution`: Pixel size in degrees (0.001 ≈ 100m).
*   `--mode`: Set to `mineral` to target ore bodies (default is `void`).

## 4. Validation

We have included a validation script specifically for mineral targets, calibrated against major US deposits (Mountain Pass, Bingham Canyon, Red Dog, etc.).

**Command:**
```bash
python validate_mining.py outputs/mountain_pass_run_mineral_void_probability.tif
```

This will generate a report detailing the overlap between detected anomalies and known mineral deposits.

## 5. Troubleshooting

*   **"Argument --region: expected one argument":** Ensure your region string is enclosed in quotes and does not have spaces after commas (e.g., `"-116.0,35.0..."`).
*   **"File not found":** Run `python setup_data.py` to check for missing files and perform necessary conversions (like `.asc` to `.tif`).