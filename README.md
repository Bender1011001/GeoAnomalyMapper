# GeoAnomalyMapper v2.0

**Advanced Geophysical Anomaly Detection & Multi-Scale Fusion Pipeline**

GeoAnomalyMapper v2.0 is a complete rebuild of the anomaly detection system, designed to achieve >95% accuracy in detecting Deep Underground Military Bunkers (DUMB) and other subterranean anomalies. It addresses previous limitations through a sophisticated multi-stage pipeline integrating signal processing, physics-informed analysis, and machine learning.

## Key Features (v2.0)

*   **Multi-Scale Fusion:** Combines Bayesian Compressive Sensing (BCS) for resolution enhancement with Dempster-Shafer theory for uncertainty-weighted belief fusion.
*   **Stable Structure Detection:** Leverages InSAR Coherence Change Detection (CCD), GLCM texture analysis, and structural artificiality metrics to identify surface footprints of underground structures.
*   **Physics-Informed Analysis:** Utilizes Poisson's relation between gravity and magnetic fields to validate density contrasts against magnetic susceptibility, reducing false positives from geological features.
*   **Advanced Signal Processing:** Implements Continuous Wavelet Transform (CWT) for multi-scale decomposition and Tilt Derivative (TDR) for precise edge detection of potential voids.
*   **ML Classification:** Deploys One-Class SVM (OC-SVM) and Isolation Forest models trained on fused belief maps to probabilistically classify anomalies.

See [`ARCHITECTURE_v2.md`](ARCHITECTURE_v2.md) for the full technical specification.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/GeoAnomalyMapper.git
    cd GeoAnomalyMapper
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The core of v2.0 is the `workflow.py` orchestrator, which manages the entire pipeline from raw data processing to final anomaly classification.

### Running the Full Pipeline

```bash
python workflow.py --region "lon_min,lat_min,lon_max,lat_max" --resolution 0.001 --output-name "outputs/project_name"
```

**Example:**
```bash
python workflow.py --region "-105.5,31.5,-103.5,33.5" --resolution 0.001 --output-name "outputs/carlsbad_v2"
```

### Command Line Arguments

*   `--region`: Bounding box in WGS84 coordinates (min_lon, min_lat, max_lon, max_lat).
*   `--resolution`: Output grid resolution in degrees (default: 0.001, approx 100m).
*   `--output-name`: Prefix for all generated output files (includes directory path).
*   `--skip-visuals`: Flag to skip generation of PNG/KMZ visualizations (useful for batch processing).

## Pipeline Phases & Outputs

The workflow executes in 6 sequential steps. All outputs are GeoTIFFs prefixed with the `output-name` provided.

1.  **Gravity Processing:**
    *   `_gravity_residual.tif`: CWT-decomposed residual gravity (local anomalies).
    *   `_gravity_tdr.tif`: Tilt Derivative edge detection.

2.  **InSAR Feature Extraction:**
    *   `_coherence_change.tif`: Temporal coherence stability.
    *   `_structural_artificiality.tif`: Combined metric for man-made structure likelihood.

3.  **Poisson Analysis:**
    *   `_poisson_correlation.tif`: Correlation between gravity and magnetic fields (validates voids).

4.  **Bayesian Fusion:**
    *   `_gravity_prior_highres.tif`: Downscaled gravity map using higher-resolution covariates (DEM, InSAR).

5.  **Dempster-Shafer Fusion:**
    *   `_fused_belief_reinforced.tif`: Combined belief map representing the probability of a void, weighted by source uncertainty.

6.  **Anomaly Classification:**
    *   `_dumb_probability_v2.tif`: **Final Output**. Probability map of DUMB presence (>95% confidence target).

## Repository Layout

```
.
├── workflow.py                      # Main CLI orchestrator
├── process_data.py                  # Gravity/Magnetic processing (CWT, TDR)
├── insar_features.py                # InSAR CCD, GLCM, Artificiality
├── poisson_analysis.py              # Physics-informed correlation
├── multi_resolution_fusion.py       # Bayesian Compressive Sensing
├── detect_voids.py                  # Dempster-Shafer Fusion
├── classify_anomalies.py            # OC-SVM & Isolation Forest
├── utils/                           # Shared raster and math utilities
├── tests/                           # Unit and integration tests
├── ARCHITECTURE_v2.md               # Technical documentation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Data Configuration

The system expects raw data in a `data/` directory (not version controlled). You can override this by setting the `GEOANOMALYMAPPER_DATA_DIR` environment variable.

**Expected Structure:**
```
data/
├── raw/
│   ├── gravity/    # EGM2008 or similar
│   ├── magnetic/   # EMAG2
│   ├── dem/        # SRTM/Copernicus
│   └── insar/      # Sentinel-1 Coherence
```

## License

MIT License.
