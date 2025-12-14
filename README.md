# GeoAnomalyMapper

**Multi-Source Geophysical Anomaly Detection Pipeline**

GeoAnomalyMapper is a an advanced anomaly detection system designed to identify subsurface geological features. It uses a multi-stage pipeline integrating signal processing, physics-informed analysis, and machine learning.

## Key Features

*   **Multi-Scale Fusion:** Combines Bayesian Compressive Sensing (BCS) for resolution enhancement with Dempster-Shafer theory for uncertainty-weighted belief fusion.
*   **Stable Structure Detection:** Leverages InSAR Coherence Change Detection (CCD), GLCM texture analysis, and structural artificiality metrics.
*   **Physics-Informed Analysis:** Utilizes Poisson's relation between gravity and magnetic fields to validate density contrasts.
*   **Advanced Signal Processing:** Implements Continuous Wavelet Transform (CWT) and Tilt Derivative (TDR) for edge detection.
*   **ML Classification:** Deploys One-Class SVM (OC-SVM) and Isolation Forest models trained on fused belief maps.

See [`ARCHITECTURE_v2.md`](ARCHITECTURE_v2.md) for the technical specification.

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

The core of the system is the `workflow.py` orchestrator.

### Running the Pipeline

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
*   `--output-name`: Prefix for all generated output files.
*   `--mode`: Target mode. Options: `void` (default) or `mineral`.
*   `--skip-visuals`: Flag to skip generation of PNG/KMZ visualizations.

## Pipeline Phases & Outputs

The workflow executes in 6 sequential steps. All outputs are GeoTIFFs prefixed with the `output-name` provided.

1.  **Gravity Processing:**
    *   `_gravity_residual.tif`: CWT-decomposed residual gravity.
    *   `_gravity_tdr.tif`: Tilt Derivative edge detection.

2.  **InSAR Feature Extraction:**
    *   `_coherence_change.tif`: Temporal coherence stability.
    *   `_structural_artificiality.tif`: Combined metric for man-made structure likelihood.

3.  **Poisson Analysis:**
    *   `_poisson_correlation.tif`: Correlation between gravity and magnetic fields.

4.  **Bayesian Fusion:**
    *   `_gravity_prior_highres.tif`: Downscaled gravity map.

5.  **Dempster-Shafer Fusion:**
    *   `_fused_belief_reinforced.tif`: Combined belief map weighted by source uncertainty.

6.  **Anomaly Classification:**
    *   `_mineral_void_probability.tif`: **Final Output**. Probability map of target presence.

## Repository Layout

```
.
├── workflow.py                      # Main CLI orchestrator
├── process_data.py                  # Gravity/Magnetic processing
├── insar_features.py                # InSAR/texture analysis
├── poisson_analysis.py              # Physics-informed correlation
├── multi_resolution_fusion.py       # Bayesian Compressive Sensing
├── detect_voids.py                  # Dempster-Shafer Fusion
├── classify_anomalies.py            # OC-SVM & Isolation Forest
├── utils/                           # Utilities
├── tests/                           # Tests
├── ARCHITECTURE_v2.md               # Documentation
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Data Configuration

The system expects raw data in a `data/` directory (not version controlled). You can override this by setting the `GEOANOMALYMAPPER_DATA_DIR` environment variable.

## License

MIT License.
