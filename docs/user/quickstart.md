# Quickstart Guide

## Overview

Welcome to GeoAnomalyMapper (GAM)! This guide walks you through your first analysis using GAM to detect subsurface anomalies in the Giza Pyramids region (Egypt). We'll cover installation verification, basic CLI usage, Python API, understanding outputs, and common workflows.

This tutorial assumes you've [installed GAM](installation.md). It should take 15-30 minutes for a local run (longer for global). The example uses public data sources—no API keys needed initially.

**Prerequisites**:
- GAM installed (core + geophysics/visualization extras recommended).
- Internet access for data fetching (caches afterward).
- Sample config: Use the provided [config.yaml](../config.yaml) or create your own.

## Step 1: Verify Installation and Setup

1. **Activate Environment**:
   ```bash
   # If using venv/conda
   conda activate gam  # Or source gam_env/bin/activate
   ```

2. **Check GAM**:
   ```bash
   gam --version
   gam --help
   ```
   Expected: Version info and CLI options. If `gam` not found, reinstall with `pip install -e .`.

3. **Prepare Configuration** (optional but recommended):
   Copy [config.yaml](../config.yaml) to your working directory and edit for Giza:
   ```yaml
   data:
     bbox: [29.9, 30.0, 31.1, 31.2]  # Giza: min_lat, max_lat, min_lon, max_lon
     modalities: ["gravity", "seismic", "insar"]  # Start with these
   core:
     output_dir: "results/giza"  # Custom output folder
   visualization:
     map_type: "2d"  # Simple 2D map
   ```
   For data sources, ensure [data_sources.yaml](../data_sources.yaml) is in place. For InSAR (Sentinel-1), register a free account at [Copernicus Open Access Hub](https://scihub.copernicus.eu/) and set env vars:
   ```bash
   export COPERNICUS_USERNAME=your_username
   export COPERNICUS_PASSWORD=your_password
   ```

## Step 2: Run Your First Analysis (CLI)

GAM's CLI (`gam run`) orchestrates the full pipeline: ingestion → preprocessing → modeling → anomaly detection → visualization.

1. **Basic Local Run** (Giza Region):
   ```bash
   gam run --bbox 29.9 30.0 31.1 31.2 --modalities gravity seismic insar --output results/giza
   ```
   
   **What Happens**:
   - **Ingestion**: Fetches gravity (USGS), seismic (IRIS FDSN), InSAR (Copernicus) data for the bbox.
   - **Preprocessing**: Aligns to a 0.1° grid, filters noise, standardizes units.
   - **Modeling**: Performs joint inversion (linear for gravity, eikonal for seismic, elastic for InSAR), fuses modalities.
   - **Detection**: Flags anomalies with z-score > 2.0 (configurable).
   - **Visualization**: Generates 2D map, 3D volume (if enabled), and exports.

   Progress is logged to console (INFO level). First run downloads ~100MB data; subsequent runs use cache.

2. **With Custom Config**:
   ```bash
   gam run --config config_giza.yaml --parallel-workers 4
   ```
   This uses your YAML file for advanced options like grid resolution or thresholds.

3. **Monitor Output**:
   Look for logs like:
   ```
   INFO:gam.core:Fetching gravity data for bbox [29.9, 30.0, 31.1, 31.2]
   INFO:gam.modeling:Detected 15 anomalies with confidence > 0.7
   INFO:gam.visualization:Exported map to results/giza/anomaly_map.png
   ```

   Run time: 5-15 minutes on a standard laptop.

## Step 3: Explore Outputs

Outputs are saved in `results/giza/` (or specified dir). Key files:

- **anomalies.csv**: Tabular results (Pandas DataFrame).
  | lat | lon | depth_m | confidence | anomaly_type | score | modalities_contrib |
  |-----|-----|---------|------------|--------------|-------|--------------------|
  | 29.95 | 31.13 | -150 | 0.85 | void | 2.5 | {'gravity': 0.6, 'seismic': 0.4} |
  - Columns: Location (lat/lon/depth), confidence (0-1), type (e.g., 'void', 'fault'), z-score, contributions from each modality.
  - Use: Load with `pd.read_csv('anomalies.csv')` for further analysis.

- **anomaly_map.png / anomaly_map.geotiff**: 2D heatmap of anomaly confidence (viridis colormap).
  - Red/yellow hotspots indicate high-confidence anomalies (e.g., potential voids under pyramids).
  - GeoTIFF includes geospatial metadata for GIS tools (QGIS/ArcGIS).

- **model_fused.h5**: HDF5 file with 3D fused model (xarray.Dataset).
  - Load: `import h5py; f = h5py.File('model_fused.h5', 'r')`.
  - Contains grids for each modality + fused uncertainty.

- **report.html**: Interactive summary (if Folium enabled).
  - Includes maps, stats (e.g., "15 anomalies detected, avg depth 200m"), and interpretations.

- **Logs**: `gam.log` with detailed steps for debugging.

**Interpreting Results**:
- High-confidence (>0.7) anomalies near known sites (e.g., 29.979° N, 31.134° E for Great Pyramid) may indicate subsurface features.
- Cross-check with literature: Giza voids reported at ~140m depth align with GAM outputs.
- If no anomalies: Try finer grid (`grid_res: 0.01`) or more modalities.

![Sample Giza Output](images/giza_sample_map.png)
*Figure: Example 2D anomaly map for Giza, with hotspots indicating potential voids.*

## Step 4: Python API Usage (Interactive)

For Jupyter or scripts, use the API for fine control.

1. **Basic Pipeline**:
   ```python
   import yaml
   from gam.core.pipeline import run_pipeline
   from gam.visualization.manager import generate_visualization
   import pandas as pd

   # Load config
   with open('config_giza.yaml', 'r') as f:
       config = yaml.safe_load(f)

   # Run analysis
   bbox = config['data']['bbox']
   results = run_pipeline(
       bbox=bbox,
       modalities=config['data']['modalities'],
       config=config,
       output_dir='results/giza'
   )

   # Load and explore anomalies
   anomalies = pd.read_csv('results/giza/anomalies.csv')
   print(anomalies[anomalies['confidence'] > 0.8].head())

   # Generate custom viz
   fig = generate_visualization(
       anomalies,
       type='interactive',  # Folium map
       color_scheme='plasma'
   )
   fig.save('giza_interactive.html')
   ```

2. **Modular Usage** (Advanced):
   ```python
   from gam.ingestion.manager import fetch_data
   from gam.preprocessing.manager import preprocess_data
   from gam.modeling.manager import model_and_detect

   # Fetch
   raw_data = fetch_data(bbox=(29.9, 30.0, 31.1, 31.2), modalities=['gravity'])

   # Preprocess
   processed = preprocess_data(raw_data, grid_res=0.05)

   # Model
   anomalies = model_and_detect(processed, threshold=2.5)
   print(f"Detected {len(anomalies)} anomalies")
   ```

   See [API Reference](../developer/api_reference.md) for full details.

## Common Workflows and Use Cases

### Workflow 1: Local Site Analysis (e.g., Archaeology)
- Use small bbox (0.1° span), 2-3 modalities.
- Focus on high-res grid (0.01°), export to GeoTIFF for GIS overlay.
- Example: Giza for void detection.

### Workflow 2: Regional Exploration (e.g., Mineral Resources)
- Larger bbox (1-5°), all modalities.
- Enable 3D visualization (`map_type: "3d"`), export VTK for ParaView.
- Tune priors for sparsity (`regularization: "l1"`).

### Workflow 3: Global Monitoring (e.g., Hazards)
- Use `--global` flag with tile_size=5 for worldwide fault mapping.
- Parallelize (`parallel_workers: -1`), monitor with Dask dashboard.
- Example: `gam run --global --modalities seismic --output global_faults/`.

**Tips**:
- Start small: Test with gravity-only to verify setup.
- Cache Management: Clear old cache with `gam cache clear --modality gravity` if data updates.
- Performance: For large runs, use a machine with 32GB+ RAM or cloud (e.g., AWS EC2 with Dask-Kubernetes).
- Best Practices: Always validate outputs against known geology; combine with field data.

## Next Steps

- **Advanced Tutorials**: See [Multi-Modality Fusion](../tutorials/02_multi_modality_fusion.ipynb) for joint inversion.
- **CLI Reference**: Full options in [User Guide](user_guide.md).
- **Examples**: Real-world cases in [Examples](../examples/archaeological.md).
- **Troubleshooting**: If errors occur (e.g., API timeouts), see [Common Issues](../troubleshooting/common_issues.md).

Congratulations! You've run your first GAM analysis. Share your results or contribute via [Contributing Guide](../developer/contributing.md).

---

*Last Updated: 2025-09-23 | GAM v1.0.0*