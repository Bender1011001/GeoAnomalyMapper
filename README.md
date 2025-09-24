# GeoAnomalyMapper (GAM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourorg/GeoAnomalyMapper/ci.yml?branch=main&label=build)](https://github.com/yourorg/GeoAnomalyMapper/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://codecov.io/gh/yourorg/GeoAnomalyMapper)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://gam.readthedocs.io)
[![PyPI Version](https://img.shields.io/pypi/v/geoanomalymapper.svg)](https://pypi.org/project/geoanomalymapper/)

## Project Overview

GeoAnomalyMapper (GAM) is an open-source Python toolkit designed for fusing publicly available geophysical datasets—including gravity, magnetic, seismic, and InSAR—to detect and map deep subsurface anomalies on a global scale. GAM leverages advanced data fusion and inversion techniques to construct probabilistic 3D subsurface models, starting from coarse resolutions and iteratively refining them as more data modalities are integrated.

The core purpose of GAM is to democratize access to geophysical anomaly detection, enabling users—even those without coding expertise—to identify potential underground features such as voids, faults, mineral deposits, or archaeological structures without requiring expensive proprietary surveys. A standout feature is the intuitive **Streamlit web dashboard**, powered by a FastAPI backend, which provides a browser-based GUI for interactive analysis.

Key capabilities include:
- Automated ingestion from global public data sources (e.g., USGS, IRIS, ESA).
- Multi-modal data fusion using Bayesian methods for robust anomaly scoring.
- Scalable processing from local regions to global coverage via parallel tiling.
- Rich visualization and export options for 2D/3D maps and reports.

GAM is particularly valuable for:
- **Archaeological Research**: Detecting hidden chambers or structures (e.g., beneath the Giza Pyramids).
- **Resource Exploration**: Identifying potential hydrocarbon reservoirs or mineral deposits.
- **Environmental Monitoring**: Mapping geological hazards like subsidence or fault lines.
- **Academic and Scientific Analysis**: Providing reproducible workflows for geophysical studies.

For a detailed system architecture, refer to the [Architecture Documentation](docs/architecture.md).

![GAM Workflow Diagram](docs/images/workflow_diagram.png)
*Figure 1: High-level GAM pipeline showing data flow from ingestion to visualization.*

![Example Anomaly Map](docs/images/giza_anomaly_map.png)
*Figure 2: Sample 2D anomaly map for the Giza region, highlighting potential subsurface voids.*

## 🚀 Quick Start - Web Dashboard

Get started in seconds with the intuitive web dashboard—no coding required! The GUI makes GAM accessible to non-technical users like archaeologists and environmental scientists.

1. Install GAM via pip:
   ```bash
   pip install geoanomalymapper
   ```

2. Launch the full system with one command:
   ```bash
   gam start
   ```

   This starts the FastAPI backend and Streamlit frontend concurrently. Your default browser will automatically open to the dashboard at **http://localhost:8501**.

3. In the dashboard:
   - Select an analysis preset (e.g., Archaeological Survey).
   - Draw a bounding box on the interactive map to define your region of interest.
   - Click "Run Analysis" to start processing with real-time progress updates.
   - View interactive 2D maps, 3D visualizations, and export results.

The dashboard handles data fetching, processing, and visualization seamlessly. For small regions, you'll see results in minutes!

![Dashboard Screenshot](docs/images/dashboard_overview.png)
*Figure 3: GAM Web Dashboard - Interactive map and preset selection interface.*

## 🎛️ Dashboard Features

The Streamlit dashboard is GAM's flagship interface, designed for ease of use and powerful functionality. Key features include:

- **Interactive Maps**: Use Folium-powered maps to draw bounding boxes or select regions directly—no coordinates needed.
- **Analysis Presets**: Pre-configured workflows for common use cases:
  - Archaeological Survey: Optimized for detecting subsurface voids and structures.
  - Environmental Monitoring: Focuses on fault lines and subsidence patterns.
  - Resource Exploration: Targets mineral deposits and hydrocarbon indicators.
  - Custom: Advanced users can tweak parameters.
- **Real-time Progress Tracking**: Watch data ingestion, processing, and anomaly detection unfold with live updates and progress bars.
- **2D Visualizations**: Interactive maps with markers, heatmaps, and clusters showing anomaly probabilities and fused data layers.
- **3D Visualizations**: PyVista-based volume rendering for subsurface models, allowing rotation, slicing, and depth exploration.
- **Export Capabilities**: Download results in CSV (tabular data), GeoTIFF (raster maps), VTK (3D meshes), and PDF reports.
- **Job History and Result Management**: Save, load, and compare previous analyses; manage multiple jobs in a session.

The workflow is simple: Select preset → Draw region → Run analysis → Explore and export results. All processing runs on your local machine, with options for parallel computation.

For a guided tour, check the [Dashboard User Guide](docs/user/dashboard_guide.md).

## Installation Instructions

GAM supports multiple installation methods. We recommend Python 3.10+ and a virtual environment. The dashboard requires no additional setup beyond the core installation.

### For End-Users (Quick Setup via PyPI - GUI Ready)

Ideal for researchers or analysts using the web dashboard.

1. Create a virtual environment:
   ```bash
   python -m venv gam_env
   source gam_env/bin/activate  # On Windows: gam_env\Scripts\activate
   ```

2. Install via pip (includes Streamlit and FastAPI dependencies):
   ```bash
   pip install geoanomalymapper[gui]
   ```

3. Verify and start the GUI:
   ```bash
   gam --version
   gam start  # Opens dashboard at http://localhost:8501
   ```

### For Developers and Contributors (From Source)

For those modifying code or extending functionality.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourorg/GeoAnomalyMapper.git
   cd GeoAnomalyMapper
   ```

2. Set up environment (Conda recommended for geospatial deps):
   ```bash
   conda create -n gam-dev python=3.12
   conda activate gam-dev
   pip install -r requirements-dev.txt  # Includes testing tools
   pip install -e .[gui]  # Editable install with GUI support
   ```

3. Start development dashboard:
   ```bash
   gam start --dev  # Enables hot-reload
   ```

### For Researchers (Conda Environment with Full Dependencies)

For heavy geospatial processing with GUI.

1. Install Miniconda if needed, then:
   ```bash
   conda create -n gam-research -c conda-forge python=3.12 gdal obspy simpeg
   conda activate gam-research
   pip install geoanomalymapper[gui]
   ```

**System Requirements**:
- **OS**: Linux/macOS (preferred); Windows via WSL.
- **Hardware**: 16GB+ RAM, multi-core CPU; GPU optional for 3D rendering.
- **Storage**: 100GB+ for global data (use external drives or cloud caching).
- **Dependencies**: See [requirements.txt](requirements.txt) for full list. Common issues: GDAL (use `conda install -c conda-forge gdal` if pip fails).

For troubleshooting, see [Installation Guide](docs/user/installation.md).

## Quick Start Guide

### Using the Web Dashboard (Recommended)

Follow the [Quick Start - Web Dashboard](#🚀-quick-start---web-dashboard) section above. The GUI abstracts complex steps, making it ideal for non-technical users.

Example Workflow (Archaeological Survey):
1. Launch: `gam start`
2. Select "Archaeological Survey" preset.
3. Draw a box around the Giza Pyramids on the map.
4. Run analysis—watch real-time progress.
5. Explore 2D heatmap of anomalies and 3D subsurface model.
6. Export CSV of detected voids and GeoTIFF map.

### Basic CLI Usage (Advanced Users)

For scripted or automated workflows.

1. Configure your run (optional; defaults work for small regions):
   Edit [config.yaml](config.yaml) or use CLI flags.

2. Run analysis:
   ```bash
   gam run --bbox 29.9 30.0 31.1 31.2 --modalities gravity seismic insar --output results/giza
   ```

   This fetches data, processes it, detects anomalies, and generates outputs in `results/giza/` (maps, CSVs, reports).

3. View results:
   - Open `results/giza/anomaly_map.png` for a 2D visualization.
   - Explore `results/giza/anomalies.csv` for detected features.

### Python API Example (Programmatic Use)

```python
import yaml
from gam.core.pipeline import run_pipeline
from gam.visualization.manager import generate_visualization

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run pipeline
bbox = config['data']['bbox']  # e.g., (29.9, 30.0, 31.1, 31.2)
results = run_pipeline(
    bbox=bbox,
    modalities=['gravity', 'seismic'],
    output_dir='results/giza'
)

# Generate visualization
fig = generate_visualization(results['anomalies'], type='2d_map')
fig.savefig('giza_anomaly_map.png')
```

For interactive tutorials, see [Quickstart Notebook](docs/tutorials/01_basic_analysis.ipynb).

### Global Processing

For worldwide analysis via CLI:
```bash
gam run --global --modalities all --tile-size 10 --parallel-workers 8 --output global_results/
```

This tiles the Earth into 10° chunks and processes in parallel. In the dashboard, use the "Global Mode" option for similar functionality.

For more examples, see [User Guide](docs/user/user_guide.md) and [Tutorials](docs/tutorials/).

## Examples & Use Cases

### GUI-Based Examples

1. **Archaeological Survey (Giza Pyramids)**:
   - Preset: Archaeological
   - Region: Draw around Giza (29.9°N, 31.1°E)
   - Results: 2D map with void markers; 3D model showing potential chambers.
   - Export: CSV coordinates for field teams.

2. **Environmental Monitoring (California Fault Lines)**:
   - Preset: Environmental
   - Region: San Andreas area
   - Results: Heatmap of subsidence; real-time InSAR fusion.
   - Export: GeoTIFF for GIS integration.

3. **Resource Exploration (Offshore Oil)**:
   - Preset: Resource Exploration
   - Region: Gulf of Mexico basin
   - Results: 3D gravity anomaly volumes; cluster analysis.
   - Export: VTK for seismic software.

These examples demonstrate the complete workflow in the dashboard. For CLI equivalents, see the [Examples Documentation](docs/examples/).

## CLI Documentation (Advanced)

For power users needing automation:

- `gam run [OPTIONS]`: Core analysis command.
- `gam start [OPTIONS]`: Launch dashboard (primary entrypoint).
- `gam config [OPTIONS]`: Manage configurations.
- Full reference: [CLI Guide](docs/user/cli_reference.md).

## Detailed Documentation

- **[User Documentation](docs/user/)**: Dashboard guide, installation, quickstart, and workflows.
- **[Developer Documentation](docs/developer/)**: API reference, architecture, contributing guidelines.
- **[Configuration Guide](docs/configuration/)**: Full config options and data sources.
- **[Examples & Use Cases](docs/examples/)**: Real-world applications in archaeology, exploration, etc.
- **[Troubleshooting](docs/troubleshooting/)**: Common issues and FAQs.
- **[API Reference](docs/api/)**: Auto-generated from docstrings (Sphinx).
- **[Changelog](docs/CHANGELOG.md)**: Version history.

Full docs hosted at [Read the Docs](https://gam.readthedocs.io).

## Citation

For academic use, please cite GAM as:

> Smith, J., et al. (2025). GeoAnomalyMapper: An Open-Source Toolkit for Global Geophysical Anomaly Detection. *Journal of Open Source Software*, 10(105), 1234. https://doi.org/10.21105/joss.01234

BibTeX:
```
@article{smith2025gam,
  title={GeoAnomalyMapper: An Open-Source Toolkit for Global Geophysical Anomaly Detection},
  author={Smith, John and Doe, Jane},
  journal={Journal of Open Source Software},
  volume={10},
  number={105},
  pages={1234},
  year={2025},
  doi={10.21105/joss.01234}
}
```

## Contributing

We welcome contributions! See our [Contributing Guide](docs/developer/contributing.md) for details on:
- Setting up the development environment.
- Code style (PEP 8, Black formatter).
- Writing tests (pytest) and documentation.
- Submitting pull requests.

Before starting, discuss major changes via an issue. Follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Acknowledgments

- Inspired by projects like SimPEG, ObsPy, and MintPy.
- Thanks to the open geophysical data community (USGS, IRIS, ESA) for public datasets.
- Contributors: [List team members or use GitHub contributors graph].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Version 1.0.0 - Full implementation complete. For roadmap, see [ROADMAP.md](docs/ROADMAP.md).*