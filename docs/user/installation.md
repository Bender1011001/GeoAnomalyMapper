# Installation Guide

## Overview

This guide provides detailed, step-by-step instructions for installing GeoAnomalyMapper (GAM) for different user types: end-users seeking quick setup, developers contributing code, and researchers handling advanced geophysical processing. GAM requires Python 3.10 or higher and is optimized for Linux/macOS, with Windows support via WSL (Windows Subsystem for Linux) for best performance.

GAM's dependencies include scientific computing libraries (NumPy, SciPy), geospatial tools (GeoPandas, Rasterio), and optional geophysics/visualization packages. We recommend using a virtual environment to isolate dependencies and avoid conflicts.

**Estimated Time**: 10-30 minutes, depending on your method and internet speed.

## System Requirements

### Hardware
- **CPU**: Multi-core processor (4+ cores recommended for parallel processing).
- **RAM**: 16GB minimum; 32GB+ for global-scale analyses or large datasets.
- **Storage**: 100GB+ free space for caching datasets (use external drives or cloud storage for global runs).
- **GPU**: Optional, but beneficial for advanced inversions if using compatible backends (e.g., CuPy with SimPEG).

### Software
- **Python**: 3.10 or higher (3.12 recommended).
- **Operating System**: 
  - Linux (Ubuntu 20.04+ or equivalent) or macOS (10.15+).
  - Windows: Use WSL2 with Ubuntu; native Windows may require additional setup for GDAL.
- **Internet**: Required for initial data ingestion from public APIs (e.g., USGS, IRIS); subsequent runs use caching.
- **Other**: Git for cloning the repository; Conda (Miniconda/Anaconda) recommended for geospatial dependencies.

### Dependencies Overview
GAM's core dependencies are listed in [requirements.txt](../requirements.txt). Key categories:

- **Core Scientific**: NumPy (>=1.26.0), SciPy (>=1.14.0), Pandas (>=2.2.0), xarray (>=2024.3.0), Matplotlib (>=3.9.0).
- **Utilities**: Requests (>=2.31.0) for API calls, PyYAML (>=6.0.2) for config, Click (>=8.1.0) for CLI.
- **Parallelism**: Dask[complete] (>=2024.8.0), Joblib (>=1.4.0).
- **Storage**: SQLAlchemy (>=2.0.0), h5py (>=3.11.0).
- **Geospatial**: GeoPandas (>=1.0.0), Rasterio (>=1.3.10), PyProj (>=3.6.0). **Note**: GDAL/GEOS are system-level; install via Conda if pip fails.
- **Optional (Geophysics)**: SimPEG (>=0.21.0), ObsPy (>=1.4.1), MintPy (>=1.6.0), PyGIMLi (>=1.5.0) – install with `pip install .[geophysics]`.
- **Optional (Visualization)**: PyGMT (>=0.12.0), PyVista (>=0.44.0), Folium (>=0.17.0), VTK (>=9.3.0) – install with `pip install .[visualization]`.
- **Development**: pytest (>=8.3.0), Black (>=24.4.2), etc. – install with `pip install -e .[dev]`.

For full list and versions, see [requirements.txt](../requirements.txt).

## Installation Methods

### Method 1: Quick Install via PyPI (End-Users)

For running GAM without modifying code. Installs core dependencies only.

1. **Install Python and pip** (if not already installed): Download from [python.org](https://www.python.org/downloads/).

2. **Create and activate a virtual environment**:
   ```bash
   # Using venv (built-in)
   python -m venv gam_env
   # Activate:
   # On Linux/macOS:
   source gam_env/bin/activate
   # On Windows:
   gam_env\Scripts\activate
   ```

   Alternatively, use Conda for better geospatial support:
   ```bash
   conda create -n gam python=3.12
   conda activate gam
   ```

3. **Install GAM**:
   ```bash
   pip install geoanomalymapper
   ```

   This installs the core package. For optional features:
   ```bash
   pip install geoanomalymapper[geophysics,visualization]
   ```

4. **Verify Installation**:
   ```bash
   gam --version
   python -c "import gam; print(gam.__version__)"
   ```

   Expected output: Version number (e.g., 1.0.0) and no import errors.

### Method 2: Install from Source (Developers/Contributors)

For editing code, testing, or extending GAM.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourorg/GeoAnomalyMapper.git
   cd GeoAnomalyMapper
   ```

2. **Set Up Environment** (Conda recommended):
   ```bash
   # Install Miniconda if needed: https://docs.conda.io/en/latest/miniconda.html
   conda create -n gam-dev python=3.12
   conda activate gam-dev
   ```

3. **Install System Dependencies** (if needed for geospatial):
   On Ubuntu/Debian:
   ```bash
   sudo apt update
   sudo apt install gdal-bin libgdal-dev libgeos-dev proj-bin
   ```
   On macOS (with Homebrew):
   ```bash
   brew install gdal geos proj
   ```

4. **Install GAM in Editable Mode**:
   ```bash
   # Core + dev dependencies
   pip install -e .[dev]
   ```

   This enables `import gam` and CLI access while allowing code changes.

5. **Verify**:
   Run the verification commands from Method 1. Additionally, run tests:
   ```bash
   pytest tests/ -v
   ```

### Method 3: Conda-Based Install for Researchers (Full Geospatial Stack)

For environments with heavy geophysical processing.

1. **Install Miniconda/Anaconda** if not present.

2. **Create Environment with Key Geospatial Deps**:
   ```bash
   conda create -n gam-research -c conda-forge python=3.12 gdal geos proj obspy simpeg pygimli
   conda activate gam-research
   ```

3. **Install GAM**:
   ```bash
   pip install geoanomalymapper[geophysics,visualization]
   # Or from source for latest:
   git clone https://github.com/yourorg/GeoAnomalyMapper.git
   cd GeoAnomalyMapper
   pip install -e .[geophysics,visualization,dev]
   ```

4. **Verify** as above, plus test a sample import:
   ```python
   python -c "from gam.modeling.gravity import GravityInverter; print('Success')"
   ```

## Environment Setup and Configuration

After installation:

1. **Download Sample Data** (optional, for testing):
   GAM fetches data on-the-fly, but for offline testing, download samples from [tests/data/](../tests/data/).

2. **Configure Data Sources**:
   Edit [data_sources.yaml](../data_sources.yaml) for API keys if required (most public sources are keyless).
   Example:
   ```yaml
   sources:
     gravity:
       base_url: "https://mrdata.usgs.gov/services/gravity"
       auth: null
   ```

3. **Set Environment Variables** (optional):
   - `GAM_CACHE_DIR`: Override default cache location (e.g., `export GAM_CACHE_DIR=/path/to/cache`).
   - `DASK_SCHEDULER`: For custom Dask clusters (e.g., `'distributed'` for cloud).

### Environment variables (.env)

GAM supports loading credentials from a `.env` file for sensitive API keys. The CLI and ingestion modules auto-load this file using python-dotenv.

1. Copy the example file:
   ```bash
   cp GeoAnomalyMapper/.env.example GeoAnomalyMapper/.env
   ```

2. Fill in your credentials in `GeoAnomalyMapper/.env`:
   - `ESA_USERNAME`: Your ESA username
   - `ESA_PASSWORD`: Your ESA password
   - `EARTHDATA_USER`: Your Earthdata username
   - `EARTHDATA_PASS`: Your Earthdata password

The `.env` file is automatically ignored by Git (see `.gitignore`).

4. **Update PATH** (if CLI not found):
   Ensure your virtual environment's bin/Scripts is in PATH.

## Troubleshooting Common Installation Issues

### Dependency Conflicts (e.g., GDAL)
- **Symptom**: `ImportError: No module named 'osgeo'` or build failures.
- **Solution**: Use Conda for GDAL: `conda install -c conda-forge gdal`. Avoid mixing pip/conda for geospatial libs.

### Version Mismatches
- **Symptom**: `ModuleNotFoundError` for NumPy/SciPy.
- **Solution**: Check Python version (`python --version`) and reinstall: `pip install --upgrade pip setuptools wheel`.

### Windows-Specific Issues
- **Symptom**: GDAL not found or slow performance.
- **Solution**: Install WSL2 (Ubuntu), then follow Linux instructions. For native: Use OSGeo4W installer for GDAL binaries.

### Slow Installation or Network Errors
- **Symptom**: Pip hangs on large packages like Dask.
- **Solution**: Use `--no-cache-dir` flag: `pip install --no-cache-dir geoanomalymapper`. Or use a mirror: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ ...`.

### Optional Dependencies Not Installing
- **Symptom**: Missing SimPEG or PyVista.
- **Solution**: Install extras explicitly: `pip install .[geophysics]`. For VTK/PyVista, ensure system libs (e.g., `apt install libvtk9-dev`).

### Verification Fails
- **Symptom**: `gam --version` not found.
- **Solution**: Reinstall in editable mode (`pip install -e .`) and restart terminal. Check `which gam` (Linux/macOS) or `where gam` (Windows).

If issues persist, check the [FAQ](../troubleshooting/faq.md) or open an issue on GitHub with your OS, Python version, and error logs.

## Next Steps

Once installed, proceed to the [Quickstart Guide](../quickstart.md) for your first analysis. For advanced configuration, see [Configuration Reference](../configuration/config_reference.md).

---

*Last Updated: 2025-09-23 | Compatible with GAM v1.0.0*