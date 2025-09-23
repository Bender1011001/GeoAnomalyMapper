# Common Problems and Solutions

This document covers frequent issues in GeoAnomalyMapper (GAM), grouped by category. Always check console/logs (`gam.log`) for details. For unresolved problems, see [FAQ](faq.md) or [Support](support.md).

## Installation Issues and Dependency Conflicts

### GDAL/GEOS Not Found or Build Errors
- **Symptoms**: `ImportError: No module named 'osgeo._gdal'` or compilation failures during pip install.
- **Causes**: Geospatial libs (GDAL, GEOS) require system binaries.
- **Solutions**:
  - Use Conda: `conda install -c conda-forge gdal geos proj`.
  - On Ubuntu: `sudo apt install gdal-bin libgdal-dev libgeos-dev libproj-dev`.
  - On macOS: `brew install gdal geos proj`.
  - Windows: Use OSGeo4W or WSL with Ubuntu.
  - Verify: `python -c "from osgeo import gdal; print(gdal.__version__)"`.
- **Prevention**: Install geospatial deps via Conda before pip GAM.

### Version Mismatches (e.g., NumPy/SciPy Conflicts)
- **Symptoms**: `AttributeError` or incompatible array shapes.
- **Causes**: Mixed Python environments or outdated deps.
- **Solutions**:
  - Fresh env: `conda create -n gam python=3.12; conda activate gam; pip install -r requirements.txt`.
  - Upgrade: `pip install --upgrade numpy scipy pandas xarray`.
  - Check versions: `pip list | grep numpy` (match [requirements.txt](../requirements.txt)).
- **Tip**: Use `pip check` to detect conflicts.

### Optional Dependencies Missing (e.g., SimPEG, ObsPy)
- **Symptoms**: `ModuleNotFoundError: No module named 'simpeg'` during modeling.
- **Causes**: Extras not installed.
- **Solutions**:
  - Install extras: `pip install geoanomalymapper[geophysics,visualization]`.
  - Or from source: `pip install -e .[all]`.
  - For ObsPy: `conda install -c conda-forge obspy`.
- **Verify**: `python -c "import simpeg; print('OK')"` for each.

## Data Access and API Authentication Problems

### API Fetch Failures (e.g., USGS Timeout)
- **Symptoms**: `DataFetchError: Fetch failed for USGS: Timeout`.
- **Causes**: Network issues, rate limits, or invalid bbox.
- **Solutions**:
  - Increase delay: `core.rate_limit_delay: 5.0` in config.
  - Check bbox: Ensure min < max, lat [-90,90], lon [-180,180].
  - Retry: Run with `--verbose` for details; use `@retry` in code.
  - Offline: Pre-download data, load with `fetch_data(..., use_cache_only=True)`.
- **Test**: `gam run --modalities gravity --bbox 0 1 0 1 --verbose`.

### Copernicus InSAR Authentication Error
- **Symptoms**: `AuthenticationError: Invalid username/password`.
- **Causes**: Missing or incorrect credentials.
- **Solutions**:
  - Register at [scihub.copernicus.eu](https://scihub.copernicus.eu/).
  - Set env: `export COPERNICUS_USERNAME=your_user; export COPERNICUS_PASSWORD=your_pass`.
  - Verify: `python -c "from sentinelsat import SentinelAPI; api = SentinelAPI(os.getenv('COPERNICUS_USERNAME'), os.getenv('COPERNICUS_PASSWORD'))"`.
  - Alternative: Use ASDAS (Alaska Satellite Facility) for open access.
- **Note**: Free, but quota-limited (500GB/month).

### No Data Returned for Bbox/Modality
- **Symptoms**: Empty RawData or "No stations found".
- **Causes**: Sparse coverage (e.g., seismic in oceans) or time range.
- **Solutions**:
  - Expand bbox/time: Seismic `starttime: "2010-01-01"`.
  - Check sources: Edit [data_sources.yaml](../data_sources.yaml) for params.
  - Fallback: Use synthetic data from `tests/data/` for testing.
- **Debug**: Add `logging_level: "DEBUG"` to see API responses.

## Performance and Memory Issues

### Out of Memory (OOM) During Global Run
- **Symptoms**: `MemoryError` or Dask worker crash.
- **Causes**: Large grids or many tiles.
- **Solutions**:
  - Reduce grid_res: 1.0 for global (vs 0.01 local).
  - Increase chunks: In Dask Client, `chunks=(500, 500)` for arrays.
  - Spill to disk: Set `temporary_directory: "/tmp/dask"` in Client.
  - Scale horizontally: Use cloud cluster (see [Deployment](../configuration/deployment.md)).
  - Monitor: Dask dashboard shows memory usage.
- **Tip**: Run `htop` or `nvidia-smi` (GPU) during execution.

### Slow Processing or Stuck Pipeline
- **Symptoms**: Run hangs on ingestion/modeling.
- **Causes**: API delays or heavy inversions.
- **Solutions**:
  - Parallelize: `parallel_workers: 8`.
  - Coarse mesh: `mesh_start_coarse: 20.0`.
  - Skip modalities: Test with "gravity" only.
  - Profile: Use `cProfile` or `--verbose` logs to identify bottleneck.
- **Benchmark**: Use `tests/test_performance.py` for timing.

### Dask Workers Failing or Dashboard Not Loading
- **Symptoms**: `Worker failed` or dashboard inaccessible.
- **Causes**: Resource limits or network.
- **Solutions**:
  - Restart Client: `client.restart()`.
  - Limit workers: `n_workers=2, memory_limit='2GB'`.
  - Local only: `scheduler='threads'` for simple runs.
  - Firewall: Open port 8787 for dashboard.
- **Debug**: `client.get_logs()` for errors.

## Visualization and Export Problems

### No Output Files or Empty Maps
- **Symptoms**: Directory empty or blank PNG.
- **Causes**: No anomalies detected or export failure.
- **Solutions**:
  - Lower threshold: `threshold: 1.5`.
  - Check confidence_min: Set to 0.0.
  - Verify formats: Install PyGMT/PyVista for "3d".
  - Test: `generate_visualization(pd.DataFrame())` with sample data.
- **Tip**: Run with "png" format first.

### VTK/GeoTIFF Corrupted or Unreadable
- **Symptoms**: Files not opening in ParaView/QGIS.
- **Causes**: Missing CRS or large size.
- **Solutions**:
  - Add CRS: Ensure xarray attrs {'crs': 'EPSG:4326'}.
  - Reduce size: Lower grid_res or crop bbox.
  - Validate: `gdalinfo anomaly_map.geotiff`.
  - Alternative: Export CSV for manual import.
- **Libraries**: `conda install -c conda-forge gdal vtk`.

## Configuration and Usage Errors

### Invalid Config or Validation Error
- **Symptoms**: `ValidationError: bbox must have 4 elements`.
- **Causes**: YAML syntax or type mismatch.
- **Solutions**:
  - Validate: `gam validate-config config.yaml`.
  - Check types: bbox as list[float], not str.
  - Defaults: Use [config.yaml](../config.yaml) as template.
- **Editor**: Use VS Code with YAML extension for linting.

### CLI Command Not Found
- **Symptoms**: `gam: command not found`.
- **Causes**: Not in PATH or not installed editable.
- **Solutions**:
  - Reinstall: `pip install -e .`.
  - Activate env: `conda activate gam`.
  - Alias: Add `alias gam='python -m gam.core.cli'` to .bashrc.
- **Verify**: `which gam` (Linux/macOS).

### No Anomalies Detected Despite Data
- **Symptoms**: Empty CSV, "0 anomalies".
- **Causes**: High threshold or poor data quality.
- **Solutions**:
  - Tune threshold: Start at 1.0.
  - Check data: `fetch_data(bbox, modalities)` and inspect raw_data.values.
  - Fusion weight: Increase joint_weight to 0.8.
  - Synthetic test: Use `tests/data/synthetic_gravity.json`.
- **Debug**: Set logging "DEBUG" for detection steps.

If your issue isn't listed, provide logs/OS/Python version in [GitHub Issues](https://github.com/yourorg/GeoAnomalyMapper/issues).

---

*Last Updated: 2025-09-23 | GAM v1.0.0*