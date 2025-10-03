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

## Globe and API Quick Hits

### Cesium terrain not loading (globe renders flat/ellipsoid)
- **Symptoms**
  - Globe loads but appears flat; terrain tiles fail to load; console may show 401/403 from Cesium Ion.
- **Cause**
  - Missing Cesium Ion token or invalid token.
- **Fix**
  - Prefer environment variable CESIUM_TOKEN (env-first). Optional Streamlit secrets fallback is supported.
  - Set token (pick your OS):
    - Linux/macOS (bash/zsh):
      ```
      export CESIUM_TOKEN="your_token_here"
      ```
    - Windows PowerShell:
      ```
      $Env:CESIUM_TOKEN = "your_token_here"
      ```
    - Windows CMD:
      ```
      set CESIUM_TOKEN=your_token_here
      ```
  - Optional secrets file (project-local or user-level):
    - Use example: [secrets.example.toml](GeoAnomalyMapper/dashboard/.streamlit/secrets.example.toml)
- **Notes**
  - Without CESIUM_TOKEN, the globe still works, but terrain is disabled. See [globe_viewer.md](GeoAnomalyMapper/docs/user/globe_viewer.md).

### StreamlitSecretNotFoundError on startup
- **Symptoms**
  - Error previously raised when secrets were missing.
- **Cause**
  - Earlier versions prioritized Streamlit secrets; the current implementation prefers environment variable first.
- **Fix**
  - Set CESIUM_TOKEN via environment variable as shown above.
  - If using secrets, ensure TOML structure matches the example in [secrets.example.toml](GeoAnomalyMapper/dashboard/.streamlit/secrets.example.toml).
- **References**
  - UI page: [3_3D_Globe.py](GeoAnomalyMapper/dashboard/pages/3_3D_Globe.py)
  - User setup: [installation.md](GeoAnomalyMapper/docs/user/installation.md), [quickstart.md](GeoAnomalyMapper/docs/user/quickstart.md)

### Heatmap looks misaligned or rotated relative to region of interest
- **Symptoms**
  - Heatmap layer appears shifted, rotated, or mirrored versus expected bounding area.
- **Likely Causes**
  - Bbox order confusion (lon/lat swapped) or different CRS assumptions.
- **Fix**
  - Ensure bbox order is [minLon, minLat, maxLon, maxLat] (degrees, WGS84).
  - Verify data CRS is geographic (EPSG:4326). Reproject before loading if necessary.
  - If providing custom ranges in the UI, confirm inputs match the expected lon/lat order displayed in [3_3D_Globe.py](GeoAnomalyMapper/dashboard/pages/3_3D_Globe.py).
- **References**
  - Globe page: [3_3D_Globe.py](GeoAnomalyMapper/dashboard/pages/3_3D_Globe.py)

### 3D Tiles appear off-earth, at wrong scale, or invisible
- **Symptoms**
  - Tiles render far from the earth, at the core, or not visible.
- **Causes**
  - Incorrect CRS conversion to ECEF (EPSG:4978), swapped lon/lat, radians vs. degrees, or altitude units mismatch.
- **Fix**
  - Input coordinates must be lon/lat degrees (EPSG:4326) with altitude in meters (above WGS84 ellipsoid).
  - Convert to ECEF (EPSG:4978) with pyproj before writing tiles. See pipeline details in [architecture.md](GeoAnomalyMapper/docs/architecture.md) and implementation in [tiles_builder.py](GeoAnomalyMapper/gam/visualization/tiles_builder.py).
  - Sanity check a sample point conversion with pyproj locally to confirm expected XYZ magnitude.
- **References**
  - Tiles pipeline: [architecture.md](GeoAnomalyMapper/docs/architecture.md)
  - Builder: [tiles_builder.py](GeoAnomalyMapper/gam/visualization/tiles_builder.py)

### Dashboard becomes sluggish or crashes with many entities
- **Symptoms**
  - UI freezes, memory spikes, slow interactions when plotting large point clouds or many cylinders.
- **Causes**
  - Rendering many individual entities in Cesium is expensive.
- **Fix**
  - Prefer 3D Tiles for large datasets; build tilesets and load via URL pattern "/tiles/&lt;myset&gt;/tileset.json".
  - Reduce entity counts: turn off "Show high anomalies as cylinders" and use heatmap visualization; limit points or downsample.
  - Split data by region/time if applicable.
- **References**
  - Viewer: [globe_viewer.py](GeoAnomalyMapper/gam/visualization/globe_viewer.py)
  - UI controls: [3_3D_Globe.py](GeoAnomalyMapper/dashboard/pages/3_3D_Globe.py)
  - Tiles docs: [architecture.md](GeoAnomalyMapper/docs/architecture.md)

### 404 for tiles or scene artifacts (local or behind proxy)
- **Symptoms**
  - 404 GET /tiles/... or 404 GET /analysis/&lt;id&gt;/scene.json
- **Causes**
  - Tileset directory missing/not mounted; wrong tileset name; analysis artifacts absent; reverse proxy path mismatch.
- **Fix**
  - Verify tiles exist under data/outputs/tilesets/&lt;myset&gt;/tileset.json.
  - Confirm API serves tiles and scenes per [main.py](GeoAnomalyMapper/gam/api/main.py) and docs in [api_reference.md](GeoAnomalyMapper/docs/developer/api_reference.md).
  - For Docker/proxy, ensure routes in [nginx.conf](GeoAnomalyMapper/deployment/docker/nginx.conf) match expected:
    - / → dashboard, /analysis → API, /tiles → API
  - Compose stack and env config: [docker-compose.yml](GeoAnomalyMapper/deployment/docker/docker-compose.yml), [deployment.md](GeoAnomalyMapper/docs/configuration/deployment.md)

### 3D Tiles Generation Issues
- **Symptoms**
  - 404s when loading tiles in the viewer or missing tileset.json.
- **Likely Causes and Checks**
  - Wrong output directory or server not mounting the tiles path used by your app/API; verify the tiles_out path and server/static mount.
  - CRS mismatch: ensure inputs to the globe/3D pipeline use ECEF coordinates when required; verify using [`reproject_wgs84_to_ecef()`](GeoAnomalyMapper/gam/visualization/tiles_builder.py:160).
  - py3dtiles not on PATH; wrapper will raise FileNotFoundError. Install py3dtiles and ensure it’s available.
- **References**
  - Tiles utilities: [API Reference - Tiles Utilities](GeoAnomalyMapper/docs/developer/api_reference.md)
### Quick diagnostics checklist
- CESIUM_TOKEN set? If not, terrain disabled by design. See [installation.md](GeoAnomalyMapper/docs/user/installation.md).
- Tileset URL correct? Test in a browser:
  - http://localhost:8000/tiles/myset/tileset.json (direct)
  - http://localhost/tiles/myset/tileset.json (via Docker proxy)
- Scene available?
  - http://localhost:8000/analysis/&lt;analysis_id&gt;/scene.json
- CRS verified?
  - Inputs in EPSG:4326; tiles in ECEF EPSG:4978. See [architecture.md](GeoAnomalyMapper/docs/architecture.md).
If your issue isn't listed, provide logs/OS/Python version in [GitHub Issues](https://github.com/yourorg/GeoAnomalyMapper/issues).

---

*Last Updated: 2025-09-23 | GAM v1.0.0*