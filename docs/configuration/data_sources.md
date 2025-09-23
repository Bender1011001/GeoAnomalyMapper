# Data Source Configuration Guide

## Overview

GAM ingests data from public geophysical APIs. This guide covers supported providers, authentication, and custom integration. Sources are configured in [data_sources.yaml](../data_sources.yaml), loaded automatically.

**Key Concepts**:
- **Client Types**: "requests" (HTTP), "obspy" (FDSN), "sentinelsat" (ESA API).
- **Parameters**: bbox, time range, etc., passed from main config.
- **Auth**: Env vars for keys (e.g., ${API_KEY}); public sources need none.
- **Rate Limiting**: Respect `core.rate_limit_delay` to avoid bans.

For custom sources, see [Plugin System](../developer/architecture.md#plugin-system).

## Supported Data Providers and APIs

### Gravity (USGS)
- **Description**: Global gravity anomaly data (mGal).
- **API**: USGS ScienceBase/MR Data (GeoJSON).
- **Base URL**: `https://mrdata.usgs.gov/services/gravity?bbox={bbox}&format=geojson`
- **Client Type**: "requests"
- **Default Params**: {}
- **Auth**: None (public).
- **Coverage**: Worldwide, resolution ~5-10km.
- **Example Query**: bbox=29.9,30.0,31.1,31.2 returns point data.
- **Notes**: Filtered for land stations; offshore limited.

### Magnetic (USGS)
- **Description**: Magnetic anomaly data (nT).
- **API**: USGS MR Data (GeoJSON).
- **Base URL**: `https://mrdata.usgs.gov/services/magnetic?bbox={bbox}&format=geojson`
- **Client Type**: "requests"
- **Default Params**: {}
- **Auth**: None.
- **Coverage**: Global aeromagnetic surveys.
- **Notes**: Similar to gravity; combine for density/magnetization models.

### Seismic (IRIS FDSN)
- **Description**: Earthquake waveforms and station data.
- **API**: FDSN Web Services (MiniSEED).
- **Client Type**: "obspy" (obspy.clients.fdsn.Client)
- **Server**: "IRIS" (or "USGS", "GFZ").
- **Default Params**:
  - network: "*"
  - station: "*"
  - location: "*"
  - channel: "BH?" (broadband)
  - starttime: "2020-01-01"
  - endtime: "2025-09-22"
- **Auth**: None (public, but usage policy: <1000 events/day).
- **Coverage**: Global stations; velocity models for tomography.
- **Example**: Client.get_waveforms(network="*", station="*", ... ) → obspy.Stream.
- **Notes**: For velocity anomalies; process with bandpass filter.

### InSAR (Copernicus Sentinel-1)
- **Description**: Surface displacement from SAR interferometry.
- **API**: Copernicus Open Access Hub (DHuS).
- **Base URL**: `https://scihub.copernicus.eu/dhus/search?q=`
- **Client Type**: "sentinelsat" (SentinelAPI)
- **Default Params**:
  - platformname: "Sentinel-1"
  - producttype: "SLC" (or "GRD")
  - date: ["20200101", "20250922"]
- **Auth**: Required (free registration).
  - username: ${COPERNICUS_USERNAME}
  - password: ${COPERNICUS_PASSWORD}
- **Coverage**: Global since 2014, 5-20m resolution.
- **Example**: api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus'); api.query(bbox, date=..., producttype='SLC').
- **Notes**: Download ZIPs (~1GB); process with MintPy for interferograms. Rate limit: 10 downloads/min.

## Authentication Setup

1. **Public Sources** (Gravity, Magnetic, Seismic): No keys needed. Set in data_sources.yaml: `auth: null`.

2. **InSAR (Copernicus)**:
   - Register at [scihub.copernicus.eu](https://scihub.copernicus.eu/).
   - Set env vars:
     ```bash
     export COPERNICUS_USERNAME=your_email
     export COPERNICUS_PASSWORD=your_password
     ```
   - In YAML: `auth: {username: "${COPERNICUS_USERNAME}", password: "${COPERNICUS_PASSWORD}"}`
   - Test: `python -c "from sentinelsat import SentinelAPI; api = SentinelAPI(os.getenv('COPERNICUS_USERNAME'), os.getenv('COPERNICUS_PASSWORD'))"`

3. **Custom/Private APIs**: Add `auth: {api_key: "${MY_API_KEY}"}`. Use python-dotenv for .env files (gitignore'd).

**Security**: Never commit keys; use .env. For production, use secrets managers (AWS Secrets, etc.).

## Custom Data Source Integration

To add new sources (e.g., LiDAR):

1. **Edit data_sources.yaml**:
   ```yaml
   lidar:
     base_url: "https://example.com/api/lidar?bbox={bbox}"
     client_type: "requests"
     default_params: {resolution: "high"}
     auth: null
   ```

2. **Implement Fetcher** (if new client_type):
   - Subclass DataSource in `gam/ingestion/fetchers.py`.
   - Example:
     ```python
     class LidarSource(DataSource):
         def fetch(self, bbox, params):
             # Custom logic
             return RawData(metadata={}, values=fetch_lidar(bbox))
     ```
   - Register in __init__.py.

3. **Plugin Package**: For external, use entry_points (see Developer Guide).

4. **Test**: Run `gam run --modalities lidar --bbox ...` or API `fetch_data(bbox, ["lidar"])`.

**Data Formats**: New sources must return RawData (values