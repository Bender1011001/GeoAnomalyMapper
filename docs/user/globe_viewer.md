# 3D Globe Viewer

## Overview

The 3D Globe Viewer provides a global, Google Earth-style visualization for exploring anomalies and data layers in an interactive 3D environment. It renders heatmap overlays, point entities, and optional terrain using CesiumJS, allowing users to navigate geospatial results from GeoAnomalyMapper analyses.

This page is accessible via the Streamlit dashboard entrypoint [`dashboard/app.py`](../../../dashboard/app.py) and powered by the visualization engine [`gam/visualization/globe_viewer.py`](../../../gam/visualization/globe_viewer.py).

For installation and setup, see the [Installation Guide](installation.md) and [Quickstart](quickstart.md).

## Requirements

The 3D Globe requires a Cesium Ion access token for full functionality, including terrain rendering. Set `CESIUM_TOKEN` via environment variable (preferred) or Streamlit secrets (optional). Without the token, the globe renders with ellipsoid (non-terrain) mode, and imagery layers still load.

To obtain a token, sign up at [cesium.com/ion](https://cesium.com/ion/) and generate an access token.

Set the token with these commands (replace `your_token_here`):

**Linux/macOS (bash/zsh):**
```
export CESIUM_TOKEN="your_token_here"
```

**Windows PowerShell:**
```
$env:CESIUM_TOKEN = "your_token_here"
```

**Windows CMD:**
```
set CESIUM_TOKEN=your_token_here
```

For Streamlit secrets, add to `.streamlit/secrets.toml`:
```
CESIUM_TOKEN = "your_token_here"
```

## Access

1. Start the dashboard:
   ```
   streamlit run GeoAnomalyMapper/dashboard/app.py
   ```
2. Open http://localhost:8501 in your browser.
3. Navigate to the "3D Globe" page in the sidebar.

## Page Controls

### Data Source

Use the "Data Source" radio buttons in the sidebar to select visualization mode:

- **Demo**: Loads a built-in example scene (random anomalies near Northern California) to showcase 3D navigation, heatmap overlays, and entity styles. No analysis required.
- **Load Analysis**: Enter an "Analysis ID" (default: "latest") in the text input to load results from a previous run. If the loader is unavailable or fails, it falls back to Demo mode.

### Layer Toggles and Styling

- **Heatmap Opacity**: Adjust the slider (0.0 to 1.0, default 0.7) to control the transparency of the anomaly heatmap overlay (uses 'hot' colormap by default).
- **Show high anomalies as cylinders**: Toggle the checkbox (default: on) to display the top 1% of anomalies (99th percentile threshold) as 3D cylinders. Heights scale with anomaly values; up to 150 points for performance.

No additional sliders for time, thresholds, or color ramps are present.

### Export/Download

- **Download scene.json**: Click the button to download the current scene configuration (camera position, layers, opacity, entities) as a JSON file to your browser's default download location. This file can be shared or reloaded in future sessions if supported.
- **Save scene.json**: Enter a "Save as Analysis ID" (default: "demo") and click the button to persist the scene to the server at `data/outputs/state/{id}/scene.json`. Useful for saving Demo configurations or analysis views.

Note: Scenes are saved to data/outputs/state/&lt;analysis_id&gt;/scene.json (and scene.html); the current 3D Globe page does not yet reload persisted scenes—open scene.html directly or fetch scene.json via the API.
## Camera & UX

The camera initializes centered on the data's bounding box at 150 km altitude. Use mouse controls for navigation:
- Left-click and drag: Rotate the view.
- Right-click and drag: Pan.
- Scroll wheel: Zoom in/out.
- Touch gestures supported on mobile.

Terrain elevation is enabled only with a valid `CESIUM_TOKEN`; otherwise, the globe uses a smooth ellipsoid without surface details. Imagery (e.g., satellite basemaps) loads regardless.

## Tips

- **Performance**: For large datasets with many entities, zoom in to reduce rendered points and improve responsiveness. Prefer 3D Tiles for heavy volumetric data (see Developer Guide).
- **CRS Note**: Visualizations use ECEF (EPSG:4978) coordinates. 3D Tiles, if used, are served via the API under the "/tiles" endpoint (e.g., "/tiles/myset/tileset.json").
### Minimal KML overlay export

To export a basic KML GroundOverlay for Google Earth, use the `export_anomaly_kml` function from the visualization module.

BBox is (west, south, east, north) in degrees (WGS84/EPSG:4326).

```python
import numpy as np
from GeoAnomalyMapper.gam.visualization.kml_export import export_anomaly_kml

anomalies = np.array([[0.0, 1.0],
                      [np.nan, 0.5]], dtype=float)
bbox = (-1.0, -1.0, 1.0, 1.0)  # (west, south, east, north) in degrees
out_kml = "example_overlay.kml"
export_anomaly_kml(anomalies, bbox, out_kml)
```