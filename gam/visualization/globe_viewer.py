from __future__ import annotations

"""
CesiumJS Globe Viewer utilities for Streamlit embedding.

This module provides a lightweight Python-side scene builder and an HTML renderer
for a CesiumJS-based globe. It is designed for use within the GAM Dashboard
(Streamlit) via streamlit.components.v1.html.

Usage
-----
Example:
    from gam.visualization.globe_viewer import GlobeViewer
    import numpy as np

    viewer = GlobeViewer(cesium_token="YOUR_CESIUM_ION_TOKEN")

    # Add a heatmap from a 2D anomaly array over a geodetic bbox (w, s, e, n)
    anomalies = np.random.randn(256, 256)
    viewer.add_anomaly_heatmap(anomalies, bbox=(-125.0, 32.0, -113.0, 42.0),
                               cmap="hot", opacity=0.7, name="Anomalies")

    # Add point entities (cylinders) scaled by value (meters = value * 1000)
    points = [
        {"lat": 37.7749, "lon": -122.4194, "value": 2.0, "label": "SF"},
        {"lat": 34.0522, "lon": -118.2437, "value": 1.2, "label": "LA"},
    ]
    viewer.add_point_entities(points, shape="cylinder", name="Points")

    # Optional 3D Tileset
    viewer.add_3d_tiles("https://assets.cesium.com/1461/tileset.json", opacity=0.6, name="Subsurface")

    # Camera preset
    viewer.set_camera(lon=-120.0, lat=36.0, height_m=2_000_000.0, heading_deg=0.0, pitch_deg=-30.0)

    # Render HTML string (suitable for Streamlit components.html)
    html = viewer.render_streamlit_html()

Notes
-----
- The heatmap is encoded as a single transparent PNG and added as a Cesium SingleTileImageryProvider
  over the provided bounding box in degrees.
- Entities are currently rendered as cylinders (vertical), with length scaled from the input value.
- 3D Tiles are styled with a global opacity using Cesium3DTileStyle.
"""

import base64
import json
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class GlobeViewer:
    """
    CesiumJS Globe viewer scene builder and HTML renderer.

    Parameters
    ----------
    cesium_token : Optional[str]
        Cesium Ion access token. If provided, it will be set in the rendered HTML.
    """

    def __init__(self, cesium_token: Optional[str]) -> None:
        self.cesium_token: str = cesium_token or ""
        self.layers: List[Dict] = []
        self.camera: Optional[Dict[str, float]] = None

    def add_anomaly_heatmap(
        self,
        anomalies: np.ndarray,
        bbox: Tuple[float, float, float, float],
        cmap: str = "hot",
        opacity: float = 0.7,
        name: str = "Anomalies",
    ) -> None:
        """
        Add a single-tile heatmap layer from a 2D anomaly array.

        Parameters
        ----------
        anomalies : np.ndarray
            2D array of anomaly values; NaNs will be transparent.
        bbox : Tuple[float, float, float, float]
            Geographic bounding box in degrees as (west, south, east, north).
        cmap : str, default "hot"
            Matplotlib colormap name.
        opacity : float, default 0.7
            Layer opacity on the globe.
        name : str, default "Anomalies"
            Layer display name.
        """
        # Ensure array is float and handle empty arrays gracefully
        arr = np.array(anomalies, dtype=np.float64)
        if arr.size == 0:
            # Create a 1x1 fully transparent pixel
            rgba = np.zeros((1, 1, 4), dtype=np.float64)
        else:
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                norm = np.zeros_like(arr, dtype=np.float64)
            else:
                norm = (arr - vmin) / (vmax - vmin)
            rgba = cm.get_cmap(cmap)(norm)
            # Transparent where anomalies are not finite
            mask = ~np.isfinite(arr)
            if np.any(mask):
                rgba = np.array(rgba, copy=True)
                rgba[mask, 3] = 0.0

        # Encode RGBA array as a PNG in-memory
        buf = BytesIO()
        plt.imsave(buf, rgba, format="png")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        layer = {
            "kind": "singleTile",
            "type": "singleTile",
            "name": name,
            "imageBase64": img_b64,
            "bbox": {"w": float(bbox[0]), "s": float(bbox[1]), "e": float(bbox[2]), "n": float(bbox[3])},
            "opacity": float(opacity),
        }
        self.layers.append(layer)

    def add_point_entities(
        self,
        points: List[Dict],
        shape: str = "cylinder",
        name: str = "Points",
    ) -> None:
        """
        Add point entities as cylinders.

        Parameters
        ----------
        points : List[Dict]
            List of dicts with required keys 'lat', 'lon' and optional 'value', 'label'.
            Height (meters) is computed as max(value, 0) * 1000.0.
        shape : str, default "cylinder"
            Shape indicator stored with each entity (currently cylinders are rendered).
        name : str, default "Points"
            Layer display name.
        """
        entities: List[Dict] = []
        for p in points:
            lat = float(p.get("lat"))
            lon = float(p.get("lon"))
            raw_val = p.get("value", 0.0)
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                val = 0.0
            height_m = max(val, 0.0) * 1000.0
            label = p.get("label")
            entities.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "height": height_m,
                    "label": label,
                    "shape": shape,
                }
            )

        layer = {
            "kind": "entities",
            "type": "entities",
            "name": name,
            "entities": entities,
        }
        self.layers.append(layer)

    def add_3d_tiles(self, tiles_url: str, opacity: float = 0.6, name: str = "Subsurface") -> None:
        """
        Add a 3D Tiles layer.

        Parameters
        ----------
        tiles_url : str
            URL to the Cesium 3D Tiles tileset.json.
        opacity : float, default 0.6
            Global opacity for the tileset via style.
        name : str, default "Subsurface"
            Layer display name.
        """
        layer = {
            "kind": "3dtiles",
            "type": "3dtiles",
            "name": name,
            "url": tiles_url,
            "opacity": float(opacity),
        }
        self.layers.append(layer)

    def set_camera(
        self,
        lon: float,
        lat: float,
        height_m: float,
        heading_deg: float = 0.0,
        pitch_deg: float = -30.0,
    ) -> None:
        """
        Set the initial camera.

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.
        height_m : float
            Camera height in meters above the ellipsoid.
        heading_deg : float, default 0
            Heading in degrees (clockwise from north).
        pitch_deg : float, default -30
            Pitch in degrees (negative looks down).
        """
        self.camera = {
            "lon": float(lon),
            "lat": float(lat),
            "height": float(height_m),
            "heading": float(heading_deg),
            "pitch": float(pitch_deg),
        }

    def export_scene_config(self) -> str:
        """
        Export the current scene configuration as a JSON string.

        Returns
        -------
        str
            JSON string with keys "layers" and "camera".
        """
        return json.dumps({"layers": self.layers, "camera": self.camera or {}}, indent=2)

    def render_streamlit_html(self) -> str:
        """
        Render an embeddable HTML document string with CesiumJS.

        Returns
        -------
        str
            A complete HTML document string containing a Cesium.Viewer instance
            and JavaScript to add the configured layers and optional camera.
        """
        scene_json_str = self.export_scene_config()

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GAM Globe Viewer</title>
  <link href="https://cdn.jsdelivr.net/npm/cesium@1.117.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
  <style>
    html, body, #cesiumContainer { width: 100%; height: 100%; margin: 0; padding: 0; }
    /* Fixed viewer height suitable for Streamlit container */
    #cesiumContainer { height: 800px; }
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <script src="https://cdn.jsdelivr.net/npm/cesium@1.117.0/Build/Cesium/Cesium.js"></script>
  <script>
    const CESIUM_TOKEN = TOKEN_PLACEHOLDER;
    if (CESIUM_TOKEN) { Cesium.Ion.defaultAccessToken = CESIUM_TOKEN; }

    const viewer = new Cesium.Viewer('cesiumContainer', {
      terrain: Cesium.Terrain.fromWorldTerrain(),
      timeline: false,
      animation: false,
      sceneModePicker: true,
      baseLayerPicker: false
    });

    // Helper: add single-tile imagery (PNG over bbox)
    function addSingleTile(layer) {
      const provider = new Cesium.SingleTileImageryProvider({
        url: 'data:image/png;base64,' + layer.imageBase64,
        rectangle: Cesium.Rectangle.fromDegrees(layer.bbox.w, layer.bbox.s, layer.bbox.e, layer.bbox.n)
      });
      const imageryLayer = viewer.imageryLayers.addImageryProvider(provider);
      imageryLayer.alpha = (layer.opacity !== undefined) ? layer.opacity : 1.0;
    }

    // Helper: add entities as cylinders
    function addEntities(layer) {
      const items = layer.entities || [];
      for (const e of items) {
        const position = Cesium.Cartesian3.fromDegrees(e.lon, e.lat, 0.0);
        const height = (e.height !== undefined) ? e.height : 0.0;
        const length = Math.max(height, 10.0);
        const radius = Math.max(100.0, length * 0.1);

        viewer.entities.add({
          position: position,
          cylinder: {
            length: length,
            topRadius: radius,
            bottomRadius: radius,
            material: Cesium.Color.RED.withAlpha(0.6)
          },
          label: e.label ? {
            text: String(e.label),
            font: '14px sans-serif',
            fillColor: Cesium.Color.WHITE,
            pixelOffset: new Cesium.Cartesian2(0, -20)
          } : undefined
        });
      }
    }

    // Helper: add polylines (load-if-present)
    function addPolylines(viewer, items) {
      if (!items || items.length === 0) return;
      for (const item of items) {
        const pts = item.positions;
        if (!pts || pts.length === 0) continue;
        const hasHeights = Array.isArray(pts[0]) && pts[0].length >= 3;
        const flat = [];
        for (const p of pts) {
          const lon = Number(p[0]);
          const lat = Number(p[1]);
          if (hasHeights) {
            const h = Number(p[2]);
            flat.push(lon, lat, h);
          } else {
            flat.push(lon, lat);
          }
        }
        const positions = hasHeights
          ? Cesium.Cartesian3.fromDegreesArrayHeights(flat)
          : Cesium.Cartesian3.fromDegreesArray(flat);

        const width = (item.width !== undefined) ? Number(item.width) : 2;
        const c = item.color || {};
        const r = (c.r !== undefined) ? Number(c.r) : 255;
        const g = (c.g !== undefined) ? Number(c.g) : 255;
        const b = (c.b !== undefined) ? Number(c.b) : 0;
        const a = (c.a !== undefined) ? Number(c.a) : 0.8;
        const aByte = Math.max(0, Math.min(255, Math.round(a * 255)));

        viewer.entities.add({
          polyline: {
            positions: positions,
            width: width,
            material: Cesium.Color.fromBytes(
              Math.max(0, Math.min(255, r|0)),
              Math.max(0, Math.min(255, g|0)),
              Math.max(0, Math.min(255, b|0)),
              aByte
            )
          }
        });
      }
    }

    // Helper: add polygons (load-if-present; outer ring only)
    function addPolygons(viewer, items) {
      if (!items || items.length === 0) return;
      for (const item of items) {
        const ring = item.hierarchy || item.positions;
        if (!ring || ring.length === 0) continue;
        const hasHeights = Array.isArray(ring[0]) && ring[0].length >= 3;
        const flat = [];
        for (const p of ring) {
          const lon = Number(p[0]);
          const lat = Number(p[1]);
          if (hasHeights) {
            const h = Number(p[2]);
            flat.push(lon, lat, h);
          } else {
            flat.push(lon, lat);
          }
        }
        const positions = hasHeights
          ? Cesium.Cartesian3.fromDegreesArrayHeights(flat)
          : Cesium.Cartesian3.fromDegreesArray(flat);

        const c = item.color || {};
        const r = (c.r !== undefined) ? Number(c.r) : 0;
        const g = (c.g !== undefined) ? Number(c.g) : 255;
        const b = (c.b !== undefined) ? Number(c.b) : 255;
        const a = (c.a !== undefined) ? Number(c.a) : 0.4;
        const aByte = Math.max(0, Math.min(255, Math.round(a * 255)));
        const material = Cesium.Color.fromBytes(
          Math.max(0, Math.min(255, r|0)),
          Math.max(0, Math.min(255, g|0)),
          Math.max(0, Math.min(255, b|0)),
          aByte
        );

        const outline = (item.outline !== undefined) ? Boolean(item.outline) : true;
        const oc = item.outlineColor || {};
        const orr = (oc.r !== undefined) ? Number(oc.r) : 0;
        const org = (oc.g !== undefined) ? Number(oc.g) : 0;
        const orb = (oc.b !== undefined) ? Number(oc.b) : 0;
        const oaa = (oc.a !== undefined) ? Number(oc.a) : 0.8;
        const oaByte = Math.max(0, Math.min(255, Math.round(oaa * 255)));
        const outlineColor = Cesium.Color.fromBytes(
          Math.max(0, Math.min(255, orr|0)),
          Math.max(0, Math.min(255, org|0)),
          Math.max(0, Math.min(255, orb|0)),
          oaByte
        );

        viewer.entities.add({
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(positions),
            material: material,
            outline: outline,
            outlineColor: outlineColor
          }
        });
      }
    }

    // Helper: add 3D Tiles with opacity style
    async function add3DTiles(layer) {
      try {
        const tileset = await Cesium.Cesium3DTileset.fromUrl(layer.url);
        tileset.style = new Cesium.Cesium3DTileStyle({
          color: "color('white'," + ((layer.opacity !== undefined) ? layer.opacity : 1.0) + ")"
        });
        tileset.show = (layer.show !== undefined) ? Boolean(layer.show) : false;
        viewer.scene.primitives.add(tileset);
      } catch (err) {
        console.error('3D Tiles load failed', err);
      }
    }

    // Apply scene configuration
    const scene = SCENE_JSON_PLACEHOLDER;

    (async () => {
      const layers = scene.layers || [];
      for (const layer of layers) {
        const kind = layer.type || layer.kind;
        if (kind === 'singleTile') {
          addSingleTile(layer);
        } else if (kind === 'entities') {
          addEntities(layer);
        } else if (kind === 'polylines') {
          addPolylines(viewer, layer.items || layer.polylines || []);
        } else if (kind === 'polygons') {
          addPolygons(viewer, layer.items || layer.polygons || []);
        } else if (kind === '3dtiles') {
          await add3DTiles(layer);
        }
      }

      const cam = scene.camera || null;
      if (cam && Object.keys(cam).length > 0) {
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.height),
          orientation: {
            heading: Cesium.Math.toRadians(cam.heading || 0.0),
            pitch: Cesium.Math.toRadians(cam.pitch || -30.0),
            roll: 0.0
          },
          duration: 0.0
        });
      } else {
        // Fallback: use first singleTile layer's bbox center if available
        const st = (scene.layers || []).find(l => ((l.type || l.kind) === 'singleTile') && l.bbox);
        if (st) {
          const w = st.bbox.w, s = st.bbox.s, e = st.bbox.e, n = st.bbox.n;
          const lon = (w + e) / 2.0;
          const lat = (s + n) / 2.0;
          const height = Math.max(150000.0, 1000.0 * Math.max(Math.abs(e - w), Math.abs(n - s)) * 1000.0);
          const destination = Cesium.Cartesian3.fromDegrees(lon, lat, height);
          viewer.camera.flyTo({
            destination,
            orientation: {
              heading: 0.0,
              pitch: -Math.PI / 4.0,
              roll: 0.0
            }
          });
        }
      }
    })();
  </script>
</body>
</html>
""".lstrip()

        html = (
            html_template.replace("TOKEN_PLACEHOLDER", json.dumps(self.cesium_token))
            .replace("SCENE_JSON_PLACEHOLDER", scene_json_str)
        )
        return html