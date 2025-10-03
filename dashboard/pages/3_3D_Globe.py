"""3D Globe visualization using CesiumJS via GlobeViewer."""
import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Prefer secrets; fallback to environment variable CESIUM_TOKEN.
# GlobeViewer import path confirmed by discovery.
from gam.visualization.globe_viewer import GlobeViewer  # type: ignore
from gam.core.artifacts import save_scene_config

# Attempt to import a results loader; discovery indicates it is absent.
# Keep a defensive import to support future availability without breaking this page.
try:
    # Expected (but currently absent) API:
    # from gam.core.pipeline import load_results
    from gam.core.pipeline import load_results  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - robustness for environments without this symbol
    load_results = None  # type: ignore


def _get_demo_data() -> dict:
    """Return demo anomalies and bbox for visualization.

    Returns
    -------
    dict
        {
            "anomalies": np.ndarray shape (H, W) with values in [0, 1],
            "bbox": tuple(min_lon, min_lat, max_lon, max_lat)
        }
    """
    rng = np.random.RandomState(42)
    anomalies = rng.random((200, 200))  # values already in [0, 1]
    # Example bbox near Northern California (min_lon, min_lat, max_lon, max_lat)
    bbox = (-122.6, 38.1, -121.7, 38.7)
    return {"anomalies": anomalies, "bbox": bbox}


def _validate_and_normalize_payload(payload: dict) -> dict:
    """Validate payload structure and coerce anomalies to np.ndarray.

    Expected structure:
    - payload["anomalies"]: 2D np.ndarray-like with values [0, 1]
    - payload["bbox"]: (min_lon, min_lat, max_lon, max_lat)

    Returns a sanitized copy.
    """
    if not isinstance(payload, dict):
        raise ValueError("Results payload must be a dict")
    if "anomalies" not in payload or "bbox" not in payload:
        raise KeyError("Missing required keys 'anomalies' and/or 'bbox'")
    anomalies = payload["anomalies"]
    if not isinstance(anomalies, np.ndarray):
        anomalies = np.asarray(anomalies)
    if anomalies.ndim != 2:
        raise ValueError("'anomalies' must be a 2D array")
    bbox = tuple(payload["bbox"])
    if len(bbox) != 4:
        raise ValueError("'bbox' must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)")
    return {"anomalies": anomalies, "bbox": bbox}


# Streamlit page setup
st.set_page_config(page_title="3D Globe", layout="wide")
st.title("🌍 GeoAnomalyMapper — 3D Globe")

# Sidebar controls
with st.sidebar:
    st.subheader("Controls")
    data_source = st.radio("Data Source", ["Load Analysis", "Demo"], index=0)
    analysis_id = None
    if data_source == "Load Analysis":
        analysis_id = st.text_input("Analysis ID", value="latest")
    opacity = st.slider("Heatmap Opacity", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    show_points = st.checkbox("Show high anomalies as cylinders", value=True)

# Acquire data
data = None
if data_source == "Load Analysis":
    if load_results is None:
        st.warning(
            "Analysis loader not available (gam.core.pipeline.load_results missing). Falling back to Demo data."
        )
        data = _get_demo_data()
    else:
        try:
            data = load_results(analysis_id)  # type: ignore[misc]
            data = _validate_and_normalize_payload(data)
        except Exception as exc:
            st.warning(
                f"Failed to load analysis '{analysis_id}' ({exc}). Falling back to Demo data."
            )
            data = _get_demo_data()
else:
    data = _get_demo_data()

# If for any reason data is still None, instruct the user.
if not data:
    st.info("Select or generate data in the sidebar.")
else:
    anomalies: np.ndarray = data["anomalies"]
    # bbox order: (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = data["bbox"]

    # Build GlobeViewer scene
    token = (st.secrets.get("CESIUM_TOKEN", None) or os.getenv("CESIUM_TOKEN") or "")
    gv = GlobeViewer(cesium_token=token)

    # Add heatmap overlay with user-controlled opacity; colormap 'hot' by default
    gv.add_anomaly_heatmap(anomalies, (min_lon, min_lat, max_lon, max_lat), cmap="hot", opacity=opacity)

    # Optionally add top anomalies as cylinders
    if show_points:
        # Compute top 1% threshold
        threshold = float(np.quantile(anomalies, 0.99))
        indices = np.argwhere(anomalies >= threshold)  # rows -> (y, x) pairs
        if indices.size > 0:
            values = anomalies[indices[:, 0], indices[:, 1]]
            order = np.argsort(-values)  # descending by anomaly value
            # Cap to at most 150 points for clarity/performance
            top_k = min(150, indices.shape[0])
            sel = indices[order[:top_k]]

            h, w = anomalies.shape  # h = rows (lat), w = cols (lon)
            lon_span = max_lon - min_lon
            lat_span = max_lat - min_lat

            points = []
            for (y, x) in sel:
                # Map grid indices to geographic coordinates.
                # x -> longitude (min_lon .. max_lon), y -> latitude (min_lat .. max_lat)
                lon = min_lon + (float(x) / max(1, (w - 1))) * lon_span
                lat = min_lat + (float(y) / max(1, (h - 1))) * lat_span
                val = float(anomalies[y, x])
                points.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        "value": val,  # used by GlobeViewer to drive cylinder height
                        "label": f"{val:.3f}",
                    }
                )

            gv.add_point_entities(points, shape="cylinder", name="Top anomalies (>= 99th pct)")

    # Set camera to bbox center. Height is in meters.
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    gv.set_camera(center_lon, center_lat, height_m=150000.0)

    # Render embeddable HTML (contains "Cesium.Viewer") and display
    html = gv.render_streamlit_html()
    components.html(html, height=820, scrolling=False)

    # Download current scene configuration as JSON
    st.download_button(
        label="Download scene.json",
        data=gv.export_scene_config(),
        file_name="scene.json",
        mime="application/json",
    )

    # Save current scene.json to server
    save_id = None
    if data_source == "Demo":
        save_id = st.text_input(
            "Save as Analysis ID",
            value="demo",
            key="save_analysis_id_main",
            help="Provide an identifier to save this scene configuration under data/outputs/state/{id}/scene.json",
        )

    if st.button("Save scene.json", type="primary"):
        # Derive effective_id from the appropriate source
        if data_source == "Load Analysis":
            effective_id = (analysis_id or "").strip()  # reuse sidebar input
        else:
            effective_id = (save_id or "").strip()

        if not effective_id:
            st.error("Please provide a non-empty Analysis ID to save the scene.")
        else:
            saved_path = save_scene_config(effective_id, gv.export_scene_config())
            st.success(f"Saved scene to {saved_path}")
            st.code(f"/analysis/{effective_id}/scene.json", language="bash")