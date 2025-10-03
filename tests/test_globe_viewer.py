import json

import numpy as np
from gam.visualization.globe_viewer import GlobeViewer


def test_scene_build_minimal():
    a = np.random.RandomState(0).rand(64, 64)
    bbox = (-122.3, 37.6, -121.9, 38.0)

    gv = GlobeViewer(cesium_token="test")
    gv.add_anomaly_heatmap(a, bbox)
    gv.set_camera(-122.1, 37.8, 120000.0)

    html = gv.render_streamlit_html()
    assert isinstance(html, str)
    assert "Cesium.Viewer" in html

    s = gv.export_scene_config()
    assert isinstance(s, str)
    assert '"layers":' in s


def test_export_scene_json_keys():
    # Use a small deterministic array for speed and determinism
    a = np.zeros((8, 8))
    bbox = (-122.3, 37.6, -121.9, 38.0)

    gv = GlobeViewer(cesium_token="test")
    gv.add_anomaly_heatmap(a, bbox)
    gv.set_camera(-122.1, 37.8, 120000.0)

    s = gv.export_scene_config()
    obj = json.loads(s)
    assert isinstance(obj, dict)
    assert "layers" in obj
    assert "camera" in obj