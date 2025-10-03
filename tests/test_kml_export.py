"""Test for KML export functionality."""

import pytest
import numpy as np
from pathlib import Path
from GeoAnomalyMapper.gam.visualization.kml_export import export_anomaly_kml


def test_export_anomaly_kml(tmp_path: Path):
    """Test basic KML export with small anomalies array."""
    # Create minimal 2x2 anomalies array with NaN to test normalization
    anomalies = np.array([[1.0, 2.0], [np.nan, 3.0]])

    # Small bbox in degrees
    bbox = (-1.0, -1.0, 1.0, 1.0)

    # Output path using tmp_path
    out_kml = tmp_path / "test_overlay.kml"

    # Call the export function
    export_anomaly_kml(anomalies, bbox, str(out_kml))

    # Assert file exists and has content
    assert out_kml.exists()
    assert out_kml.stat().st_size > 0

    # Read and check content for key KML tags
    with open(out_kml, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<kml" in content
    assert any(tag in content for tag in ["<GroundOverlay", "<Placemark"])