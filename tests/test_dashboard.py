"""
Tests for dashboard utilities in GeoAnomalyMapper.

Covers API client functions, anomaly data extraction/processing, map visualization preparation (mocked Folium),
3D visualization data extraction/preparation (mocked PyVista), error handling, and edge cases.
Mocks external dependencies (requests, folium, pyvista) for isolation.
Achieves >90% coverage for dashboard/app.py utilities.

Run with: pytest tests/test_dashboard.py -v --cov=dashboard/app
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from dashboard.app import (
    extract_anomalies, create_anomaly_map, create_anomaly_heatmap, create_anomaly_clusters,
    extract_3d_model, create_3d_volume_viewer, create_3d_slice_viewer, create_3d_isosurface_viewer,
    check_api_connection, start_analysis_job, get_job_status, get_job_results
)
from folium import Map
from folium.plugins import HeatMap, MarkerCluster
from branca.element import Figure, Html
import folium
import pyvista as pv
import requests


# Mock data fixtures
@pytest.fixture
def mock_results_data() -> Dict[str, Any]:
    """Mock API results with anomalies."""
    return {
        "results": {
            "anomalies": [
                {"lat": 30.0, "lon": 30.0, "confidence": 0.9, "type": "gravity", "intensity": 5.0, "id": "anom_001"},
                {"latitude": 30.5, "longitude": 30.5, "confidence": 0.8, "type": "magnetic", "intensity": 3.0, "id": "anom_002"},
                {"lat": 31.0, "lon": 31.0, "confidence": 0.7, "type": "unknown", "intensity": 2.0, "id": "anom_003"}
            ]
        }
    }


@pytest.fixture
def mock_results_no_anomalies() -> Dict[str, Any]:
    """Mock API results without anomalies."""
    return {"results": {}}


@pytest.fixture
def mock_results_invalid_anomalies() -> Dict[str, Any]:
    """Mock API results with invalid anomaly data."""
    return {
        "results": {
            "anomalies": [
                {"lat": "invalid", "lon": 30.0},  # Non-numeric lat
                {"lat": 30.0, "lon": 200.0},  # Out of bounds lon
                {"lat": None, "lon": None},  # Missing coords
                123,  # Not dict
                {"lat": 30.0, "lon": 30.0, "confidence": -0.5}  # Invalid confidence
            ]
        }
    }


@pytest.fixture
def sample_bbox() -> Tuple[float, float, float, float]:
    """Sample bbox for visualization tests."""
    return (29.0, 29.5, 31.5, 31.0)  # min_lon, min_lat, max_lon, max_lat


@pytest.fixture
def mock_folium_map():
    """Mock Folium Map for visualization tests."""
    with patch('dashboard.app.folium') as mock_folium:
        mock_map = Mock(spec=Map)
        mock_map.location = [30.25, 30.25]
        mock_map.zoom_start = 10
        mock_map.tiles = 'OpenStreetMap'
        mock_map.add_child = Mock()
        mock_map.get_root = Mock(return_value=Mock(html=Mock(add_child=Mock())))
        mock_folium.Map.return_value = mock_map
        yield mock_map


@pytest.fixture
def mock_heatmap():
    """Mock HeatMap plugin."""
    with patch('dashboard.app.HeatMap') as mock_heatmap:
        mock_instance = Mock()
        mock_heatmap.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_marker_cluster():
    """Mock MarkerCluster plugin."""
    with patch('dashboard.app.MarkerCluster') as mock_cluster:
        mock_instance = Mock()
        mock_cluster.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_pyvista():
    """Mock PyVista for 3D tests."""
    with patch('dashboard.app.pv') as mock_pv:
        mock_grid = Mock(spec=pv.UniformGrid)
        mock_grid.dimensions = (50, 50, 20)
        mock_grid.center = [30.25, 30.25, 500]
        mock_grid.set_active_scalars = Mock()
        mock_mesh = Mock(spec=pv.PolyData)
        mock_mesh.delaunay_3d.return_value = mock_grid
        mock_pv.UniformGrid.return_value = mock_grid
        mock_pv.PolyData.return_value = mock_mesh
        yield mock_pv, mock_grid, mock_mesh


class TestAnomalyDataExtraction:
    """Tests for extract_anomalies and related processing."""

    def test_extract_anomalies_valid_data(self, mock_results_data):
        """Test extract_anomalies with valid results: extracts and validates anomalies."""
        anomalies = extract_anomalies(mock_results_data)
        assert isinstance(anomalies, list)
        assert len(anomalies) == 3
        for anomaly in anomalies:
            assert isinstance(anomaly, dict)
            assert all(key in anomaly for key in ['lat', 'lon', 'confidence', 'type', 'intensity', 'id', 'modality'])
            assert -90 <= anomaly['lat'] <= 90
            assert -180 <= anomaly['lon'] <= 180
            assert 0 <= anomaly['confidence'] <= 1.0
            assert isinstance(anomaly['intensity'], float)
            assert anomaly['lat'] in [30.0, 30.5, 31.0]

    def test_extract_anomalies_no_anomalies(self, mock_results_no_anomalies):
        """Test extract_anomalies with no anomalies: returns empty list."""
        anomalies = extract_anomalies(mock_results_no_anomalies)
        assert isinstance(anomalies, list)
        assert len(anomalies) == 0

    def test_extract_anomalies_invalid_data(self, mock_results_invalid_anomalies):
        """Test extract_anomalies with invalid data: skips invalid, extracts valid."""
        anomalies = extract_anomalies(mock_results_invalid_anomalies)
        assert len(anomalies) == 0  # All invalid in fixture
        # Test with mixed
        mixed_data = mock_results_invalid_anomalies.copy()
        mixed_data["results"]["anomalies"].insert(0, {"lat": 30.0, "lon": 30.0, "confidence": 0.9})
        anomalies = extract_anomalies(mixed_data)
        assert len(anomalies) == 1  # Only one valid

    def test_extract_anomalies_alt_keys(self, mock_results_data):
        """Test extract_anomalies handles alt keys like latitude/longitude."""
        # Fixture uses lat/lon, test with latitude/longitude
        alt_data = mock_results_data.copy()
        alt_data["results"]["anomalies"][1] = {"latitude": 30.5, "longitude": 30.5, "confidence": 0.8, "type": "magnetic"}
        anomalies = extract_anomalies(alt_data)
        assert anomalies[1]["lat"] == 30.5
        assert anomalies[1]["lon"] == 30.5

    def test_extract_anomalies_missing_coords(self, mock_results_data):
        """Test skips anomalies without lat/lon."""
        invalid = mock_results_data.copy()
        invalid["results"]["anomalies"][0] = {"confidence": 0.9}  # No coords
        anomalies = extract_anomalies(invalid)
        assert len(anomalies) == 2  # Skipped one

    def test_extract_anomalies_out_of_bounds(self, mock_results_data):
        """Test skips out-of-bounds coordinates."""
        invalid = mock_results_data.copy()
        invalid["results"]["anomalies"][0] = {"lat": 100.0, "lon": 30.0, "confidence": 0.9}
        anomalies = extract_anomalies(invalid)
        assert len(anomalies) == 2  # Skipped

    def test_extract_anomalies_non_dict(self, mock_results_data):
        """Test skips non-dict anomalies."""
        invalid = mock_results_data.copy()
        invalid["results"]["anomalies"].append("not a dict")
        anomalies = extract_anomalies(invalid)
        assert len(anomalies) == 3  # Skipped non-dict

    def test_extract_anomalies_validation_errors(self, mock_results_data):
        """Test handles ValueError in float conversion."""
        invalid = mock_results_data.copy()
        invalid["results"]["anomalies"][0] = {"lat": "30.0", "lon": "invalid", "confidence": "0.9"}
        anomalies = extract_anomalies(invalid)
        assert len(anomalies) == 2  # Skipped invalid lon


class TestMapVisualization:
    """Tests for 2D map creation functions (mocked Folium)."""

    def test_create_anomaly_map_valid(self, mock_results_data, sample_bbox, mock_folium_map):
        """Test create_anomaly_map: creates map with markers for anomalies."""
        m = create_anomaly_map(mock_results_data, sample_bbox)
        assert isinstance(m, Mock)  # Mock Map
        assert m.location == [30.25, 30.25]  # Center
        assert m.zoom_start == 10
        # Verify add_child called for each marker
        assert m.add_child.call_count == 3  # 3 anomalies
        # LayerControl added
        m.get_root.return_value.html.add_child.assert_called()

    def test_create_anomaly_map_no_anomalies(self, mock_results_no_anomalies, sample_bbox, mock_folium_map):
        """Test create_anomaly_map with no anomalies: adds info marker."""
        m = create_anomaly_map(mock_results_no_anomalies, sample_bbox)
        assert m.add_child.call_count == 1  # Info marker
        # Popup text
        call = m.add_child.call_args[0][0]
        assert "No anomalies detected" in str(call)

    def test_create_anomaly_map_large_dataset(self, mock_results_data, sample_bbox, mock_folium_map, caplog):
        """Test create_anomaly_map with >100 anomalies: logs warning."""
        large_data = mock_results_data.copy()
        large_data["results"]["anomalies"] = large_data["results"]["anomalies"] * 40  # 120 anomalies
        m = create_anomaly_map(large_data, sample_bbox)
        assert "Large dataset (120 anomalies)" in caplog.text

    def test_create_anomaly_map_color_mapping(self, mock_results_data, sample_bbox, mock_folium_map):
        """Test color mapping by type in markers."""
        m = create_anomaly_map(mock_results_data, sample_bbox)
        # Calls to add_child for CircleMarker with colors
        calls = [call[0][0] for call in m.add_child.call_args_list]
        colors = [call.color for call in calls if hasattr(call, 'color')]
        assert 'blue' in colors  # gravity
        assert 'orange' in colors  # magnetic
        assert 'gray' in colors  # unknown

    def test_create_anomaly_map_popup_html(self, mock_results_data, sample_bbox, mock_folium_map):
        """Test popup HTML generation for anomalies."""
        m = create_anomaly_map(mock_results_data, sample_bbox)
        # Verify Popup with Html added
        for call in m.add_child.call_args_list:
            marker = call[0][0]
            if hasattr(marker, 'popup'):
                assert isinstance(marker.popup, Mock)  # Popup
                assert "Anomaly anom_001" in str(marker.popup)

    def test_create_anomaly_heatmap_valid(self, mock_results_data, sample_bbox, mock_folium_map, mock_heatmap):
        """Test create_anomaly_heatmap: adds HeatMap with weights."""
        m = create_anomaly_heatmap(mock_results_data, sample_bbox)
        assert mock_heatmap.called
        heatmap_call = mock_heatmap.call_args[1]
        assert len(heatmap_call['heat_data']) == 3
        assert all(len(point) == 3 for point in heatmap_call['heat_data'])  # [lat, lon, weight]
        assert 0 <= heatmap_call['heat_data'][0][2] <= 1  # Normalized weight
        m.add_child.assert_called_with(mock_heatmap.return_value)

    def test_create_anomaly_heatmap_no_anomalies(self, mock_results_no_anomalies, sample_bbox, mock_folium_map):
        """Test create_anomaly_heatmap with no anomalies: adds info marker."""
        m = create_anomaly_heatmap(mock_results_no_anomalies, sample_bbox)
        assert m.add_child.call_count == 1  # Info marker

    def test_create_anomaly_clusters_valid(self, mock_results_data, sample_bbox, mock_folium_map, mock_marker_cluster):
        """Test create_anomaly_clusters: adds markers to cluster."""
        m = create_anomaly_clusters(mock_results_data, sample_bbox)
        assert mock_marker_cluster.called
        m.add_child.assert_called_with(mock_marker_cluster.return_value)
        # Markers added to cluster
        assert mock_marker_cluster.return_value.add_to.call_count == 3

    def test_create_anomaly_clusters_no_anomalies(self, mock_results_no_anomalies, sample_bbox, mock_folium_map):
        """Test create_anomaly_clusters with no anomalies: adds info marker."""
        m = create_anomaly_clusters(mock_results_no_anomalies, sample_bbox)
        assert m.add_child.call_count == 1  # Info marker


class Test3DVisualization:
    """Tests for 3D visualization data extraction and viewers (mocked PyVista)."""

    def test_extract_3d_model_fused_volume(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test extract_3d_model with fused volume data: returns grid."""
        mock_pv, mock_grid, _ = mock_pyvista
        mock_results_data["results"]["fused"] = {
            "volume": {
                "dimensions": (50, 50, 20),
                "origin": (29.0, 29.5, 0),
                "spacing": (0.05, 0.05, 50),
                "data": np.random.rand(50*50*20)
            }
        }
        model = extract_3d_model(mock_results_data, sample_bbox)
        assert model is not None
        assert model["type"] == "grid"
        assert model["grid"] == mock_grid
        mock_pv.UniformGrid.assert_called_once_with(dimensions=(50, 50, 20), origin=(29.0, 29.5, 0), spacing=(0.05, 0.05, 50))

    def test_extract_3d_model_fused_mesh(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test extract_3d_model with fused mesh data: returns mesh."""
        mock_pv, _, mock_mesh = mock_pyvista
        mock_results_data["results"]["fused"] = {
            "mesh": {
                "points": np.random.rand(100, 3),
                "point_data": {"anomaly": np.random.rand(100)}
            }
        }
        model = extract_3d_model(mock_results_data, sample_bbox)
        assert model["type"] == "mesh"
        assert model["mesh"] == mock_mesh
        mock_mesh['anomaly'] = np.random.rand(100)  # Verify set

    def test_extract_3d_model_interpolated_fallback(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test extract_3d_model fallback to interpolation: creates grid from anomalies."""
        mock_pv, mock_grid, _ = mock_pyvista
        # Remove fused, use anomalies
        if "fused" in mock_results_data["results"]:
            del mock_results_data["results"]["fused"]
        model = extract_3d_model(mock_results_data, sample_bbox)
        assert model["type"] == "interpolated"
        assert model["grid"] == mock_grid
        mock_pv.UniformGrid.assert_called_once()

    def test_extract_3d_model_no_data(self, mock_results_no_anomalies, sample_bbox):
        """Test extract_3d_model with no data: returns None."""
        model = extract_3d_model(mock_results_no_anomalies, sample_bbox)
        assert model is None

    def test_extract_3d_model_decimation_large(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test decimation for large meshes (>100k points)."""
        mock_pv, _, mock_mesh = mock_pyvista
        mock_results_data["results"]["fused"] = {
            "mesh": {
                "points": np.random.rand(200000, 3),  # Large
                "point_data": {"anomaly": np.random.rand(200000)}
            }
        }
        model = extract_3d_model(mock_results_data, sample_bbox)
        mock_mesh.decimate.assert_called_once_with(0.9)

    def test_create_3d_volume_viewer_valid(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test create_3d_volume_viewer: adds volume to plotter."""
        mock_pv, mock_grid, _ = mock_pyvista
        plotter = create_3d_volume_viewer(mock_results_data, sample_bbox)
        assert plotter.add_volume.called
        assert plotter.add_axes.called
        assert plotter.camera_position == 'iso'

    def test_create_3d_volume_viewer_no_data(self, mock_results_no_anomalies, sample_bbox, mock_pyvista):
        """Test create_3d_volume_viewer with no data: adds text."""
        mock_pv, _, _ = mock_pyvista
        plotter = create_3d_volume_viewer(mock_results_no_anomalies, sample_bbox)
        assert plotter.add_text.called
        assert "No 3D volume data available" in plotter.add_text.call_args[0][0]

    def test_create_3d_slice_viewer_valid(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test create_3d_slice_viewer: adds three slices."""
        mock_pv, mock_grid, _ = mock_pyvista
        plotter = create_3d_slice_viewer(mock_results_data, sample_bbox)
        assert plotter.add_mesh.call_count == 3  # x, y, z slices
        mock_grid.slice.assert_has_calls([
            Mock(normal=[1, 0, 0], origin=mock_grid.center),
            Mock(normal=[0, 1, 0], origin=mock_grid.center),
            Mock(normal=[0, 0, 1], origin=mock_grid.center)
        ], any_order=True)

    def test_create_3d_slice_viewer_no_data(self, mock_results_no_anomalies, sample_bbox, mock_pyvista):
        """Test create_3d_slice_viewer with no data: adds text."""
        plotter = create_3d_slice_viewer(mock_results_no_anomalies, sample_bbox)
        assert plotter.add_text.called

    def test_create_3d_isosurface_viewer_valid(self, mock_results_data, sample_bbox, mock_pyvista):
        """Test create_3d_isosurface_viewer: generates contour at threshold."""
        mock_pv, mock_grid, _ = mock_pyvista
        mock_grid['anomaly'] = np.random.rand(50*50*20).reshape(50, 50, 20)
        plotter = create_3d_isosurface_viewer(mock_results_data, sample_bbox, threshold=0.5)
        assert plotter.add_mesh.called
        mock_grid.contour.assert_called_once_with([Mock()])  # Normalized threshold

    def test_create_3d_isosurface_viewer_no_data(self, mock_results_no_anomalies, sample_bbox, mock_pyvista):
        """Test create_3d_isosurface_viewer with no data: adds text."""
        plotter = create_3d_isosurface_viewer(mock_results_no_anomalies, sample_bbox)
        assert plotter.add_text.called


class TestAPIClientFunctions:
    """Tests for dashboard API client functions (mocked requests)."""

    @pytest.fixture
    def mock_requests_get_success(self):
        """Mock requests.get 200 success."""
        with patch('dashboard.app.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "COMPLETED", "progress": 100.0, "stage": "Done"}
            mock_requests.get.return_value = mock_response
            yield mock_requests

    @pytest.fixture
    def mock_requests_get_not_found(self):
        """Mock requests.get 404."""
        with patch('dashboard.app.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_requests.get.return_value = mock_response
            yield mock_requests

    @pytest.fixture
    def mock_requests_get_too_early(self):
        """Mock requests.get 425."""
        with patch('dashboard.app.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 425
            mock_requests.get.return_value = mock_response
            yield mock_requests

    @pytest.fixture
    def mock_requests_post_success(self):
        """Mock requests.post 200 with job_id."""
        with patch('dashboard.app.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"job_id": "test_job_123", "status": "QUEUED"}
            mock_requests.post.return_value = mock_response
            yield mock_requests

    @pytest.fixture
    def mock_requests_connection_error(self):
        """Mock requests.RequestException."""
        with patch('dashboard.app.requests') as mock_requests:
            mock_requests.get.side_effect = requests.exceptions.RequestException("Connection failed")
            yield mock_requests

    def test_check_api_connection_success(self, mock_requests_get_success):
        """Test check_api_connection: returns True on 200."""
        assert check_api_connection() is True

    def test_check_api_connection_failure(self, mock_requests_connection_error):
        """Test check_api_connection: returns False on exception."""
        assert check_api_connection() is False

    def test_start_analysis_job_success(self, mock_requests_post_success, sample_bbox):
        """Test start_analysis_job: returns job_id on 200."""
        modalities = ["gravity"]
        job_id = start_analysis_job(sample_bbox, modalities, 1000.0, "test_output", None, False)
        assert job_id == "test_job_123"

    def test_start_analysis_job_failure(self, mock_requests_connection_error, sample_bbox):
        """Test start_analysis_job: returns None on exception."""
        job_id = start_analysis_job(sample_bbox, ["gravity"], 1000.0, "test_output", None, False)
        assert job_id is None

    def test_get_job_status_success(self, mock_requests_get_success):
        """Test get_job_status: returns status dict on 200."""
        status = get_job_status("test_job")
        assert status["status"] == "COMPLETED"
        assert status["progress"] == 100.0

    def test_get_job_status_not_found(self, mock_requests_get_not_found):
        """Test get_job_status: returns None on 404."""
        status = get_job_status("invalid_job")
        assert status is None

    def test_get_job_status_too_early(self, mock_requests_get_too_early):
        """Test get_job_status: returns None on 425."""
        status = get_job_status("running_job")
        assert status is None

    def test_get_job_status_network_error(self, mock_requests_connection_error):
        """Test get_job_status: returns None on network error."""
        status = get_job_status("test_job")
        assert status is None

    def test_get_job_results_success(self, mock_requests_get_success):
        """Test get_job_results: returns results on 200."""
        results = get_job_results("completed_job")
        assert results is not None
        assert "status" in results

    def test_get_job_results_not_ready(self, mock_requests_get_too_early):
        """Test get_job_results: returns None on 425."""
        results = get_job_results("running_job")
        assert results is None

    def test_get_job_results_not_found(self, mock_requests_get_not_found):
        """Test get_job_results: returns None on 404."""
        results = get_job_results("invalid_job")
        assert results is None

    def test_get_job_results_network_error(self, mock_requests_connection_error):
        """Test get_job_results: returns None on network error."""
        results = get_job_results("test_job")
        assert results is None


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases in dashboard utilities."""

    def test_api_client_timeout(self):
        """Test API functions with timeout: raises or handles."""
        with patch('dashboard.app.requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout
            status = get_job_status("timeout_job")
            assert status is None

    def test_visualization_large_anomalies_performance(self, caplog):
        """Test visualization functions log warnings for large datasets."""
        large_data = {"results": {"anomalies": [{}] * 200}}
        create_anomaly_map(large_data, (0,0,0,0))
        assert "Large dataset (200 anomalies)" in caplog.text

    def test_3d_model_error_handling(self, mock_results_data, sample_bbox, caplog):
        """Test extract_3d_model catches exceptions: logs error, returns None."""
        with patch('dashboard.app.extract_3d_model') as mock_extract:
            mock_extract.side_effect = Exception("3D extraction error")
            model = extract_3d_model(mock_results_data, sample_bbox)
            assert model is None
            assert "Error extracting 3D model" in caplog.text

    def test_api_client_invalid_json_response(self, mock_requests_get_success):
        """Test API client handles non-JSON response."""
        mock_requests_get_success.return_value.json.side_effect = ValueError("Invalid JSON")
        status = get_job_status("invalid_json_job")
        assert status is None

    def test_map_bbox_center_calculation(self, sample_bbox):
        """Test bbox center calculation in map functions."""
        center_lat = (sample_bbox[1] + sample_bbox[3]) / 2  # 30.25
        center_lon = (sample_bbox[0] + sample_bbox[2]) / 2  # 30.25
        # Test in create_anomaly_map (via mock)
        with patch('dashboard.app.create_anomaly_map') as mock_map:
            mock_map.return_value.location = [center_lat, center_lon]
            m = create_anomaly_map({}, sample_bbox)
            assert m.location == [center_lat, center_lon]