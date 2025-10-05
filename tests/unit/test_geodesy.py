"""Unit tests for geodesy.py module."""

import pytest
import numpy as np
import numpy.testing as npt
import pyproj
from gam.core.geodesy import ensure_crs, build_transformer, geodetic_to_projected, bbox_extent_meters
from gam.core.exceptions import ConfigurationError as ConfigError


class TestEnsureCRS:
    """Tests for ensure_crs function."""

    @pytest.mark.parametrize("crs_input,expected_epsg", [
        ("EPSG:4326", 4326),
        ("EPSG:32631", 32631),
        ("+proj=utm +zone=31 +datum=WGS84", None),  # Proj string, no EPSG
        ("GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]", 4326),  # WKT
    ])
    def test_valid_crs_inputs(self, crs_input, expected_epsg):
        """Test ensure_crs with valid inputs."""
        crs = ensure_crs(crs_input)
        assert isinstance(crs, pyproj.CRS)
        if expected_epsg:
            assert crs.to_epsg() == expected_epsg

    @pytest.mark.parametrize("invalid_input", [
        "INVALID",
        12345,
        None,
        "",
    ])
    def test_invalid_crs_inputs(self, invalid_input):
        """Test ensure_crs raises ConfigError for invalid inputs."""
        with pytest.raises(ConfigError):
            ensure_crs(invalid_input)


class TestBuildTransformer:
    """Tests for build_transformer function."""

    def test_build_transformer_basic(self):
        """Test building a transformer between two CRS."""
        src_crs = pyproj.CRS("EPSG:4326")
        dst_crs = pyproj.CRS("EPSG:32631")  # UTM zone 31N
        transformer = build_transformer(src_crs, dst_crs)
        assert isinstance(transformer, pyproj.Transformer)

    def test_transformer_always_xy(self):
        """Test that transformer has always_xy=True."""
        src_crs = pyproj.CRS("EPSG:4326")
        dst_crs = pyproj.CRS("EPSG:32631")
        transformer = build_transformer(src_crs, dst_crs)
        # Transform a point and check order
        x, y = transformer.transform(0.0, 0.0)
        # For UTM, x should be positive, y around 0
        assert x > 0
        assert abs(y) < 1000  # Near equator


class TestGeodeticToProjected:
    """Tests for geodetic_to_projected function."""

    @pytest.mark.parametrize("lon,lat,dst_epsg,expected_x,expected_y", [
        # Null Island (0,0) to UTM 31N
        (np.array([0.0]), np.array([0.0]), 32631, 166021.44308053973, 0.0),
        # Equator, Prime Meridian
        (np.array([0.0]), np.array([0.0]), 32631, 166021.44308053973, 0.0),
        # Array of points
        (np.array([0.0, 1.0]), np.array([0.0, 0.0]), 32631, np.array([166021.44308053973, 277404.560324]), np.array([0.0, 0.0]))  # Approximate
    ])
    def test_geodetic_to_projected_known_values(self, lon, lat, dst_epsg, expected_x, expected_y):
        """Test geodetic_to_projected with known coordinate transformations."""
        dst_crs = pyproj.CRS(f"EPSG:{dst_epsg}")
        x, y = geodetic_to_projected(lon, lat, dst_crs)
        npt.assert_allclose(x, expected_x, rtol=1e-6)
        npt.assert_allclose(y, expected_y, rtol=1e-6)

    def test_geodetic_to_projected_arrays(self):
        """Test with numpy arrays of different sizes."""
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([0.0, 0.0, 0.0])
        dst_crs = pyproj.CRS("EPSG:32631")
        x, y = geodetic_to_projected(lon, lat, dst_crs)
        assert len(x) == 3
        assert len(y) == 3
        # Check first point
        npt.assert_allclose(x[0], 166021.44308053973, rtol=1e-6)
        npt.assert_allclose(y[0], 0.0, rtol=1e-6)


class TestBboxExtentMeters:
    """Tests for bbox_extent_meters function."""

    @pytest.mark.parametrize("bbox,dst_epsg,expected_width,expected_height", [
        # Small bbox around null island in UTM 31N
        ((-1.0, -1.0, 1.0, 1.0), 32631, 222638.0, 221603.665093),  # Approximate meters
        # Equator bbox
        ((0.0, 0.0, 1.0, 1.0), 32631, 111319.490793, 110574.0),  # Width approx 111km, height 110km
    ])
    def test_bbox_extent_meters_known_values(self, bbox, dst_epsg, expected_width, expected_height):
        """Test bbox_extent_meters with known bounding boxes."""
        dst_crs = pyproj.CRS(f"EPSG:{dst_epsg}")
        width, height = bbox_extent_meters(bbox, dst_crs)
        npt.assert_allclose(width, expected_width, rtol=1e-3)
        npt.assert_allclose(height, expected_height, rtol=1e-3)

    @pytest.mark.parametrize("invalid_bbox", [
        (1.0, 0.0, 0.0, 1.0),  # min_lon >= max_lon
        (0.0, 1.0, 1.0, 0.0),  # min_lat >= max_lat
        (0.0, 0.0, 0.0, 0.0),  # zero extent
    ])
    def test_bbox_extent_meters_invalid_bbox(self, invalid_bbox):
        """Test bbox_extent_meters raises ValueError for invalid bounding boxes."""
        dst_crs = pyproj.CRS("EPSG:32631")
        with pytest.raises(ValueError, match="Invalid bounding box"):
            bbox_extent_meters(invalid_bbox, dst_crs)

    def test_bbox_extent_meters_poles(self):
        """Test bbox_extent_meters near poles (edge case)."""
        # Bbox near north pole, using polar stereographic
        bbox = (-180.0, 85.0, 180.0, 90.0)
        dst_crs = pyproj.CRS("EPSG:3413")  # NSIDC North Polar Stereographic
        width, height = bbox_extent_meters(bbox, dst_crs)
        assert width > 0
        assert height > 0
        # Exact values depend on projection, but should be reasonable