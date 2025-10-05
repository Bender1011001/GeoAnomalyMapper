"""Unit tests for gam.core.tiles module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from gam.core.tiles import (
    tiles_10x10_ids,
    tile_bounds_10x10,
    grid_spec_0p1,
    _lat_label_from_min,
    _lon_label_from_min,
    _lat_bands_10deg,
    _lon_bands_10deg,
    _parse_tile_id,
    TILE_SIZE_DEG,
    RESOLUTION_0P1,
    PIXELS_PER_TILE,
    WORLD_BOUNDS,
)


class TestLatLabelFromMin:
    """Test _lat_label_from_min function."""

    @pytest.mark.parametrize(
        "min_lat,expected",
        [
            (0, "N00"),
            (10, "N10"),
            (80, "N80"),
            (-10, "S10"),
            (-90, "S90"),
        ],
    )
    def test_valid_labels(self, min_lat, expected):
        assert _lat_label_from_min(min_lat) == expected

    @pytest.mark.parametrize("min_lat", [5, -5, 95, -95])
    def test_invalid_multiple_of_10(self, min_lat):
        with pytest.raises(ValueError, match="Latitude band minimum must be multiple of 10 degrees"):
            _lat_label_from_min(min_lat)


class TestLonLabelFromMin:
    """Test _lon_label_from_min function."""

    @pytest.mark.parametrize(
        "min_lon,expected",
        [
            (0, "E000"),
            (10, "E010"),
            (170, "E170"),
            (-10, "W010"),
            (-180, "W180"),
        ],
    )
    def test_valid_labels(self, min_lon, expected):
        assert _lon_label_from_min(min_lon) == expected

    @pytest.mark.parametrize("min_lon", [5, -5, 175, -185])
    def test_invalid_multiple_of_10(self, min_lon):
        with pytest.raises(ValueError, match="Longitude band minimum must be multiple of 10 degrees"):
            _lon_label_from_min(min_lon)


class TestLatBands10deg:
    """Test _lat_bands_10deg function."""

    def test_bands_structure(self):
        bands = _lat_bands_10deg()
        assert len(bands) == 18  # From -90 to 80, step 10
        assert bands[0] == ("S90", -90.0, -80.0)
        assert bands[-1] == ("N80", 80.0, 90.0)

    def test_bands_coverage(self):
        bands = _lat_bands_10deg()
        # Check no gaps or overlaps
        for i in range(len(bands) - 1):
            assert bands[i][2] == bands[i + 1][1]


class TestLonBands10deg:
    """Test _lon_bands_10deg function."""

    def test_bands_structure(self):
        bands = _lon_bands_10deg()
        assert len(bands) == 36  # From -180 to 170, step 10
        assert bands[0] == ("W180", -180.0, -170.0)
        assert bands[-1] == ("E170", 170.0, 180.0)

    def test_bands_coverage(self):
        bands = _lon_bands_10deg()
        # Check no gaps or overlaps
        for i in range(len(bands) - 1):
            assert bands[i][2] == bands[i + 1][1]


class TestTiles10x10Ids:
    """Test tiles_10x10_ids function."""

    def test_ids_length(self):
        ids = tiles_10x10_ids()
        assert len(ids) == 18 * 36  # 648 tiles

    def test_ids_format(self):
        ids = tiles_10x10_ids()
        for tile_id in ids:
            assert tile_id.startswith("t_")
            assert len(tile_id) == 10  # t_S90_W180

    def test_ids_order(self):
        ids = tiles_10x10_ids()
        # First few should be S90 with W180 to E170
        assert ids[0] == "t_S90_W180"
        assert ids[35] == "t_S90_E170"
        assert ids[36] == "t_S80_W180"

    def test_ids_uniqueness(self):
        ids = tiles_10x10_ids()
        assert len(set(ids)) == len(ids)


class TestParseTileId:
    """Test _parse_tile_id function."""

    @pytest.mark.parametrize(
        "tile_id,expected",
        [
            ("t_N00_E000", ("N", 0, "E", 0)),
            ("t_N20_E120", ("N", 20, "E", 120)),
            ("t_S10_W010", ("S", 10, "W", 10)),
            ("t_S90_W180", ("S", 90, "W", 180)),
        ],
    )
    def test_valid_parse(self, tile_id, expected):
        assert _parse_tile_id(tile_id) == expected

    @pytest.mark.parametrize(
        "tile_id",
        [
            "t_N05_E000",  # Not multiple of 10
            "t_N00_E005",
            "t_N95_E000",  # Out of range
            "t_S95_W000",
            "t_N00_E175",  # >170 for E
            "t_N00_W185",  # >180 for W
            "invalid",
            "t_N00_E000_extra",
        ],
    )
    def test_invalid_parse(self, tile_id):
        with pytest.raises(ValueError):
            _parse_tile_id(tile_id)


class TestTileBounds10x10:
    """Test tile_bounds_10x10 function."""

    @pytest.mark.parametrize(
        "tile_id,expected_bounds",
        [
            ("t_N00_E000", (0.0, 0.0, 10.0, 10.0)),
            ("t_S10_W010", (-10.0, -10.0, 0.0, 0.0)),
            ("t_N20_E120", (120.0, 20.0, 130.0, 30.0)),
            ("t_S90_W180", (-180.0, -90.0, -170.0, -80.0)),
            ("t_N80_E170", (170.0, 80.0, 180.0, 90.0)),
        ],
    )
    def test_valid_bounds(self, tile_id, expected_bounds):
        bounds = tile_bounds_10x10(tile_id)
        assert_allclose(bounds, expected_bounds, rtol=1e-10)

    @pytest.mark.parametrize("tile_id", ["invalid", "t_N95_E000"])
    def test_invalid_tile_id(self, tile_id):
        with pytest.raises(ValueError):
            tile_bounds_10x10(tile_id)


class TestGridSpec0p1:
    """Test grid_spec_0p1 function."""

    def test_grid_spec_structure(self):
        spec = grid_spec_0p1()
        required_keys = ["transform", "pixel_size", "bounds", "width", "height", "crs"]
        for key in required_keys:
            assert key in spec

    def test_grid_spec_values(self):
        spec = grid_spec_0p1()
        assert spec["pixel_size"] == (0.1, 0.1)
        assert_allclose(spec["bounds"], (-180.0, -90.0, 180.0, 90.0), rtol=1e-10)
        assert spec["width"] == 3600
        assert spec["height"] == 1800
        assert spec["crs"] == "EPSG:4326"

    def test_transform(self):
        spec = grid_spec_0p1()
        transform = spec["transform"]
        # Check origin
        assert_allclose(transform.c, -180.0, rtol=1e-10)  # x origin
        assert_allclose(transform.f, 90.0, rtol=1e-10)    # y origin
        assert_allclose(transform.a, 0.1, rtol=1e-10)     # x pixel size
        assert_allclose(transform.e, -0.1, rtol=1e-10)    # y pixel size (negative for north-up)


class TestConstants:
    """Test module constants."""

    def test_tile_size_deg(self):
        assert TILE_SIZE_DEG == 10.0

    def test_resolution_0p1(self):
        assert RESOLUTION_0P1 == 0.1

    def test_pixels_per_tile(self):
        assert PIXELS_PER_TILE == 100

    def test_world_bounds(self):
        assert_allclose(WORLD_BOUNDS, (-180.0, -90.0, 180.0, 90.0), rtol=1e-10)