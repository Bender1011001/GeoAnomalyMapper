"""Unit tests for ingestion base components."""

import pytest
import numpy as np
from gam.ingestion.base import DataSource
from gam.ingestion.data_structures import RawData
from gam.core.exceptions import DataValidationError


class TestDataSource:
    """Test cases for the DataSource abstract base class."""

    def test_abstract_cannot_instantiate(self):
        """Test that DataSource cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class DataSource"):
            DataSource()

    def test_subclass_must_implement_fetch_data(self):
        """Test that subclasses must implement fetch_data method."""

        class IncompleteSource(DataSource):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteSource"):
            IncompleteSource()

    def test_subclass_can_implement_fetch_data(self):
        """Test that subclasses can implement fetch_data method."""

        class CompleteSource(DataSource):
            def fetch_data(self, bbox, **kwargs):
                return RawData(np.array([1.0]), {}, "EPSG:4326")

        source = CompleteSource()
        result = source.fetch_data((0, 1, 0, 1))
        assert isinstance(result, RawData)


class TestRawData:
    """Test cases for the RawData dataclass."""

    @pytest.mark.parametrize(
        "data,metadata,crs",
        [
            (np.array([1.0, 2.0, 3.0]), {"source": "test"}, "EPSG:4326"),
            (np.array([[1, 2], [3, 4]]), {}, 4326),
            (np.array([0.0]), {"units": "mGal"}, "WGS84"),
            (np.array([1.0]), {}, "EPSG:4326"),
        ],
    )
    def test_instantiation_valid(self, data, metadata, crs):
        """Test valid instantiation of RawData."""
        raw = RawData(data, metadata, crs)
        assert np.array_equal(raw.data, data)
        assert raw.metadata == metadata
        assert raw.crs == crs

    @pytest.mark.parametrize(
        "data,metadata,crs",
        [
            ([1.0, 2.0], {}, "EPSG:4326"),  # data not ndarray
            (np.array([]), {}, "EPSG:4326"),  # empty array
            (np.array([1.0]), "not_dict", "EPSG:4326"),  # metadata not dict
            (np.array([1.0]), {}, ""),  # empty crs string
            (np.array([1.0]), {}, None),  # crs None
        ],
    )
    def test_instantiation_invalid(self, data, metadata, crs):
        """Test invalid instantiation of RawData."""
        raw = RawData(data, metadata, crs)
        with pytest.raises(DataValidationError):
            raw.validate()

    def test_validate_valid(self):
        """Test validate method with valid data."""
        raw = RawData(np.array([1.0, 2.0]), {"source": "test"}, "EPSG:4326")
        raw.validate()  # Should not raise

    @pytest.mark.parametrize(
        "data,metadata,crs,expected_error",
        [
            ([1.0], {}, "EPSG:4326", "data must be a NumPy array"),
            (np.array([]), {}, "EPSG:4326", "data array must not be empty"),
            (np.array([1.0]), "not_dict", "EPSG:4326", "metadata must be a dictionary"),
            (np.array([1.0]), {}, "", "crs must be a non-empty string or integer"),
        ],
    )
    def test_validate_invalid(self, data, metadata, crs, expected_error):
        """Test validate method with invalid data."""
        raw = RawData(data, metadata, crs)
        with pytest.raises(DataValidationError, match=expected_error):
            raw.validate()

    def test_to_dict(self):
        """Test to_dict method."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = {"source": "test", "units": "mGal"}
        crs = "EPSG:4326"
        raw = RawData(data, metadata, crs)
        result = raw.to_dict()
        expected = {
            "data": [1.0, 2.0, 3.0],
            "metadata": metadata,
            "crs": crs,
        }
        assert result == expected

    def test_from_dict_valid(self):
        """Test from_dict classmethod with valid data."""
        data_dict = {
            "data": [1.0, 2.0, 3.0],
            "metadata": {"source": "test"},
            "crs": "EPSG:4326",
        }
        raw = RawData.from_dict(data_dict)
        assert np.array_equal(raw.data, np.array([1.0, 2.0, 3.0]))
        assert raw.metadata == {"source": "test"}
        assert raw.crs == "EPSG:4326"

    @pytest.mark.parametrize(
        "data_dict",
        [
            {},  # missing data key
            {"data": []},  # empty data
        ],
    )
    def test_from_dict_invalid(self, data_dict):
        """Test from_dict classmethod with invalid data."""
        with pytest.raises(DataValidationError):
            RawData.from_dict(data_dict)

    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = RawData(np.array([1.0, 2.0]), {"source": "test"}, "EPSG:4326")
        serialized = original.to_dict()
        deserialized = RawData.from_dict(serialized)
        assert np.array_equal(original.data, deserialized.data)
        assert original.metadata == deserialized.metadata
        assert original.crs == deserialized.crs