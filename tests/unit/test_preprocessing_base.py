"""Unit tests for foundational components of the preprocessing module."""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime
from abc import ABC

from gam.preprocessing.base import Preprocessor
from gam.preprocessing.data_structures import ProcessedGrid
from gam.preprocessing.schema import GridSchema, validate
from gam.preprocessing.units import UnitConverter
from gam.core.exceptions import PreprocessingError


class TestPreprocessor:
    """Test the abstract Preprocessor base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Preprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Preprocessor"):
            Preprocessor()

    def test_subclass_must_implement_process(self):
        """Test that subclasses must implement the process method."""

        class IncompletePreprocessor(Preprocessor):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompletePreprocessor"):
            IncompletePreprocessor()

    def test_subclass_with_process_works(self):
        """Test that a proper subclass can be instantiated."""

        class ConcretePreprocessor(Preprocessor):
            def process(self, data, **kwargs):
                return "processed"

        preprocessor = ConcretePreprocessor()
        assert isinstance(preprocessor, Preprocessor)
        assert preprocessor.process(None) == "processed"


class TestProcessedGrid:
    """Test the ProcessedGrid class."""

    @pytest.fixture
    def valid_data_dict(self):
        """Valid data dictionary for ProcessedGrid."""
        return {
            'data': np.random.rand(5, 5).astype(np.float32),
            'lat': np.linspace(30, 31, 5),
            'lon': np.linspace(0, 1, 5),
            'units': 'mGal',
            'grid_resolution': 0.1,
        }

    @pytest.fixture
    def valid_xr_dataset(self, valid_data_dict):
        """Valid xarray Dataset."""
        da = xr.DataArray(
            valid_data_dict['data'],
            dims=['lat', 'lon'],
            coords={'lat': valid_data_dict['lat'], 'lon': valid_data_dict['lon']}
        )
        ds = xr.Dataset({'data': da})
        ds.attrs.update({
            'units': 'mGal',
            'grid_resolution': 0.1,
            'processed_at': datetime.now(),
            'coordinate_system': 'EPSG:4326',
            'processing_params': {}
        })
        return ds

    @pytest.mark.parametrize("input_data", [
        "valid_data_dict",
        "valid_xr_dataset"
    ])
    def test_instantiation_valid(self, request, input_data):
        """Test instantiation with valid data."""
        data = request.getfixturevalue(input_data)
        grid = ProcessedGrid(data)
        assert isinstance(grid, ProcessedGrid)
        assert 'data' in grid.ds
        assert grid.units == 'mGal'
        assert grid.coordinate_system == 'EPSG:4326'
        grid.validate()  # Should not raise

    def test_instantiation_missing_data_key(self):
        """Test instantiation fails with missing 'data' key."""
        with pytest.raises(ValueError, match="Dict must contain 'data' key"):
            ProcessedGrid({'lat': [1, 2], 'lon': [3, 4]})

    @pytest.mark.parametrize("missing_dim", ['lat', 'lon'])
    def test_validation_missing_required_dims(self, valid_data_dict, missing_dim):
        """Test validation fails with missing required dimensions."""
        data = valid_data_dict.copy()
        del data[missing_dim]
        grid = ProcessedGrid(data)
        with pytest.raises(PreprocessingError, match=f"Missing required dimensions.*{missing_dim}"):
            grid.validate()

    def test_validation_non_monotonic_lat(self, valid_data_dict):
        """Test validation fails with non-monotonic lat coordinates."""
        data = valid_data_dict.copy()
        data['lat'] = [31, 30, 29, 28, 27]  # Decreasing
        grid = ProcessedGrid(data)
        with pytest.raises(PreprocessingError, match="must be strictly increasing"):
            grid.validate()

    def test_validation_invalid_crs(self, valid_data_dict):
        """Test validation fails with invalid CRS."""
        data = valid_data_dict.copy()
        data['coordinate_system'] = 'invalid_crs'
        grid = ProcessedGrid(data)
        with pytest.raises(PreprocessingError, match="Invalid CRS"):
            grid.validate()

    def test_validation_data_not_float(self, valid_data_dict):
        """Test validation fails if data is not floating-point."""
        data = valid_data_dict.copy()
        data['data'] = np.random.randint(0, 10, (5, 5))  # Integer
        grid = ProcessedGrid(data)
        with pytest.raises(PreprocessingError, match="must be floating-point type"):
            grid.validate()

    def test_properties(self, valid_data_dict):
        """Test property getters and setters."""
        grid = ProcessedGrid(valid_data_dict)
        assert grid.units == 'mGal'
        grid.units = 'm/s²'
        assert grid.units == 'm/s²'

        assert grid.coordinate_system == 'EPSG:4326'
        grid.coordinate_system = 'EPSG:3857'
        assert grid.coordinate_system == 'EPSG:3857'

    def test_convert_units(self, valid_data_dict):
        """Test unit conversion."""
        grid = ProcessedGrid(valid_data_dict)
        original_data = grid.ds['data'].values.copy()
        new_grid = grid.convert_units('m/s²', 1e-5)
        np.testing.assert_allclose(new_grid.ds['data'].values, original_data * 1e-5)
        assert new_grid.units == 'm/s²'

    def test_convert_units_zero_factor(self, valid_data_dict):
        """Test unit conversion with zero factor raises error."""
        grid = ProcessedGrid(valid_data_dict)
        with pytest.raises(ValueError, match="Conversion factor cannot be zero"):
            grid.convert_units('m/s²', 0)

    def test_add_metadata(self, valid_data_dict):
        """Test adding metadata."""
        grid = ProcessedGrid(valid_data_dict)
        grid.add_metadata('test_key', 'test_value')
        assert grid.ds.attrs['test_key'] == 'test_value'

    def test_copy(self, valid_data_dict):
        """Test deep copy."""
        grid = ProcessedGrid(valid_data_dict)
        copied = grid.copy()
        assert isinstance(copied, ProcessedGrid)
        assert copied is not grid
        assert copied.ds is not grid.ds

    def test_repr(self, valid_data_dict):
        """Test string representation."""
        grid = ProcessedGrid(valid_data_dict)
        repr_str = repr(grid)
        assert 'ProcessedGrid' in repr_str
        assert 'shape' in repr_str
        assert 'units' in repr_str


class TestGridSchema:
    """Test the GridSchema dataclass and validate function."""

    @pytest.fixture
    def valid_dataset(self):
        """Valid xarray Dataset for schema validation."""
        lat = np.linspace(30, 31, 5)
        lon = np.linspace(0, 1, 5)
        depth = np.linspace(0, 100, 5)
        data = np.random.rand(5, 5, 5).astype(np.float32)
        ds = xr.Dataset(
            {
                'data': (['lat', 'lon', 'depth'], data),
                'lat': lat,
                'lon': lon,
                'depth': depth
            },
            attrs={'obs_elev_m': 0.0, 'B_T': 5e-5}
        )
        return ds

    def test_grid_schema_frozen(self):
        """Test GridSchema is frozen."""
        schema = GridSchema()
        with pytest.raises(AttributeError):
            schema.required_vars = ('new',)

    @pytest.mark.parametrize("missing_var", GridSchema.required_vars)
    def test_validate_missing_var(self, valid_dataset, missing_var):
        """Test validate fails with missing required variable."""
        ds = valid_dataset.copy()
        if missing_var in ds:
            del ds[missing_var]
        with pytest.raises(AssertionError, match=f"missing var {missing_var}"):
            validate(ds)

    def test_validate_wrong_ndim(self, valid_dataset):
        """Test validate fails with wrong dimension ndim."""
        ds = valid_dataset.copy()
        # Make lat 2D
        ds = ds.assign_coords(lat=(['lat', 'lon'], np.random.rand(5, 5)))
        with pytest.raises(AssertionError):
            validate(ds)

    def test_validate_sets_defaults(self, valid_dataset):
        """Test validate sets default attributes."""
        ds = valid_dataset.copy()
        # Remove some attrs
        if 'B_inc_deg' in ds.attrs:
            del ds.attrs['B_inc_deg']
        validated = validate(ds)
        assert 'B_inc_deg' in validated.attrs
        assert validated.attrs['B_inc_deg'] == 60.0

    def test_validate_valid(self, valid_dataset):
        """Test validate passes with valid dataset."""
        validated = validate(valid_dataset)
        assert isinstance(validated, xr.Dataset)
        for var in GridSchema.required_vars:
            assert var in validated


class TestUnitConverter:
    """Test the UnitConverter class."""

    @pytest.fixture
    def converter(self):
        """UnitConverter instance."""
        return UnitConverter()

    @pytest.mark.parametrize("modality,unit,expected", [
        ('gravity', 'mGal', True),
        ('gravity', 'invalid', False),
        ('magnetic', 'nT', True),
        ('magnetic', 'invalid', False),
        ('seismic', 'm/s', True),
        ('seismic', 'invalid', False),
    ])
    def test_validate_unit(self, converter, modality, unit, expected):
        """Test unit validation."""
        if expected:
            assert converter.validate_unit(modality, unit) == expected
        else:
            assert converter.validate_unit(modality, unit) == expected

    def test_validate_unit_unknown_modality(self, converter):
        """Test validate_unit with unknown modality."""
        with pytest.raises(PreprocessingError, match="Unknown modality"):
            converter.validate_unit('unknown', 'unit')

    @pytest.mark.parametrize("modality,from_unit,to_unit,expected_factor", [
        ('gravity', 'mGal', 'm/s²', 1e-5),
        ('gravity', 'μGal', 'm/s²', 1e-8),
        ('gravity', 'mGal', 'μGal', 1e3),
        ('magnetic', 'μT', 'nT', 1e3),
        ('magnetic', 'gamma', 'nT', 1.0),
        ('seismic', 'nm/s', 'm/s', 1e-9),
        ('seismic', 'counts', 'm/s', 1e-9),
    ])
    def test_get_factor(self, converter, modality, from_unit, to_unit, expected_factor):
        """Test getting conversion factors."""
        factor = converter.get_factor(modality, from_unit, to_unit)
        np.testing.assert_allclose(factor, expected_factor)

    def test_get_factor_invalid_unit(self, converter):
        """Test get_factor with invalid unit."""
        with pytest.raises(PreprocessingError):
            converter.get_factor('gravity', 'invalid', 'm/s²')

    @pytest.mark.parametrize("values,from_unit,to_unit,modality,expected", [
        (100.0, 'mGal', 'm/s²', 'gravity', 0.001),
        (np.array([100.0, 200.0]), 'mGal', 'm/s²', 'gravity', np.array([0.001, 0.002])),
        (1.0, 'μT', 'nT', 'magnetic', 1000.0),
        (0.0, 'mGal', 'm/s²', 'gravity', 0.0),
        (-100.0, 'mGal', 'm/s²', 'gravity', -0.001),
    ])
    def test_convert(self, converter, values, from_unit, to_unit, modality, expected):
        """Test value conversion."""
        result = converter.convert(values, from_unit, to_unit, modality)
        np.testing.assert_allclose(result, expected)

    def test_convert_invalid_input(self, converter):
        """Test convert with invalid input type."""
        with pytest.raises(PreprocessingError, match="Values must be numeric"):
            converter.convert("string", 'mGal', 'm/s²', 'gravity')

    def test_convert_grid(self, converter, valid_data_dict):
        """Test grid conversion."""
        grid = ProcessedGrid(valid_data_dict)
        grid.units = 'mGal'
        new_grid = converter.convert_grid(grid, 'm/s²', 'gravity')
        original_data = grid.ds['data'].values
        np.testing.assert_allclose(new_grid.ds['data'].values, original_data * 1e-5)
        assert new_grid.units == 'm/s²'

    def test_convert_grid_invalid_unit(self, converter, valid_data_dict):
        """Test convert_grid with invalid unit."""
        grid = ProcessedGrid(valid_data_dict)
        grid.units = 'invalid'
        with pytest.raises(PreprocessingError):
            converter.convert_grid(grid, 'm/s²', 'gravity')