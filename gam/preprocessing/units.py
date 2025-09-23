"""Unit conversion utilities for the GAM preprocessing module."""

from __future__ import annotations

import logging
import numpy as np
from typing import Union, Dict, Any, Tuple
from numbers import Number

from gam.core.exceptions import PreprocessingError


logger = logging.getLogger(__name__)


class UnitConverter:
    """
    Unit conversion class for geophysical data modalities.

    Supports conversion between common units for gravity, magnetic, and seismic data.
    Uses predefined factors; chains conversions if direct path not available (e.g.,
    mGal -> μGal -> m/s²). Validates units per modality. Integrates with ProcessedGrid
    by providing factors for its convert_units method.

    Supported Modalities and Units:
    - Gravity: 'mGal', 'μGal', 'm/s²' (SI base)
    - Magnetic: 'nT' (nanotesla), 'μT' (microtesla), 'gamma' (1 gamma = 1 nT)
    - Seismic: 'm/s' (velocity), 'nm/s' (displacement velocity), 'counts' (raw, approximate to m/s with factor)

    Parameters
    ----------
    None
        Class-level factors; instantiate for state if needed (stateless).

    Attributes
    ----------
    gravity_factors : Dict[str, Dict[str, float]]
        Conversion factors to base unit (m/s²).
    magnetic_factors : Dict[str, Dict[str, float]]
        To base (T, but using nT as practical).
    seismic_factors : Dict[str, Dict[str, float]]
        To base (m/s).

    Methods
    -------
    validate_unit(modality: str, unit: str) -> bool
        Check if unit valid for modality.
    get_factor(modality: str, from_unit: str, to_unit: str) -> float
        Get conversion factor from -> to.
    convert(values: Union[Number, np.ndarray], from_unit: str, to_unit: str, modality: str) -> Union[Number, np.ndarray]
        Convert values using factor.
    convert_grid(grid: ProcessedGrid, to_unit: str, modality: str) -> ProcessedGrid
        Convenience for ProcessedGrid integration.

    Notes
    -----
    - Factors are multiplicative (new = old * factor).
    - For seismic 'counts', assumes generic factor (e.g., 1 count = 1e-9 m/s); calibrate per instrument.
    - Reproducible and fast (vectorized).
    - Errors raise PreprocessingError for consistency.

    Examples
    --------
    >>> converter = UnitConverter()
    >>> if converter.validate_unit('gravity', 'mGal'):
    ...     factor = converter.get_factor('gravity', 'mGal', 'm/s²')
    ...     converted = converter.convert(100.0, 'mGal', 'm/s²', 'gravity')
    >>> # 100 mGal = 0.001 m/s²
    0.001
    >>> new_grid = converter.convert_grid(processed_grid, 'm/s²', 'gravity')
    """

    # Base units: gravity 'm/s²', magnetic 'nT', seismic 'm/s'
    GRAVITY_UNITS = {'mGal', 'μGal', 'm/s²'}
    MAGNETIC_UNITS = {'nT', 'μT', 'gamma'}
    SEISMIC_UNITS = {'m/s', 'nm/s', 'counts'}

    # Factors to base unit
    GRAVITY_FACTORS = {
        'm/s²': 1.0,
        'mGal': 1e-5,  # 1 mGal = 10^{-5} m/s²
        'μGal': 1e-8   # 1 μGal = 10^{-8} m/s²
    }
    MAGNETIC_FACTORS = {
        'nT': 1.0,     # nanotesla
        'μT': 1e3,     # 1 μT = 1000 nT
        'gamma': 1.0   # 1 gamma = 1 nT
    }
    SEISMIC_FACTORS = {
        'm/s': 1.0,
        'nm/s': 1e-9,  # 1 nm/s = 10^{-9} m/s
        'counts': 1e-9  # Approximate; depends on sensor (e.g., 1 count ~ 1 nm/s for some)
    }

    MODALITY_MAP = {
        'gravity': (GRAVITY_UNITS, GRAVITY_FACTORS),
        'magnetic': (MAGNETIC_UNITS, MAGNETIC_FACTORS),
        'seismic': (SEISMIC_UNITS, SEISMIC_FACTORS)
    }

    def __init__(self):
        self.gravity_factors = self.GRAVITY_FACTORS
        self.magnetic_factors = self.MAGNETIC_FACTORS
        self.seismic_factors = self.SEISMIC_FACTORS

    def validate_unit(self, modality: str, unit: str) -> bool:
        """
        Validate if unit is supported for the modality.

        Parameters
        ----------
        modality : str
            'gravity', 'magnetic', or 'seismic'.
        unit : str
            Unit to check.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        PreprocessingError
            If modality unknown.
        """
        if modality not in self.MODALITY_MAP:
            raise PreprocessingError(f"Unknown modality: {modality}. Supported: {list(self.MODALITY_MAP.keys())}")
        units, _ = self.MODALITY_MAP[modality]
        is_valid = unit in units
        if not is_valid:
            logger.warning(f"Invalid unit '{unit}' for {modality}. Supported: {units}")
        return is_valid

    def get_factor(self, modality: str, from_unit: str, to_unit: str) -> float:
        """
        Get conversion factor from from_unit to to_unit for modality.

        Chains via base if direct not available (e.g., mGal to μGal = (mGal->base) / (μGal->base)).

        Parameters
        ----------
        modality : str
            Data modality.
        from_unit : str
            Source unit.
        to_unit : str
            Target unit.

        Returns
        -------
        float
            Conversion factor (value_to = value_from * factor).

        Raises
        ------
        PreprocessingError
            If units invalid or no conversion path.
        """
        self.validate_unit(modality, from_unit)
        self.validate_unit(modality, to_unit)

        _, factors = self.MODALITY_MAP[modality]
        base_unit = list(factors.keys())[0]  # Assume first is base
        factor_from_base = factors[to_unit]
        factor_to_base = factors[from_unit]
        factor = factor_from_base / factor_to_base

        logger.debug(f"Conversion factor {from_unit} -> {to_unit} for {modality}: {factor}")
        return factor

    def convert(
        self,
        values: Union[Number, np.ndarray],
        from_unit: str,
        to_unit: str,
        modality: str
    ) -> Union[Number, np.ndarray]:
        """
        Convert values from from_unit to to_unit.

        Supports scalars and arrays (broadcasts).

        Parameters
        ----------
        values : Number or np.ndarray
            Value(s) to convert.
        from_unit : str
            Source unit.
        to_unit : str
            Target unit.
        modality : str
            Data modality.

        Returns
        -------
        Union[Number, np.ndarray]
            Converted values.

        Raises
        ------
        PreprocessingError
            If conversion invalid or non-numeric input.
        """
        if not isinstance(values, (Number, np.ndarray)):
            raise PreprocessingError(f"Values must be numeric or ndarray, got {type(values)}")

        factor = self.get_factor(modality, from_unit, to_unit)
        converted = values * factor if isinstance(values, Number) else values * factor

        logger.info(f"Converted {len(values) if hasattr(values, '__len__') else 'scalar'} values from {from_unit} to {to_unit} ({modality}): factor={factor}")
        return converted

    def convert_grid(
        self,
        grid: ProcessedGrid,
        to_unit: str,
        modality: str
    ) -> ProcessedGrid:
        """
        Convert a ProcessedGrid to new units.

        Uses internal factor; updates grid.units and metadata.

        Parameters
        ----------
        grid : ProcessedGrid
            Input grid.
        to_unit : str
            Target unit.
        modality : str
            Modality for validation.

        Returns
        -------
        ProcessedGrid
            New grid with converted data/uncertainty and updated units.

        Raises
        ------
        PreprocessingError
            If grid lacks 'data' or conversion fails.
        """
        self.validate_unit(modality, to_unit)
        from_unit = grid.units
        self.validate_unit(modality, from_unit)

        factor = self.get_factor(modality, from_unit, to_unit)
        # Use grid's convert_units if available; else manual
        if hasattr(grid, 'convert_units'):
            new_grid = grid.convert_units(to_unit, factor)
        else:
            # Fallback: manual conversion
            new_ds = grid.ds.copy()
            new_ds['data'] *= factor
            if 'uncertainty' in new_ds:
                new_ds['uncertainty'] *= abs(factor)
            new_grid = ProcessedGrid(new_ds)
            new_grid.units = to_unit

        new_grid.add_metadata('unit_conversion', {
            'modality': modality,
            'from': from_unit,
            'to': to_unit,
            'factor': factor
        })
        logger.info(f"Grid converted from {from_unit} to {to_unit} ({modality})")
        return new_grid


# Global instance for convenience
converter = UnitConverter()