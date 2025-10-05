"""
Modeling service interface and a deterministic mock implementation.

Import-safe: no I/O or heavy computation at import time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import xarray as xr

from gam.core.data_contracts import ProcessedGrid, InversionResult


class ModelingServiceInterface(ABC):
    """
    Stateless modeling/inversion interface.

    Implementations must accept a ProcessedGrid and return an InversionResult.
    """

    @abstractmethod
    def invert(self, grid: ProcessedGrid, modality: str) -> InversionResult:
        """
        Perform inversion for the given modality.

        Args:
            grid: ProcessedGrid produced by preprocessing.
            modality: modality name (e.g., "gravity", "magnetic").

        Returns:
            InversionResult containing model and uncertainty xr.Datasets.
        """
        raise NotImplementedError


class MockModelingService(ModelingServiceInterface):
    """
    Deterministic mock inverter that returns the input grid as both model and uncertainty.

    This mock is intended for scaffolding and unit tests. It avoids I/O and heavy computation.
    """

    def invert(self, grid: ProcessedGrid, modality: str) -> InversionResult:
        ds: xr.Dataset = grid.grid
        # Deep copy to ensure model and uncertainty are distinct objects
        uncertainty: xr.Dataset = ds.copy(deep=True)
        metadata: dict[str, Any] = {"inversion": "mock", "modality": modality}
        return InversionResult(model=ds, uncertainty=uncertainty, metadata=metadata)