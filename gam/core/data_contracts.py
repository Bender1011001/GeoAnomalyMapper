"""
Pydantic data contracts for Phase 1 scaffolding.

Keep this module import-safe (no I/O, no heavy computation).
"""
from typing import Any, Dict, Tuple
from pydantic import BaseModel, validator, root_validator
import xarray as xr
import math


class RawData(BaseModel):
    """
    Minimal raw data container produced by ingestion components.

    Fields:
      - source: str
      - bbox: tuple[min_lon, min_lat, max_lon, max_lat]
      - data: dict[str, Any]
      - metadata: dict[str, Any]
    """
    source: str
    bbox: Tuple[float, float, float, float]
    data: Dict[str, Any]
    metadata: Dict[str, Any]

    @validator("source")
    def source_must_not_be_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source must be a non-empty string")
        return v

    @validator("bbox")
    def bbox_must_be_valid(cls, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if not (isinstance(v, tuple) and len(v) == 4):
            raise ValueError("bbox must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)")
        min_lon, min_lat, max_lon, max_lat = v
        for coord in (min_lon, min_lat, max_lon, max_lat):
            if not (isinstance(coord, (int, float)) and math.isfinite(coord)):
                raise ValueError("bbox coordinates must be finite numbers")
        if not (min_lon < max_lon):
            raise ValueError("bbox invalid: min_lon must be < max_lon")
        if not (min_lat < max_lat):
            raise ValueError("bbox invalid: min_lat must be < max_lat")
        return v


class ProcessedGrid(BaseModel):
    """
    Wrapper around an xarray.Dataset produced by preprocessing.

    Fields:
      - grid: xr.Dataset

    Validation:
      - grid is an xr.Dataset and contains 'lat' and 'lon' coordinates.
    """
    grid: xr.Dataset

    class Config:
        arbitrary_types_allowed = True

    @validator("grid")
    def grid_must_be_xr_dataset_with_latlon(cls, v: xr.Dataset) -> xr.Dataset:
        if not isinstance(v, xr.Dataset):
            raise ValueError("grid must be an xarray.Dataset")
        coords = getattr(v, "coords", {})
        if "lat" not in coords or "lon" not in coords:
            raise ValueError("grid must contain 'lat' and 'lon' coordinates")
        return v


class InversionResult(BaseModel):
    """
    Result of an inversion/modeling step.

    Fields:
      - model: xr.Dataset
      - uncertainty: xr.Dataset
      - metadata: dict[str, Any]

    Validation:
      - model and uncertainty are xr.Dataset with matching dims and shapes.
    """
    model: xr.Dataset
    uncertainty: xr.Dataset
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def model_and_uncertainty_must_match(cls, values):
        model = values.get("model")
        uncertainty = values.get("uncertainty")
        if not isinstance(model, xr.Dataset):
            raise ValueError("model must be an xarray.Dataset")
        if not isinstance(uncertainty, xr.Dataset):
            raise ValueError("uncertainty must be an xarray.Dataset")

        # Compare dimension names and sizes
        model_dims = list(model.dims)
        uncert_dims = list(uncertainty.dims)
        if model_dims != uncert_dims:
            raise ValueError(f"model and uncertainty must have the same dimension names; got {model_dims} vs {uncert_dims}")

        for dim in model_dims:
            if model.sizes.get(dim) != uncertainty.sizes.get(dim):
                raise ValueError(f"dimension '{dim}' has mismatched sizes: {model.sizes.get(dim)} vs {uncertainty.sizes.get(dim)}")
        return values


class Anomaly(BaseModel):
    """
    Simple anomaly descriptor.

    Fields:
      - latitude: float (-90..90)
      - longitude: float (-180..180)
      - depth_meters: float (>= 0)
      - confidence: float (0..1)
      - anomaly_type: non-empty str
    """
    latitude: float
    longitude: float
    depth_meters: float
    confidence: float
    anomaly_type: str

    @validator("latitude")
    def latitude_range(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError("latitude must be between -90 and 90")
        return v

    @validator("longitude")
    def longitude_range(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @validator("depth_meters")
    def depth_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("depth_meters must be >= 0")
        return v

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        return v

    @validator("anomaly_type")
    def anomaly_type_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("anomaly_type must be a non-empty string")
        return v