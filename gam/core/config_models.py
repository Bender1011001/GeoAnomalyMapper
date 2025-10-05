"""
Application configuration models (Pydantic v1-style) for Phase 1.

Models are strict (forbid extra fields) and organized into nested components.
Import-safe: no file I/O or side-effects at import time.
"""
from typing import Any, Dict, List
from pydantic import BaseModel


class AppSettings(BaseModel):
    output_dir: str = "./data/outputs"
    cache_dir: str = "./data/cache"
    default_modalities: List[str] = ["gravity", "magnetic"]

    class Config:
        extra = "forbid"


class ProcessingConfig(BaseModel):
    grid_resolution: float = 0.1
    filters: List[str] = []

    class Config:
        extra = "forbid"


class ModelingConfig(BaseModel):
    fusion_method: str = "joint_inversion"
    anomaly_threshold: float = 95.0
    max_iterations: int = 20

    class Config:
        extra = "forbid"


class FeatureFlags(BaseModel):
    enable_cache: bool = True
    enable_parallel: bool = False

    class Config:
        extra = "forbid"


class DataSources(BaseModel):
    """
    Mapping of data source names to provider-specific configuration dictionaries.

    Example:
      data_sources:
        provider_a:
          url: "https://..."
          api_key: "..."
    """
    __root__: Dict[str, Dict[str, Any]] = {}

    class Config:
        extra = "forbid"


class AppConfig(BaseModel):
    app: AppSettings = AppSettings()
    preprocessing: ProcessingConfig = ProcessingConfig()
    modeling: ModelingConfig = ModelingConfig()
    feature_flags: FeatureFlags = FeatureFlags()
    data_sources: DataSources = DataSources()

    class Config:
        extra = "forbid"