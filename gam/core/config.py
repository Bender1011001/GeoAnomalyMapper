import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Union

import yaml
from pydantic import BaseModel, validator, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

log = logging.getLogger(__name__)

class ModalityConfig(BaseModel):
    """Configuration for a single data modality."""
    enabled: bool = True
    resolution: float = 0.01  # degrees
    sources: List[str] = Field(default_factory=list)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    modeling: Dict[str, Any] = Field(default_factory=dict)

@pydantic_dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    backend: str = 'local'
    n_workers: int = 4
    memory_limit: Optional[str] = None  # e.g., '2GB'

@pydantic_dataclass
class GAMConfig:
    """
    Main configuration for GeoAnomalyMapper pipeline.
    
    Loads from config.yaml with defaults for modalities, processing params, and paths.
    Validates required fields and types for safe pipeline execution.
    """
    # Pipeline settings
    modalities: Dict[str, ModalityConfig] = Field(default_factory=lambda: {
        'gravity': ModalityConfig(sources=['USGS']),
        'magnetic': ModalityConfig(sources=['USGS']),
        'seismic': ModalityConfig(sources=['IRIS']),
        'insar': ModalityConfig(sources=['ESA'])
    })
    fusion_method: str = 'joint_inversion'
    resolution: float = 0.01
    bbox: Optional[List[float]] = None
    output_dir: str = './results'
    cache_dir: str = './cache'
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)

    # Data sources
    data_sources_path: str = 'data_sources.yaml'
    api_keys: Dict[str, str] = Field(default_factory=dict)

    # Logging and monitoring
    log_level: str = 'INFO'
    monitoring_enabled: bool = False

    @validator('modalities', pre=True)
    def validate_modalities(cls, v):
        if not isinstance(v, dict):
            raise ValueError('modalities must be a dict')
        return v

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'GAMConfig':
        """
        Load configuration from YAML file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            log.warning(f"Config file {config_path} not found. Using defaults.")
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        import os
        data['api_keys'] = data.get('api_keys', {})
        for key, env_var in [('usgs_api', 'USGS_API_KEY'), ('iris_api', 'IRIS_API_KEY'), ('esa_api', 'ESA_API_KEY')]:
            if env_var in os.environ:
                data['api_keys'][key] = os.environ[env_var]

        config = cls(**data)
        log.info(f"Configuration loaded from {config_path}")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return self.dict()

    def save(self, path: Union[str, Path]):
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        log.info(f"Configuration saved to {path}")

# --- Compatibility shim for legacy imports expecting `config_manager` ---

from threading import RLock

class _ConfigManager:
    """
    Backward-compatible config manager to satisfy `from gam.core.config import config_manager`.
    Provides a simple process-level singleton of GAMConfig accessed via get_config().
    """

    def __init__(self):
        self._config: GAMConfig | None = None
        self._lock = RLock()

    def get_config(self) -> GAMConfig:
        with self._lock:
            if self._config is None:
                # Lazily instantiate a default config
                self._config = GAMConfig()
            return self._config

    def set_config(self, config: GAMConfig) -> None:
        with self._lock:
            self._config = config

# Public singleton instance expected by legacy modules
config_manager = _ConfigManager()

# Convenience functions to align with legacy usage patterns
def get_config() -> GAMConfig:
    return config_manager.get_config()
# --- Extend compatibility shim: add current_config property expected by legacy code ---

def _cm_get(self) -> GAMConfig:
    return self.get_config()

def _cm_set(self, cfg: GAMConfig) -> None:
    self.set_config(cfg)

# Provide attribute-style access: config_manager.current_config
_ConfigManager.current_config = property(_cm_get, _cm_set)