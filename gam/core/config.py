"""
Core Configuration Module.

Handles loading and validation of GAM configuration from YAML files.
Provides GAMConfig dataclass for pipeline settings, data sources, and processing parameters.
Supports environment variable overrides for sensitive data (API keys).
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Union

import yaml
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

log = logging.getLogger(__name__)

class ModalityConfig(BaseModel):
    """Configuration for a single data modality."""
    enabled: bool = True
    resolution: float = 0.01  # degrees
    sources: List[str] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    modeling: Dict[str, Any] = field(default_factory=dict)

@pydantic_dataclass
class GAMConfig:
    """
    Main configuration for GeoAnomalyMapper pipeline.

    Loads from config.yaml with defaults for modalities, processing params, and paths.
    Validates required fields and types for safe pipeline execution.
    """
    # Pipeline settings
    modalities: Dict[str, ModalityConfig] = field(default_factory=lambda: {
        'gravity': ModalityConfig(sources=['USGS']),
        'magnetic': ModalityConfig(sources=['USGS']),
        'seismic': ModalityConfig(sources=['IRIS']),
        'insar': ModalityConfig(sources=['ESA'])
    })
    fusion_method: str = 'joint_inversion'  # or 'ml_fusion'
    resolution: float = 0.01  # global grid resolution in degrees
    bbox: Optional[List[float]] = None  # [min_lat, max_lat, min_lon, max_lon]
    output_dir: str = './results'
    cache_dir: str = './cache'
    use_parallel: bool = True
    n_workers: int = 4

    # Data sources
    data_sources_path: str = 'data_sources.yaml'
    api_keys: Dict[str, str] = field(default_factory=dict)  # Loaded from env

    # Logging and monitoring
    log_level: str = 'INFO'
    monitoring_enabled: bool = False

    @validator('modalities', pre=True)
    def validate_modalities(cls, v):
        if not isinstance(v, dict):
            raise ValueError('modalities must be a dict')
        for key, val in v.items():
            if not isinstance(val, ModalityConfig):
                raise ValueError(f'Modality {key} must be ModalityConfig')
        return v

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'GAMConfig':
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml.

        Returns:
            GAMConfig instance with loaded and validated settings.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            log.warning(f"Config file {config_path} not found. Using defaults.")
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Load API keys from environment if not in config
        import os
        data['api_keys'] = data.get('api_keys', {})
        for key, env_var in [('usgs_api', 'USGS_API_KEY'), ('iris_api', 'IRIS_API_KEY'), ('esa_api', 'ESA_API_KEY')]:
            if env_var in os.environ:
                data['api_keys'][key] = os.environ[env_var]

        # Convert to dataclass
        config_dict = {}
        for field_name, field_info in cls.__pydantic_fields__.items():
            if field_name in data:
                config_dict[field_name] = data[field_name]
            else:
                config_dict[field_name] = field_info.default

        # Handle nested ModalityConfig
        if 'modalities' in data:
            modalities = {}
            for mod_name, mod_data in data['modalities'].items():
                modalities[mod_name] = ModalityConfig(**mod_data)
            config_dict['modalities'] = modalities

        config = cls(**config_dict)
        log.info(f"Configuration loaded from {config_path}")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    def save(self, path: Union[str, Path]):
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        log.info(f"Configuration saved to {path}")