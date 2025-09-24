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
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

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

        # Handle nested ParallelConfig
        if 'parallel' in data:
            config_dict['parallel'] = ParallelConfig(**data['parallel'])

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


def load_config(config_path: Union[str, Path]) -> GAMConfig:
    """Load and validate configuration from YAML file."""
    return GAMConfig.from_yaml(config_path)


def validate_config(config: GAMConfig) -> bool:
    """Validate the loaded configuration."""
    # Basic validation: check required fields
    if not config.modalities:
        raise ValueError("No modalities configured")
    if config.bbox and len(config.bbox) != 4:
        raise ValueError("Invalid bbox format")
    log.info("Configuration validation passed")
    return True


class ConfigManager:
    """Singleton manager for GAM configuration.

    Holds the current GAMConfig instance and provides access methods.
    Initializes with defaults if no config file provided.
    """
    _instance = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.current_config: Optional[GAMConfig] = None
            self.initialized = True
            self.load_default()

    def load_default(self) -> None:
        """Load default configuration."""
        self.current_config = GAMConfig()

    def load_from_file(self, config_path: Union[str, Path]) -> GAMConfig:
        """Load configuration from file and set as current."""
        self.current_config = GAMConfig.from_yaml(config_path)
        validate_config(self.current_config)
        log.info("Configuration loaded and validated")
        return self.current_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dotted key (e.g., 'parallel.n_workers')."""
        if not self.current_config:
            raise ValueError("No current config loaded")
        keys = key.split('.')
        value = self.current_config
        for k in keys:
            value = getattr(value, k, None)
            if value is None:
                return default
        return value


# Global config manager instance
config_manager = ConfigManager()