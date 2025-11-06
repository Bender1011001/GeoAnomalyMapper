"""
Configuration management for GeoAnomalyMapper.
Provides a ConfigManager class that handles loading from environment variables and .env files,
with support for the GAM_USE_V2_CONFIG feature flag for backward compatibility.
When the flag is disabled, falls back to hardcoded defaults matching existing project patterns.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    def load_dotenv(dotenv_path):  # Fallback no-op
        pass


class ConfigManager:
    """
    Singleton configuration manager for the project.
    Loads configuration from environment variables, with fallback to .env file if python-dotenv is installed.
    Supports feature flag GAM_USE_V2_CONFIG to enable advanced config (true) or hardcoded defaults (false/unset).
    
    Key configuration values include:
    - DATA_DIR: Base data directory (default: 'data')
    - OUTPUT_DIR: Output directory (default: 'data/outputs')
    - CACHE_DIR: Cache directory (default: 'data/cache')
    - PROCESSED_DIR: Processed data directory (default: 'data/processed')
    
    Basic validation ensures required paths exist or can be created.
    Designed for cross-platform compatibility and minimal dependencies.
    """
    
    _instance = None
    _use_v2_config: bool = False
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        # Load .env if available
        env_path = Path('.env')
        if HAS_DOTENV and env_path.exists():
            load_dotenv(str(env_path))
        
        # Check feature flag
        self._use_v2_config = os.getenv('GAM_USE_V2_CONFIG', 'false').lower() == 'true'
        
        # Defaults differ for v1 (legacy) vs v2 (structured)
        if self._use_v2_config:
            # v2: OUTPUT_DIR/PROCESSED_DIR/CACHE_DIR are relative to DATA_DIR
            self._defaults = {
                'DATA_DIR': 'data',
                'OUTPUT_DIR': 'outputs',
                'PROCESSED_DIR': 'processed',
                'CACHE_DIR': 'cache',
                'GAM_USE_V2_CONFIG': 'true',
            }
        else:
            # v1: preserve fully-qualified defaults for backward compatibility
            self._defaults = {
                'DATA_DIR': 'data',
                'OUTPUT_DIR': 'data/outputs',
                'PROCESSED_DIR': 'data/processed',
                'CACHE_DIR': 'data/cache',
                'GAM_USE_V2_CONFIG': 'false',
            }
        
        if not self._use_v2_config:
            # Fallback to hardcoded defaults when v2 config disabled (strict legacy behavior)
            self._config = {**self._defaults}
        else:
            # v2: Load from env vars, fallback to v2 defaults
            self._config = {}
            for key, default_value in self._defaults.items():
                env_value = os.getenv(key)
                self._config[key] = env_value if env_value is not None else default_value
        
        # Validate paths
        self._validate_and_create_paths()
        
        self._initialized = True
    
    def _validate_and_create_paths(self) -> None:
        """Ensure configuration paths exist, creating directories if necessary."""
        path_keys = ['DATA_DIR', 'OUTPUT_DIR', 'PROCESSED_DIR', 'CACHE_DIR']
        for key in path_keys:
            if key in self._config:
                path = Path(self._config[key])
                path.mkdir(parents=True, exist_ok=True)
    
    @property
    def use_v2_config(self) -> bool:
        """Whether v2 configuration system is enabled."""
        return self._use_v2_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value (runtime update)."""
        self._config[key] = value
        if key in ['DATA_DIR', 'OUTPUT_DIR', 'PROCESSED_DIR', 'CACHE_DIR']:
            self._validate_and_create_paths()
    
    def keys(self) -> list[str]:
        """List all configuration keys."""
        return list(self._config.keys())
    
    def items(self) -> list[tuple[str, Any]]:
        """List all configuration key-value pairs."""
        return list(self._config.items())
    
    @property
    def config_dict(self) -> Dict[str, Any]:
        """Full configuration as a dictionary (read-only copy)."""
        return self._config.copy()


# Global instance for convenient access
config = ConfigManager()


if __name__ == "__main__":
    # Simple test/example
    print(f"Using v2 config: {config.use_v2_config}")
    print(f"Data dir: {config.get('DATA_DIR')}")
    print("Config keys:", config.keys())