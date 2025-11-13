
"""
Path management for GeoAnomalyMapper.
Provides a PathManager class that offers consistent, cross-platform path resolution
based on the ConfigManager. Supports the GAM_USE_V2_CONFIG feature flag for backward compatibility.
When disabled, falls back to hardcoded paths matching existing project structure (e.g., 'data/outputs').
"""

from pathlib import Path
from typing import Optional
from .config import config as _config


class PathManager:
    """
    Singleton path manager for the project.
    Resolves common paths using ConfigManager values, with cross-platform support via pathlib.
    When GAM_USE_V2_CONFIG is disabled, uses hardcoded defaults to maintain backward compatibility.
    
    Common paths provided:
    - data_dir: Base data directory
    - output_dir: Output directory for results and reports
    - processed_dir: Processed data subdirectory
    - cache_dir: Cache for temporary files
    
    All paths are Path objects, ensuring consistent behavior on Windows/Linux/macOS.
    Automatically creates directories if they don't exist.
    """
    
    _instance = None
    
    def __new__(cls) -> 'PathManager':
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._use_v2_config = _config.use_v2_config
        
        # Hardcoded defaults for backward compatibility (matching project structure)
        self._defaults = {
            'data_dir': Path('data'),
            'output_dir': Path('data') / 'outputs',
            'processed_dir': Path('data') / 'processed',
            'cache_dir': Path('data') / 'cache',
        }
        
        if not self._use_v2_config:
            # Fallback to hardcoded defaults
            self._paths = {key: value for key, value in self._defaults.items()}
        else:
            # Use config values to construct paths
            base_data = Path(_config.get('DATA_DIR', 'data'))
            self._paths = {
                'data_dir': base_data,
                'output_dir': base_data / _config.get('OUTPUT_DIR', 'outputs'),
                'processed_dir': base_data / _config.get('PROCESSED_DIR', 'processed'),
                'cache_dir': base_data / _config.get('CACHE_DIR', 'cache'),
            }
        
        # Ensure directories exist
        for path in self._paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
    
    def get_path(self, key: str, default: Optional[Path] = None) -> Path:
        """Retrieve a path by key, falling back to default if not found."""
        return self._paths.get(key, default)
    
    @property
    def data_dir(self) -> Path:
        """Base data directory."""
        return self.get_path('data_dir')
    
    @property
    def output_dir(self) -> Path:
        """Output directory for results and reports."""
        return self.get_path('output_dir')
    
    @property
    def processed_dir(self) -> Path:
        """Processed data subdirectory."""
        return self.get_path('processed_dir')
    
    @property
    def cache_dir(self) -> Path:
        """Cache directory for temporary files."""
        return self.get_path('cache_dir')
    
    
    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path from the project root."""
        return Path(relative_path).resolve()
    
    def join(self, base_key: str, *subpaths: str) -> Path:
        """Join subpaths to a base path key."""
        base = self.get_path(base_key)
        if base is None:
            raise ValueError(f"Base path '{base_key}' not found")
        return base.joinpath(*subpaths)
    
    @property
    def use_v2_config(self) -> bool:
        """Whether v2 path system is enabled."""
        return self._use_v2_config
    
    def keys(self) -> list[str]:
        """List all path keys."""
        return list(self._paths.keys())
    
    def items(self) -> list[tuple[str, Path]]:
        """List all path key-value pairs."""
        return list(self._paths.items())


# Global instance for convenient access
paths = PathManager()


if __name__ == "__main__":
    # Simple test/example
    print(f"Using v2 paths: {paths.use_v2_config}")
    print(f"Data dir: {paths.data_dir}")
    print(f"Output dir: {paths.output_dir}")
    print("Path keys:", paths.keys())
    # Test joining
    print(f"Sample joined path: {paths.join('output_dir', 'reports', 'test.txt')}")