"""
Compatibility shim for path access in GeoAnomalyMapper.
Provides a simple, backward-compatible interface for getting common paths.
When GAM_USE_V2_CONFIG is disabled (default), falls back to hardcoded paths or os.path,
mimicking current ad-hoc path construction in existing scripts (e.g., 'data/outputs').
When enabled, delegates to the full PathManager for structured, cross-platform resolution.

This allows gradual adoption: existing code can import from here without changes,
returning Path objects for consistency while preserving old behavior.
"""

import os
from pathlib import Path
from typing import Optional

# Try to import v2 paths, fallback if not available
try:
    from .paths import paths as _v2_paths
    HAS_V2_PATHS = True
except ImportError:
    HAS_V2_PATHS = False
    _v2_paths = None

def get_path(key: str, default: Optional[str] = None) -> Path:
    """
    Get a path by key, with backward compatibility.
    
    Args:
        key: Path key (e.g., 'data_dir', 'output_dir')
        default: Default path string if not found
    
    Returns:
        Path object for the requested path.
    
    Behavior:
    - If GAM_USE_V2_CONFIG=true and v2 paths available: Uses PathManager
    - Else: Constructs hardcoded Path based on project structure or default
    """
    if HAS_V2_PATHS and _v2_paths and _v2_paths.use_v2_config:
        return _v2_paths.get_path(key, Path(default) if default else None)
    else:
        # Backward compatible: hardcoded paths matching existing structure
        hardcoded_paths = {
            'data_dir': Path('data'),
            'output_dir': Path('data') / 'outputs',
            'processed_dir': Path('data') / 'processed',
            'cache_dir': Path('data') / 'cache',
        }
        if key in hardcoded_paths:
            return hardcoded_paths[key]
        elif default:
            return Path(default)
        else:
            raise ValueError(f"Unknown path key '{key}' and no default provided")

def join_paths(base_key: str, *subpaths: str) -> Path:
    """
    Join subpaths to a base path key, with fallback.
    
    In fallback mode, uses os.path.join for compatibility with old code.
    """
    if HAS_V2_PATHS and _v2_paths and _v2_paths.use_v2_config:
        return _v2_paths.join(base_key, *subpaths)
    else:
        # Fallback to os.path.join for backward compatibility
        base = str(get_path(base_key))
        full_path = os.path.join(base, *subpaths)
        return Path(full_path)

def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path from project root.
    
    In fallback mode, uses Path.resolve() directly.
    """
    if HAS_V2_PATHS and _v2_paths and _v2_paths.use_v2_config:
        return _v2_paths.resolve(relative_path)
    else:
        return Path(relative_path).resolve()

def is_v2_paths_enabled() -> bool:
    """Check if v2 path system is enabled."""
    if HAS_V2_PATHS and _v2_paths:
        return _v2_paths.use_v2_config
    return os.getenv('GAM_USE_V2_CONFIG', 'false').lower() == 'true'

# Common path getters for convenience (matching project needs)
def get_data_dir() -> Path:
    """Get base data directory."""
    return get_path('data_dir')

def get_output_dir() -> Path:
    """Get output directory."""
    return get_path('output_dir')

def get_processed_dir() -> Path:
    """Get processed data directory."""
    return get_path('processed_dir')

def get_cache_dir() -> Path:
    """Get cache directory."""
    return get_path('cache_dir')


if __name__ == "__main__":
    # Test/example
    print(f"V2 paths enabled: {is_v2_paths_enabled()}")
    print(f"Data dir: {get_data_dir()}")
    print(f"Output dir: {get_output_dir()}")
    # Test joining
    print(f"Sample joined path: {join_paths('output_dir', 'reports', 'test.txt')}")
    # Test resolve
    print(f"Resolved sample: {resolve_path('data/test.file')}")