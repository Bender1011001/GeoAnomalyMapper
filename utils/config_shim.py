"""
Compatibility shim for configuration access in GeoAnomalyMapper.
Provides a simple, backward-compatible interface for getting configuration values.
When GAM_USE_V2_CONFIG is disabled (default), falls back to hardcoded defaults or direct os.getenv calls,
mimicking current ad-hoc usage in existing scripts.
When enabled, delegates to the full ConfigManager for structured loading and validation.

This allows gradual migration: existing code can import from here without changes,
and new code can opt into v2 features via the flag.
"""

import os
from typing import Any, Optional

# Try to import v2 config, fallback if not available
try:
    from .config import config as _v2_config
    HAS_V2_CONFIG = True
except ImportError:
    HAS_V2_CONFIG = False
    _v2_config = None

def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value, with backward compatibility.
    
    Args:
        key: Configuration key (e.g., 'DATA_DIR')
        default: Default value if not found
    
    Returns:
        The configuration value.
    
    Behavior:
    - If GAM_USE_V2_CONFIG=true and v2 config available: Uses ConfigManager
    - Else: Falls back to os.getenv(key) or provided default
    """
    if HAS_V2_CONFIG and _v2_config and _v2_config.use_v2_config:
        return _v2_config.get(key, default)
    else:
        # Backward compatible: direct env var access or default
        return os.getenv(key, default)

def set_config(key: str, value: Any) -> None:
    """
    Set a configuration value at runtime (limited support in fallback mode).
    
    In fallback mode, sets os.environ (affects subprocesses but not current process).
    """
    if HAS_V2_CONFIG and _v2_config and _v2_config.use_v2_config:
        _v2_config.set(key, value)
    else:
        # Fallback: set env var for future gets (note: doesn't affect current process globals)
        os.environ[key] = str(value)

def is_v2_config_enabled() -> bool:
    """Check if v2 configuration is enabled."""
    if HAS_V2_CONFIG and _v2_config:
        return _v2_config.use_v2_config
    return os.getenv('GAM_USE_V2_CONFIG', 'false').lower() == 'true'

# Common config getters for convenience (matching project needs)
def get_data_dir(default: str = 'data') -> str:
    """Get data directory path."""
    return get_config('DATA_DIR', default)

def get_output_dir(default: str = 'data/outputs') -> str:
    """Get output directory path."""
    return get_config('OUTPUT_DIR', default)

def get_processed_dir(default: str = 'data/processed') -> str:
    """Get processed data directory."""
    return get_config('PROCESSED_DIR', default)

def get_cache_dir(default: str = 'data/cache') -> str:
    """Get cache directory."""
    return get_config('CACHE_DIR', default)

if __name__ == "__main__":
    # Test/example
    print(f"V2 config enabled: {is_v2_config_enabled()}")
    print(f"Data dir: {get_data_dir()}")
    set_config('TEST_KEY', 'test_value')
    print(f"Test key: {get_config('TEST_KEY', 'not_set')}")