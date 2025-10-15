# Configuration System Guide

**Unified and Flexible Configuration for GeoAnomalyMapper**

The scientific code review identified hardcoded paths and scattered settings as major issues. The new configuration system centralizes all parameters in `config/config.json` with support for environment variables (`.env`) and runtime overrides. This enables cross-platform compatibility, customization without code changes, and production deployment.

## Overview

### Key Principles
- **Centralized**: Single `config.json` for paths, data sources, fusion params, robustness settings.
- **Layered Overrides**: JSON defaults → `.env` vars → CLI flags → runtime.
- **Validation**: Automatic schema checking on load; reports errors.
- **Security**: Credentials in `.env` (gitignored); no secrets in JSON.
- **Cross-Platform**: Pathlib integration via `utils/paths.py` resolves OS-specific paths.

### File Locations
- **Core Config**: `config/config.json` - Project settings.
- **Environment**: `.env` - Secrets and overrides (e.g., credentials, API keys).
- **Validation Data**: `config/known_features.json` - For scientific benchmarking.
- **Example Templates**: `config/config.json.example`, `.env.example` (git-committed).

**Setup**:
```bash
cd GeoAnomalyMapper
cp config/config.json.example config/config.json
cp .env.example .env
# Edit .env for credentials
# Edit config.json for custom paths/settings
```

## Structure of config.json

The JSON uses a modular schema with validation. Example:

```json
{
  "project": {
    "name": "GeoAnomalyMapper",
    "version": "2.0.0",
    "data_root": "./data",
    "processed_dir": "processed",
    "outputs_dir": "outputs"
  },
  "paths": {
    "raw_data": "${data_root}/raw",
    "insar_dir": "${raw_data}/insar",
    "gravity_dir": "${raw_data}/gravity",
    "enable_symlinks": false
  },
  "data_sources": {
    "gravity": {
      "enabled": true,
      "preferred_model": "xgm2019e",
      "resolution": 0.025,
      "height_km": 0
    },
    "magnetic": {
      "enabled": true,
      "source": "emag2v3"
    },
    "insar": {
      "enabled": true,
      "sources": ["copernicus", "egms"],
      "max_baseline": 150,
      "auto_process": true
    },
    "elevation": {
      "enabled": true,
      "source": "nasadem"
    },
    "lithology": {
      "enabled": true,
      "path": "${raw_data}/LiMW_GIS 2015.gdb"
    }
  },
  "fusion": {
    "dynamic_weighting": true,
    "target_resolution": 0.001,
    "spectral_cutoff": 10,
    "uncertainty_threshold": 0.1
  },
  "robustness": {
    "max_retries": 5,
    "base_delay": 1.0,
    "backoff_factor": 2.0,
    "jitter": true,
    "circuit_threshold": 5,
    "recovery_timeout": 60,
    "timeout_connect": 10,
    "timeout_read": 30,
    "bandwidth_throttle": null,
    "services": {
      "copernicus": {
        "auth_url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        "hosts": ["identity.dataspace.copernicus.eu", "catalogue.dataspace.copernicus.eu"]
      },
      "earthdata": {
        "auth_url": "https://urs.earthdata.nasa.gov/oauth/token",
        "hosts": ["urs.earthdata.nasa.gov", "e4ftl01.cr.usgs.gov"]
      }
    }
  },
  "validation": {
    "enabled": true,
    "known_features_path": "config/known_features.json",
    "thresholds": {
      "true_positive": 0.7,
      "false_positive": 0.3
    }
  },
  "logging": {
    "level": "INFO",
    "format": "structured",
    "file": "${outputs_dir}/logs/app.log"
  }
}
```

### Variable Substitution
- `${data_root}`: Expands to project paths (via `utils/paths.py`).
- Environment vars: `${ENV_VAR}` pulls from `.env` or system.
- Cross-Platform: Automatically uses `/` on Unix, `\` on Windows.

## Environment Variables (.env)

For secrets and overrides (never commit!):

```
# Credentials
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# Overrides
DATA_ROOT=/custom/data/path
MAX_RETRIES=10
DYNAMIC_WEIGHTING=false

# SNAP Path (if not auto-detected)
SNAP_PATH=/Applications/snap/bin/gpt
```

**Loading Order**:
1. Load `config.json`.
2. Override with `.env` (e.g., `MAX_RETRIES` sets `"robustness.max_retries"`).
3. CLI flags (e.g., `--data-root /path`).
4. Runtime (e.g., in scripts).

## Usage in Code

All modules load config via a central manager:

```python
from utils.config import ConfigManager

# Initialize (loads JSON + .env)
config = ConfigManager()

# Access
data_root = config.get('project.data_root')  # './data'
insar_enabled = config.get('data_sources.insar.enabled', default=True)

# Paths (resolved)
raw_dir = config.get_path('paths.raw_data')  # Path('./data/raw')

# Validation
if not config.validate():
    raise ValueError("Invalid config")

# Overrides
config.set('fusion.target_resolution', 0.0005)  # Runtime change
```

**In data_agent.py**:
- Uses config for sources, bbox defaults, robustness params.
- Validates services before download.

**In multi_resolution_fusion.py**:
- Loads fusion/weights; applies dynamic if enabled.

## Validation and Schema

Config includes built-in schema validation (using pydantic or jsonschema):

```bash
# Validate config
python -c "from utils.config import ConfigManager; ConfigManager().validate()"

# Report issues
python setup_environment.py config-report
```

**Common Errors**:
- Missing required keys (e.g., no `data_root`).
- Invalid types (e.g., string for numeric `max_retries`).
- Unresolved paths (e.g., invalid `${VAR}`).

**Fixes**:
- Use example as template.
- Check logs for specific errors.
- Test with minimal config for troubleshooting.

## Customization Examples

### 1. Custom Data Paths (Cross-Platform)
```json
{
  "project": {
    "data_root": "/shared/geodata"  // Unix
  }
}
```
- On Windows: Auto-converts to `C:\shared\geodata`.
- Symlinks: Set `"enable_symlinks": true` for large datasets.

### 2. Disable InSAR for Testing
```json
{
  "data_sources": {
    "insar": { "enabled": false }
  }
}
```
- Or via .env: `INSAR_ENABLED=false`.

### 3. Tune Robustness for Slow Networks
```json
{
  "robustness": {
    "max_retries": 10,
    "base_delay": 2.0,
    "bandwidth_throttle": 1024000  // 1MB/s
  }
}
```

### 4. Advanced Fusion
```json
{
  "fusion": {
    "dynamic_weighting": true,
    "custom_weights": {
      "insar": 0.9,
      "gravity": 0.7
    },
    "spectral_bands": [0, 10, 50]  // Low/mid/high freq
  }
}
```

### 5. Production Logging
```json
{
  "logging": {
    "level": "DEBUG",
    "handlers": ["file", "console"],
    "file": "/var/log/geoanomaly.log"
  }
}
```

## Migration from Legacy

- **Hardcoded Paths**: Replace with config keys (e.g., `data/raw/` → `config.get_path('paths.raw_data')`).
- **Scattered Settings**: Consolidate into JSON sections.
- **Credentials**: Move from scripts to `.env`.
- **Validation**: Add `ConfigManager.validate()` to entrypoints.

**Script Update Example**:
```python
# Old: hardcoded
RAW_DIR = './data/raw'

# New:
from utils.config import ConfigManager
config = ConfigManager()
RAW_DIR = config.get_path('paths.raw_data')
```

## Best Practices

- **Version Control**: Commit `config.json.example`; ignore `.env` and `config.json` if sensitive.
- **Environments**: Use separate configs (dev/prod) via `--config-file`.
- **Documentation**: Inline comments in JSON for complex params.
- **Testing**: `python -m unittest` includes config validation tests.
- **Security**: Rotate credentials quarterly; use vault for production.

For troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

*Updated: October 2025 - v2.0 (Unified Config)*