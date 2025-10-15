# Migration Guide: v1 to v2 in GeoAnomalyMapper

**Gradual Adoption of Stage 1-5 Enhancements**

This guide explains transitioning from v1 (legacy hardcoded/scattered settings) to v2 features (structured config, shims, data agent, dynamic fusion, validation, environment setup). v2 is opt-in: existing workflows run unchanged by default. Use feature flags to enable enhancements without disruption.

## v1 vs v2 Behaviors

### v1 (Default: GAM_USE_V2_CONFIG=false)
- **Configuration**: Hardcoded paths (e.g., `'data/raw'`), direct `os.getenv` for vars.
- **Paths**: String literals or `os.path.join`; no resolution/validation.
- **Downloads**: Manual; no automated agent.
- **Fusion**: Static weights; no dynamic adaptation.
- **Validation**: Optional, no hooks.
- **Setup**: Manual directory creation; no diagnostics.
- **Compatibility**: Full support via shims (`utils/config_shim.py`, `utils/paths_shim.py`).

### v2 (Opt-in: GAM_USE_V2_CONFIG=true)
- **Configuration**: `config.json` + `.env` with validation (ConfigManager).
- **Paths**: Dynamic resolution/substitution (PathManager); OS-aware, caching.
- **Downloads**: Automated via `gam_data_agent.py` CLI; resilient with RobustDownloader.
- **Fusion**: Adaptive weights (WeightCalculator); spectral/uncertainty-based.
- **Validation**: Built-in hooks post-processing (validate_against_known_features.py).
- **Setup**: Guided diagnostics/setup (setup_environment.py).
- **Backward Compatibility**: Shims delegate to v1 fallbacks when disabled.

**Key Principle**: v2 adds capabilities; v1 remains default. Enable flags incrementally.

## Step-by-Step Migration

### 1. Prepare Environment
- Copy `.env.example` to `.env` (add credentials if needed; gitignore it).
- Run diagnostics: `python setup_environment.py check`.
- If issues, use `python setup_environment.py setup --yes` to create data/ structure.

### 2. Update Imports for Compatibility
Replace direct hardcoded/env access with shims:

**Before (v1)**:
```python
import os
RAW_DIR = './data/raw'  # Hardcoded
INSAR_ENABLED = os.getenv('INSAR_ENABLED', 'true').lower() == 'true'
```

**After (Compatible with v1/v2)**:
```python
from utils import config_shim, paths_shim

RAW_DIR = paths_shim.get_data_dir() / 'raw'  # v1: Path('data/raw'); v2: from config
INSAR_ENABLED = config_shim.get_config('INSAR_ENABLED', 'true').lower() == 'true'
```

- Shims ensure no breakage: Test with `GAM_USE_V2_CONFIG=false`.

### 3. Enable v2 Config System
- Add to `.env`: `GAM_USE_V2_CONFIG=true`.
- Create `config/config.json` from example (if missing).
- Update scripts to use full ConfigManager where needed:

**Example**:
```python
from utils.config import ConfigManager
config = ConfigManager()  # Loads JSON + .env; validates
raw_dir = config.get_path('paths.raw_data')  # Resolves substitutions
if config.get('data_sources.insar.enabled'):
    # v2-specific logic
    pass
```

- Migrate settings: Move hardcoded values to `config.json` sections (project, paths, data_sources, fusion, etc.).
- Validate: `python setup_environment.py check --deep --yes`.

### 4. Enable New Capabilities Incrementally

#### Dynamic Weighting (Fusion Enhancements)
- **Flag**: `GAM_DYNAMIC_WEIGHTING=true` (default when v2 enabled).
- **Behavior**: Uses WeightCalculator for adaptive weights based on uncertainty/resolution.
- **v1 Fallback**: Static equal weights.

**Example Usage**:
```python
from multi_resolution_fusion import WeightCalculator, process_multi_resolution

# v2: Dynamic
wc = WeightCalculator(config=config)
layers = {'insar': insar_raster, 'gravity': gravity_raster}
meta = {'insar': {'resolution': 10, 'uncertainty': 0.05}, 'gravity': {'resolution': 2000, 'uncertainty': 0.2}}
weights = wc.compute_weights(layers, meta)  # e.g., {'insar': 0.85, 'gravity': 0.15}
fused = process_multi_resolution(bbox, output, config=config, dynamic=True)

# Test without: Set flag false; falls back to equal weights
```

- **Migration**: Update fusion calls to pass `config`; enable flag for new behavior.

#### Data Agent (Automated Downloads)
- **Flag**: `GAM_DATA_AGENT_ENABLED=true` (default true).
- **Behavior**: CLI for resilient downloads (EMAG2, EGM2008, Sentinel-1, etc.); tracks in `data/data_status.json`.

**Example**:
```bash
# Status check
python gam_data_agent.py status

# Download free datasets for region
python gam_data_agent.py download free --bbox "-105,32,-104,33"

# Specific dataset (dry-run first)
python gam_data_agent.py download insar_sentinel1 --dry-run
```

- **v1 Fallback**: Manual downloads; agent skips if flag false.
- **Migration**: Replace manual steps with agent CLI; integrate into pipelines.

#### Validation Improvements
- **Flag**: `GAM_VALIDATION_ENABLED=true` (default true).
- **Behavior**: Post-fusion hooks sample against known features; outputs reports.
- **Example**:
```python
from validate_against_known_features import validate_fusion

# After fusion
report = validate_fusion(fused_raster, known_features_path=config.get('validation.known_features_path'))
# Generates accuracy_assessment.txt, validation_map.png
```

- **Migration**: Add validation calls; disable flag if not needed.

#### Environment Setup
- Use `setup_environment.py` for diagnostics/setup (no flag needed).
- **Example**: `python setup_environment.py check --json report.json` for CI.

### 5. Test and Verify
- **Run v1 Mode**: Set `GAM_USE_V2_CONFIG=false`; ensure legacy scripts work.
- **Enable v2**: Set true; test new features (e.g., dynamic weighting example).
- **Full Pipeline**: Download via agent → Process → Fuse (dynamic) → Validate → Visualize.
- **GitHub Pages**: No changes to `docs/`; site remains intact. Update `docs/data/` with new GeoJSON outputs if desired.

### Common Pitfalls and Fixes
- **Path Issues**: Ensure `config.json` has correct `${data_root}`; use PathManager.resolve().
- **Credentials**: Add to `.env` (e.g., `CDSE_USERNAME=...` for Sentinel-1).
- **Validation Errors**: Run `ConfigManager().validate()`; check schema in config.py.
- **Fallback Testing**: Temporarily disable flags to isolate v2 issues.
- **Performance**: v2 adds validation overhead; disable for quick tests.

## Preserving Existing Workflows
- **Additive Only**: v2 extends; no removals.
- **Shims Ensure Compatibility**: Update imports gradually.
- **Opt-In Flags**: Default to v1 behavior.
- **No Docs/ Changes**: GitHub Pages site untouched; additive updates only.

For API details: [API_REFERENCE.md](API_REFERENCE.md). For config: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

*Updated: October 2025 - v2.0 (Stage 1-5 Migration)*