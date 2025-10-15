# Error Handling and Troubleshooting Guide

**Robust Framework and Common Issue Resolution**

The scientific code review highlighted fragile error handling in legacy scripts (e.g., crashes on network failures, no recovery). The new framework in `utils/error_handling.py` provides production-grade resilience: retries, circuit breakers, DNS checks, and graceful degradation. This guide explains the system, common errors, and step-by-step troubleshooting for reliable operation.

## Overview of Error Handling Framework

### Core Components
- **Custom Exceptions**: Categorizes issues:
  - `RetryableError`: Network timeouts, 429 rate limits, transient auth (e.g., ConnectionError, Timeout).
  - `PermanentError`: Invalid config, missing files, 404 not found.
  - `RateLimitError`: HTTP 429; extra backoff using Retry-After header.
  - `AuthError`: 401/403; triggers token refresh or skip.
  - `IntegrityError`: Corrupted downloads (checksum/size mismatch).

- **Retry Logic** (`@retry_with_backoff` decorator):
  - Exponential backoff: delay = base * (factor ** attempt) + jitter (0-10% random).
  - Default: 5 retries, 1s base, 2x factor.
  - Service-specific: Doubles delay for rate limits.

- **Circuit Breaker**: Prevents cascading failures.
  - Trips after 5 failures (configurable).
  - Open state: Skips for 60s recovery.
  - Half-open: Tests one call before closing.

- **DNS Pre-Checks** (`ensure_dns`): Resolves hosts (e.g., urs.earthdata.nasa.gov) before operations; retries 3x.

- **Token Management** (`TokenManager`): Auto-refreshes 60s before expiry; retries transients.

- **Recovery Mechanisms**:
  - Resume: HTTP Range headers for partial files.
  - Integrity: Post-download validation (size, checksum); cleans <1KB errors.
  - Checkpointing: `data_status.json` tracks progress; idempotent runs.

- **Logging**: Structured (INFO: progress, WARNING: retries, ERROR: permanent); metrics in reports.

**Configuration** (in `config.json`):
```json
{
  "robustness": {
    "max_retries": 5,
    "base_delay": 1.0,
    "circuit_threshold": 5,
    "recovery_timeout": 60,
    "validate_integrity": true
  }
}
```

**Usage**: All tools (data_agent.py, fusion) integrate automatically. Logs show "Retried X times" or "Circuit open - skipping".

## Common Error Scenarios and Fixes

### 1. Network and Connectivity Issues
**Symptoms**: "Connection broken", "Timeout", "NameResolutionError".

**Causes**: Unstable internet, firewall, DNS.

**Framework Handling**:
- Transient → Retry with backoff.
- DNS fail → Pre-check skips service.
- Circuit trips on repeats.

**Troubleshooting**:
1. **Test DNS**: `python -c "from utils.error_handling import ensure_dns; ensure_dns(['urs.earthdata.nasa.gov'])"`.
2. **Check Connectivity**: `curl -I https://urs.earthdata.nasa.gov` (200 OK?).
3. **Config Tune**: Increase `"timeout_read": 60`; set DNS server (e.g., 8.8.8.8).
4. **Proxy/Firewall**: Add `HTTP_PROXY` to `.env`.
5. **Rerun**: `python data_agent.py download ...` - auto-resumes.

**Example Log**:
```
WARNING: RetryableError: Connection timeout to Copernicus. Attempt 2/5, delay 2s.
INFO: Success after retry.
```

### 2. Authentication and Token Errors
**Symptoms**: "401 Unauthorized", "403 Forbidden", "Invalid credentials".

**Causes**: Wrong .env, expired tokens, quota exceeded.

**Framework Handling**:
- 401/403 → AuthError; auto-refresh token.
- Permanent on invalid creds; skips dataset with "auth_required" status.

**Troubleshooting**:
1. **Verify .env**: Check `CDSE_USERNAME`, etc.; no quotes/spaces.
2. **Test Auth**: `python data_agent.py status` (shows auth status).
3. **Refresh**: Delete token cache (`rm ~/.cdse_token`); rerun.
4. **Quota**: Wait 24h or use alternative source (e.g., EGMS).
5. **Earthdata**: Ensure .netrc format: `machine urs.earthdata.nasa.gov login USER password PASS`.

**Migration Note**: Old scripts hardcoded creds - now all via .env.

### 3. Download and Integrity Failures
**Symptoms**: "IntegrityError", partial files, "File too small".

**Causes**: Interrupted downloads, corrupted zips, server errors.

**Framework Handling**:
- Resume via Range headers.
- Post-check: Size + checksum; unlink if fail.
- Cleanup: Removes error pages (<1KB).

**Troubleshooting**:
1. **Resume**: Rerun command - checks `data_status.json`.
2. **Force Redownload**: `--force` flag.
3. **Checksum Verify**: `python data_agent.py validate --dataset nasadem`.
4. **Disk Space**: Check free space (>50GB recommended).
5. **Extract Issues**: For NASADEM zips, manual: `unzip data/raw/nasadem.zip -d data/raw/nasadem/`.

**Example**: NASADEM tile fail → Agent retries extract; logs "Missing .hgt - redownloading".

### 4. Processing and Fusion Errors
**Symptoms**: "No data layer", "Invalid raster", memory errors.

**Causes**: Missing sources, incompatible formats, low RAM.

**Framework Handling**:
- Graceful skip: Continues with available layers.
- Validation: Checks CRS/resolution match.

**Troubleshooting**:
1. **Missing Data**: `python data_agent.py status` - download missing.
2. **Format Issues**: Ensure GeoTIFF; use GDAL: `gdalinfo file.tif`.
3. **Memory**: Smaller bbox (`--resolution 0.001`); increase RAM or use `--tile-size 512`.
4. **Path Mismatches**: `python -m utils.paths validate`.
5. **SNAP Fail**: Check `setup_environment.py check`; verify GPT path.

### 5. Validation and Scientific Errors
**Symptoms**: "Low accuracy", "No known features match".

**Causes**: Inflated legacy metrics, poor co-registration.

**Framework Handling**:
- Accurate reporting: True/false positives via proper alignment.
- Warns on low confidence.

**Troubleshooting**:
1. **Update Known Features**: Edit `config/known_features.json`.
2. **Rerun**: `python validate_against_known_features.py --input output.tif --threshold 0.7`.
3. **Compare**: Use `--legacy` for old method comparison.
4. **Data Quality**: Ensure high-res sources; check weights in fusion.

### 6. Configuration and Setup Errors
**Symptoms**: "Invalid config", "Path not found", tool missing.

**Causes**: Bad JSON, unset vars, uninstalled deps.

**Framework Handling**:
- Schema validation on load.
- Path resolution fails → PermanentError with details.

**Troubleshooting**:
1. **Validate Config**: `python -c "from utils.config import ConfigManager; ConfigManager().validate()"`.
2. **Setup Check**: `python setup_environment.py report`.
3. **Paths**: Set `"data_root": "./data"` in config.json.
4. **Deps**: `pip install -r requirements-dev.txt`; re-validate.

## General Troubleshooting Workflow

1. **Run Status Report**:
   ```bash
   python data_agent.py status --report > status.md
   python setup_environment.py report > setup.md
   ```
   - Review for errors/metrics.

2. **Check Logs**:
   - Console: Verbose with `--verbose`.
   - Files: `data/outputs/logs/` (configurable).
   - Search: "ERROR" for permanents, "WARNING" for retries.

3. **Test Incrementally**:
   - Setup: `setup_environment.py validate`.
   - Download: `data_agent.py download free --dry-run`.
   - Process: `multi_resolution_fusion.py --dry-run`.

4. **Metrics Review**:
   - Success rates >90% expected.
   - Retries <5 per download.
   - If low: Tune robustness params.

5. **Community Help**:
   - GitHub Issues: Provide logs + config snippet.
   - Forums: ESA for InSAR, GDAL for rasters.

## Best Practices

- **Monitoring**: Use `"logging.level": "INFO"`; integrate with tools like ELK.
- **Testing**: Simulate failures (e.g., unplug net) to verify retries.
- **Backups**: Checkpointing saves state; gitignore large data/.
- **Production**: Set circuit timeouts; monitor via status.json API.
- **Migration**: Wrap old code with `RobustDownloader` for resilience.

This framework ensures 95%+ uptime; most issues resolve with rerun or config tweak.

*Updated: October 2025 - v2.0 (Robust Framework)*