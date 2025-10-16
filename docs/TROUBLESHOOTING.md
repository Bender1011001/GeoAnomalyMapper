# Error Handling and Troubleshooting Guide

This guide details how GeoAnomalyMapper's resilience framework mitigates common
failure modes and provides actionable diagnostics when intervention is required.
The utilities described here are production-ready and used across the entire
pipeline.

## Resilience framework recap

GeoAnomalyMapper centralises robustness logic in `utils.error_handling`:

- **Custom exceptions** — `RetryableError`, `PermanentError`, `RateLimitError`,
  `AuthError`, and `IntegrityError` allow precise handling of external failures.
- **`retry_with_backoff` decorator** — Implements exponential backoff with jitter
  for transient issues.
- **`RobustDownloader`** — Wraps HTTP downloads with retries, checksum
  validation, range-based resume, and structured logging.
- **`CircuitBreaker`** — Prevents cascading failures by pausing calls after a
  configurable number of consecutive errors.
- **`TokenManager`** — Manages credential refresh for authenticated services
  (Copernicus, Earthdata, etc.).

Configuration lives under the `robustness` block in `config/config.json`:

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

## Common scenarios

### 1. Network instability

**Symptoms** — `ConnectionError`, `Timeout`, `NameResolutionError` in logs.

**Automatic handling** — Retries with exponential backoff, DNS pre-checks via
`ensure_dns`, and circuit breaker cooldowns.

**Diagnostics**

1. Verify DNS resolution:
   ```bash
   python -c "from utils.error_handling import ensure_dns; ensure_dns(['urs.earthdata.nasa.gov'])"
   ```
2. Check connectivity:
   ```bash
   curl -I https://urs.earthdata.nasa.gov
   ```
3. If behind a proxy, set `HTTP_PROXY`/`HTTPS_PROXY` in `.env` and reload the
   configuration.

### 2. Authentication failures

**Symptoms** — HTTP 401/403 responses, log entries referencing `AuthError`.

**Automatic handling** — `TokenManager` refreshes tokens 60 seconds before
expiry and retries transient errors.

**Diagnostics**

1. Confirm credentials in `.env` (no quotes or trailing spaces).
2. Run the agent status command:
   ```bash
   python -m gam.agents.gam_data_agent status --config config/data_sources.yaml
   ```
   The output lists authentication state per service.
3. Remove stale token caches (for example, `rm ~/.cdse_token`).
4. For NASA Earthdata, ensure `~/.netrc` follows the standard format.

### 3. Download integrity issues

**Symptoms** — `IntegrityError`, truncated archives, or unexpected file sizes.

**Automatic handling** — Resume downloads using HTTP range requests, validate
checksum/size, remove incomplete files, and record failures in the status log.

**Diagnostics**

1. Re-run the download; the agent resumes from checkpoints:
   ```bash
   python -m gam.agents.gam_data_agent sync --config config/data_sources.yaml --dataset xgm2019e_gravity
   ```
2. Force a clean redownload if necessary:
   ```bash
   python -m gam.agents.gam_data_agent sync --config config/data_sources.yaml --dataset xgm2019e_gravity --force
   ```
3. Inspect available disk space and ensure sufficient quota for temporary files.

### 4. Processing or fusion errors

**Symptoms** — Missing layers, CRS mismatch, or NaN-filled outputs.

**Automatic handling** — Pre-flight validation ensures rasters share dimensions
and CRS; failures raise `PermanentError` with detailed context.

**Diagnostics**

1. Confirm the STAC catalogue is up to date:
   ```bash
   python -m gam.agents.stac_index validate --catalog data/stac/catalog.json
   ```
2. Check raster metadata:
   ```bash
   gdalinfo data/features/insar_velocity.tif | head
   ```
3. Review fusion configuration for incorrect paths or resolution values.
4. Re-run harmonisation if inputs changed:
   ```bash
   python -m gam.io.reprojection run --tiling config/tiling_zones.yaml
   ```

### 5. Validation discrepancies

**Symptoms** — Unexpected accuracy metrics or mismatched known feature hits.

**Diagnostics**

1. Verify the label dataset used for training and validation:
   ```bash
   python -m gam.features.extract_points --help
   ```
2. Ensure probability rasters are aligned with the validation geometries.
3. Run the evaluation CLI with verbose logging to inspect precision-recall
   metrics:
   ```bash
   python -m gam.models.evaluate --truth data/labels/validation_points.csv --predictions data/products/predictions.csv
   ```
4. Inspect the weight distribution reported by the fusion stage to confirm that
   higher confidence layers dominate in areas with ground truth support.

### 6. Configuration problems

**Symptoms** — `KeyError` or `ValueError` when loading configuration, missing
directories at runtime.

**Diagnostics**

1. Print the active configuration:
   ```bash
   python -m utils.config
   ```
2. Validate directory settings:
   ```bash
   python -c "from utils.paths import paths; print(paths.items())"
   ```
3. Ensure environment overrides use uppercase keys with double underscores (for
   example, `GAM__PATHS__RAW_DATA`).

## General workflow for incident response

1. **Capture logs** — Structured logs include timestamps, dataset identifiers,
   retry counts, and error categories. Preserve them for audits.
2. **Reproduce with minimal scope** — Re-run the failing command with a narrowed
   dataset or bounding box to isolate the issue.
3. **Adjust configuration safely** — Apply overrides via environment variables or
   a temporary copy of `config.json`; avoid editing tracked defaults during an
   incident.
4. **Document fixes** — Record configuration changes, datasets affected, and
   verification steps in your operations log.

GeoAnomalyMapper's resilience layer is engineered for high availability. By
following these procedures you can maintain stable operations even when external
services behave unpredictably.
