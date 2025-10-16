# Changelog

All notable changes are documented here using the guidelines from
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Version numbers follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-10-14

### Added
- Unified data acquisition through `gam.agents.gam_data_agent` with resumable
  downloads, token-aware authentication, and STAC lineage tracking.
- Configuration management backed by `config/config.json`, environment override
  support, and automatic directory materialisation.
- Resolution-aware fusion pipeline exporting Cloud Optimised GeoTIFF products
  and logging per-product weight vectors.
- Dynamic SNAP graph generation (`utils.snap_templates.GraphTemplateProcessor`)
  for Sentinel-1 interferogram processing.
- Resilience framework (`utils.error_handling`) including retry policies,
  circuit breakers, DNS validation, and structured error taxonomy.
- FastAPI service (`gam.api.main`) exposing calibrated prediction endpoints for
  point and bounding-box queries.

### Changed
- Documentation refreshed to describe the single-source configuration model,
  dynamic fusion workflow, and production deployment procedures.
- Build tooling consolidated around the `make` targets defined in the root
  `Makefile`.

### Fixed
- Eliminated duplicate configuration shims and ensured all modules consume the
  canonical configuration and path managers.
- Hardened Sentinel-1 metadata parsing with explicit subswath validation and
  descriptive errors for incompatible burst ranges.

[3.0.0]: https://github.com/your-org/GeoAnomalyMapper/releases/tag/v3.0.0
