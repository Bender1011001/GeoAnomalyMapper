# Phase 3 Scientific Core Blueprint (Outline)

## 0. Alignment and Inputs
- Builds on ProcessedGrid, InversionResult, Anomaly contracts defined in [`GeoAnomalyMapper/gam/core/data_contracts.py`](GeoAnomalyMapper/gam/core/data_contracts.py:1) from Phase 1.
- Consumes Phase 2 pipeline outputs (PostgreSQL/PostGIS schemas, Zarr caches, manifests) per [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:1).
- Operates under Phase 0 execution guardrails for tooling, reproducibility, and CI enforcement (`black`, `mypy`, `pre-commit`) described in [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:26).

## 1. Algorithm Validation Framework
- **Literature Review & Provenance**
  - Require scientific method briefs referencing canonical publications; template stored in `docs/science_audit/`.
  - Mandatory reproducibility packages: data manifest (Phase 2 format), parameter sheet, notebook or script with deterministic seed control.
  - Geophysicist sign-off checklist: mathematical derivation verification, boundary-condition assumptions, unit consistency.
- **Testing Strategy**
  - Unit tests for each mathematical component with tolerances aligned to domain expectations (e.g., gravity inversion residual < 1e-6); property-based tests for invariants (symmetry, conservation).
  - Benchmark datasets curated from Phase 1 fixtures and new Phase 3 reference grids (store under `tests/data/phase3/`).
  - Golden-output regression tests hashed via manifest linking to ProcessedGrid fixture IDs.
- **Versioning & Traceability**
  - Semantic version for each validated algorithm (e.g., `SimPEG_Gravity v1.2.0`); version stored alongside metadata in PostgreSQL `models` table (Phase 2 schema).
  - Validation reports archived in object storage bucket path `s3://gam-artifacts/phase3/algorithms/<algo-version>/`.
  - CI gate enforcing validation badge before promoting algorithm to production registry.

## 2. Modeling Service Architecture
- **Service Orchestration**
  - ModelingService interface in [`GeoAnomalyMapper/gam/services/modeling_service.py`](GeoAnomalyMapper/gam/services/modeling_service.py:1) extended to accept ProcessedGrid handles and produce versioned InversionResult.
  - Orchestrator leverages Phase 2 Dask-based scheduling to spawn inversion jobs per mesh configuration; supports engine selection via config (SimPEG default, PyGIMLi optional).
- **Mesh Generation & Parameter Tuning**
  - Mesh service module referencing Phase 1 mesh helpers (`gam/modeling/mesh.py`) to create tetrahedral or hexahedral meshes; parameters stored in PostgreSQL `models.mesh_config`.
  - Parameter sweep workflow: define tuning studies in configuration, executed via distributed job queue (Dask + Kubernetes profiles from Phase 2).
  - Resource management: CPU vs GPU selectors encoded in job metadata; GPU workloads routed to dedicated node pools.
- **Engine Interface Contracts**
  - Define abstract `InversionEngine` protocol with methods: `prepare_mesh`, `run_inversion`, `export_results`.
  - Engine adapters wrap SimPEG (`gam/modeling/_archived/engines_20251003/gravity_simpeg.py`) and PyGIMLi, normalizing outputs to InversionResult.
  - Standardized output: xarray Dataset with coordinates lat, lon, depth; uncertainty dataset using same dims; metadata capturing algorithm version, parameters, solver convergence stats.

## 3. Uncertainty Quantification & Diagnostics
- **Uncertainty Methods**
  - Posterior covariance estimation for Bayesian inversions; bootstrap resampling for deterministic solvers; ensemble spread metrics for multi-run configurations.
  - Store uncertainty arrays in InversionResult.uncertainty (xarray) with attributes describing method and sample size.
  - Persist summary metrics (variance, credible intervals) in PostgreSQL `metadata` table with run ID linkage.
- **Diagnostic Artifacts**
  - Generate plots (residual histograms, convergence curves) using visualization utilities in [`GeoAnomalyMapper/gam/visualization/plots.py`](GeoAnomalyMapper/gam/visualization/plots.py:1).
  - Export diagnostic bundles (plots, JSON summaries) to `data/outputs/reports/<run-id>/`.
  - Integrate metrics with observability stack: Prometheus exporters for inversion latency, residual RMS; Grafana dashboards extended to include Phase 3 panels.
- **Operational Hooks**
  - Logging adheres to Phase 0 structured JSON standard with trace IDs; include mesh ID, algorithm version.
  - Alert thresholds for divergence or excessive uncertainty piped to alerting system configured in [`GeoAnomalyMapper/monitoring/prometheus/gam-metrics.yml`](GeoAnomalyMapper/monitoring/prometheus/gam-metrics.yml:1).

## 4. Fusion & Anomaly Detection
- **Fusion Pipeline**
  - Framework for combining multiple InversionResult objects via weighted blending, Bayesian fusion, or rule-based selection configured per modality.
  - Fusion contracts implemented in [`GeoAnomalyMapper/gam/modeling/fusion.py`](GeoAnomalyMapper/gam/modeling/fusion.py:1) extending Phase 1 joint modeling patterns.
  - Output fused models stored as ProcessedGrid-compatible xarray datasets; provenance recorded in PostgreSQL `models` table with `model_type = fused`.
- **Anomaly Detection Components**
  - Rule-based detectors (thresholding, gradient change), statistical detectors (z-score, Mahalanobis), ML models (isolation forest) defined as pluggable strategies.
  - Emit Anomaly objects conforming to [`GeoAnomalyMapper/gam/modeling/anomaly_detection.py`](GeoAnomalyMapper/gam/modeling/anomaly_detection.py:1), including confidence scoring and provenance metadata (source models, detector version).
  - Confidence scoring schema standardized: 0-1 float, with calibrations validated against benchmark datasets.
- **Human-in-the-Loop Workflow**
  - Review queue persisted in PostgreSQL `anomalies` table with status flags (pending_review, approved, rejected).
  - Threshold management via configuration service; enable overrides logged with operator ID for traceability.
  - Provide dashboard integration with review interface (extends Phase 2 observability playbooks).

## 5. Testing & Validation Strategy
- **Test Matrix**
  - Unit tests for each algorithm and detector; integration tests covering ProcessedGrid → InversionResult → fused anomalies using synthetic fixtures.
  - Regression tests using golden datasets stored in object storage; hashed outputs enforced through CI.
  - Performance benchmarks measuring inversion runtime, memory, scaling across CPU/GPU nodes; thresholds defined in CI gating rules.
- **Reproducibility Controls**
  - Deterministic seeds for stochastic methods recorded in metadata; pipeline ensures deterministic scheduling order or documented variance bounds.
  - Containerized execution environments versioned per algorithm; environment manifests stored alongside validation reports.
  - Continuous scientist-in-the-loop review documented in validation reports referencing Phase 0 governance.

## 6. Collaboration & Governance
- **Approval Workflow**
  - Deployment requires approvals from Geophysicist (scientific rigor), Pipeline Engineer (operational readiness), QA Lead (test coverage).
  - Validation reports, diagnostic summaries, and release notes captured in centralized documentation repository (`docs/science_audit/`).
- **Documentation Artifacts**
  - Scientific method briefs per algorithm; inversion runbooks; anomaly detection playbooks detailing triage procedures.
  - Update developer docs (`GeoAnomalyMapper/docs/gam.modeling.rst`) with new interfaces and extension guidelines.
- **Stakeholder Rhythm**
  - Bi-weekly Phase 3 review bridging modeling, pipeline, visualization teams to surface risks and prioritize backlog.

## 7. Assumptions, Risks, Open Questions
- **Assumptions**
  - Phase 2 data persistence (PostgreSQL + Zarr) operational and accessible for modeling workloads.
  - SimPEG remains primary engine with GPU support where available.
  - Observability stack (Prometheus, Grafana) configured as in Phase 2.
- **Risks**
  - Numerical instability in legacy algorithms without thorough validation.
  - Resource contention on shared clusters during parameter sweeps.
  - Fusion model drift if upstream calibration is inconsistent.
- **Open Questions**
  - Required cadence for re-validating algorithms against newly ingested datasets?
  - Preferred governance process for introducing new anomaly detectors (e.g., ML models) beyond existing review board?
  - Do we need regulatory compliance artifacts for specific geographies before production release?
  - Should uncertainty metrics feed into dashboard alerting thresholds automatically or via manual calibration?

## 8. Phase 3 Workflow Overview (Mermaid)
```mermaid
flowchart LR
    phase2[Phase2 outputs ProcessedGrid stores] --> validate[Algorithm validation]
    validate --> modelsvc[Modeling services InversionEngine adapters]
    modelsvc --> uncert[Uncertainty quantification diagnostics]
    uncert --> fusion[Fusion pipeline]
    fusion --> detect[Anomaly detection strategies]
    detect --> review[Human review queue]
    review --> outputs[Validated anomalies and reports]
    detect --> observability[Prometheus Grafana logs]
    modelsvc --> observability