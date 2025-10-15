# GeoAnomalyMapper Rebuild Master Guide

Authoritative aggregate of Phase 0–5 blueprints for rebuild execution.

## 1. Executive summary
- Project objectives:
  - Rebuild a clean, governed, and testable GeoAnomalyMapper codebase delivering reproducible multi‑modal geophysical anomaly mapping at scale.
  - Democratize access with a Streamlit SPA and robust backend services.
- Guiding principles (Phase 0):
  - Clean archival of legacy artifacts and clear traceability. Reference: [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:1)
  - Strict tooling and CI guardrails (black, mypy, pre-commit). Reference: [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:26)
  - Reproducibility, provenance, and test-first engineering across the pipeline.

## 2. Phase overviews (concise)

### Phase 0 — Execution foundation
- Key deliverables:
  - Archive legacy repo and snapshot scientific assets. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:1)
  - New repository skeleton, bootstrap targets, and initial CI workflows. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:13)
  - Tooling enforcement files and pre-commit integration. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:26)
- Primary responsibilities: Tech Lead, DevOps Engineer, Documentation Lead
- Dependencies: stakeholder approvals for archival, packaging tool selection.

### Phase 1 — Architectural blueprint (architect-mode artifact)
- Purpose: define system architecture and core contracts (e.g., `RawData`, `ProcessedGrid`, `InversionResult`) used by downstream phases.
- Note: core data contracts are present in the codebase at [`GeoAnomalyMapper/gam/core/data_contracts.py`](GeoAnomalyMapper/gam/core/data_contracts.py:1) and are referenced by Phase 2 and Phase 3 blueprints.
- Primary responsibilities: Architect, Core Engineers
- Deliverables (recorded in the architect-mode blueprint): API contracts, orchestrator patterns, namespace/module layout.

### Phase 2 — Data pipeline backbone
- Key deliverables:
  - Ingestion service with plugin architecture, resilience, caching, manifests. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:1)
  - Preprocessing stages (filtering, gridding, unit harmonization) and pipeline composition pattern.
  - PostgreSQL/PostGIS schema and Zarr/HDF5 cache layout.
- Responsibilities: Pipeline Engineer, Data Engineer, QA
- Dependencies: Phase 1 contracts, Phase 0 tooling/CI, object storage availability.

### Phase 3 — Scientific core
- Key deliverables:
  - Algorithm validation framework, modeling service, uncertainty quantification, fusion and anomaly detection components. [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:1)
  - Human-in-the-loop review workflows and scientific runbooks.
- Responsibilities: Geophysicist (scientific sign-off), Modeling Engineers
- Dependencies: Phase 2 persisted datasets, Phase 1 engine adapters.

### Phase 4 — SPA blueprint (architect-mode artifact)
- Purpose: define single-page application requirements (production builds, CDN, API gateway, feature flags).
- Note: Phase 4 is an architect-mode deliverable; Phase 5 assumes its production build artifacts and distribution model.
- Responsibilities: Product Engineer, Frontend Engineer
- Dependencies: API contracts from Phase 1 and backend services from Phase 2/3.

### Phase 5 — Production readiness
- Key deliverables:
  - IaC (Terraform) modules, environment layouts, secrets and policy enforcement. [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:1)
  - Containerization strategy, CI/CD expansion, observability, SLOs, runbooks.
- Responsibilities: Foundation Engineer, DevOps, Security, Support
- Dependencies: Phase 2 pipeline, Phase 3 modeling outputs, Phase 4 SPA artifacts.

## 3. Stakeholder role mapping (phase-by-phase)
- Geophysicist
  - Phase 1: Define scientific contracts and validation criteria (referenced in [`GeoAnomalyMapper/gam/core/data_contracts.py`](GeoAnomalyMapper/gam/core/data_contracts.py:1)).
  - Phase 3: Algorithm validation, sign-off on inversion/uncertainty reports, scientific runbooks.
  - Phase 5: Validate production observability thresholds for model quality.
- Pipeline Engineer
  - Phase 2: Design/implement ingestion plugins, caching, preprocessing stages, Postgres schema.
  - Phase 3: Provide data handles and job orchestration for modeling workloads.
  - Phase 5: Operationalize workers, scaling, and deployment sequencing.
- Architect
  - Phase 0–1: Repository skeleton, overall architecture, module contracts and orchestrator design.
  - Phase 4: Frontend/backend integration architecture (SPA).
- Foundation Engineer (Infrastructure)
  - Phase 5: IaC module development, environment state, networking, database provisioning, DR.
  - Phase 2: Provision staging infra for Postgres/Zarr caches during integration testing.
- QA / Test Engineer
  - Phase 0: Enforce tooling, pre-commit, CI testing standards.
  - Phase 2–3: Build unit/integration/regression matrices, golden datasets, performance benchmarks.
  - Phase 5: Run release validation, chaos tests, backup/restore verification.
- Product Engineer / Frontend
  - Phase 4: SPA implementation, feature flags, CDN and performance tuning.
  - Phase 5: Integrate SPA distribution into production pipeline and monitoring dashboards.

## 4. Critical assumptions (aggregated)
- Leadership approves public or private archival as required and provides access to legacy artifacts. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:4)
- Phase 1 interface contracts remain stable and available to downstream phases. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:5)
- Object storage with versioning and lifecycle policies is available for caches and artifacts. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:74)
- Primary parallelism engine is Dask; primary cloud target is AWS (Phase 5). [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:103) [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:112)
- CI remains GitHub Actions and enforcement of black/mypy/pre-commit is required. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:43)

## 5. Open questions (aggregated)
1. Packaging choice for Phase 0 (setuptools vs Poetry). [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:18)  
2. Data retention policies for raw vs processed caches. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:127)  
3. Required latency for near-real-time ingestion (affects retry/circuit thresholds). [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:126)  
4. Governance process for introducing new anomaly detectors (ML models) in Phase 3. [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:91)  
5. Secrets manager integration pattern (Vault agent vs external secrets controller). [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:115)

## 6. Risks and mitigations (aggregated)
- Risk: Third-party data outages impacting ingestion.
  - Mitigation: Circuit breakers, fallback caches, manifest-based replay. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:21)
- Risk: Cache corruption under concurrent writes.
  - Mitigation: zarr.ProcessSynchronizer, DB advisory locks, integrity checks. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:70)
- Risk: Numerical instability in modeling algorithms.
  - Mitigation: Algorithm validation framework, benchmark datasets, deterministic seeds, expert sign-off. [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:8)
- Risk: Infrastructure sprawl and operational cost.
  - Mitigation: IaC module standards, policy-as-code checks, Terraform governance. [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:11)
- Risk: Secrets or supply-chain compromise.
  - Mitigation: Secret rotation, vulnerability scanning, SBOMs, image signing. [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:41)

## 7. Next actions and checkpoints (transition to implementation)
- Phase 0 complete (checkpoint): Archive legacy repo, create clean repo skeleton, implement pre-commit + CI, obtain Phase 0 sign-off. [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:63)
- Phase 1 kickoff: Produce and freeze core data contracts and orchestrator API; publish contract doc and sample fixtures. (Owner: Architect)
- Phase 2 implementation sprint (checkpoint gated):
  - Finalize ingestion config schema and plugin API; prototype plugin loader and config hot-reload.
  - Stand up staging Postgres + Zarr stores; run E2E ingestion → preprocessing smoke tests. [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:151)
- Phase 3 validation sprint (checkpoint gated):
  - Execute algorithm validation on benchmark datasets; produce validation reports and sign-offs. [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:1)
- Phase 4 SPA integration:
  - Deliver production SPA artifacts, integrate CDN and API gateway per Phase 5 assumptions.
- Phase 5 production readiness (final checkpoint):
  - Complete IaC modules, Terraform policy checks, image publishing, observability dashboards, run full DR and chaos drills. [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:98)
- Release gating: require approvals from Geophysicist, Pipeline Engineer, QA Lead before production promotion. [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:73)

## 8. Appendix — blueprint files & key references
- Phase 0 Execution Blueprint: [`Phase0_Execution_Blueprint.md`](Phase0_Execution_Blueprint.md:1)
- Phase 1 Architectural Blueprint (architect-mode artifact; core contracts in repo): [`GeoAnomalyMapper/gam/core/data_contracts.py`](GeoAnomalyMapper/gam/core/data_contracts.py:1)
- Phase 2 Data Pipeline Backbone Outline: [`Phase2_Data_Pipeline_Backbone_Outline.md`](Phase2_Data_Pipeline_Backbone_Outline.md:1)
- Phase 3 Scientific Core Blueprint (Outline): [`docs/Phase3_Scientific_Core_Blueprint_Outline.md`](docs/Phase3_Scientific_Core_Blueprint_Outline.md:1)
- Phase 4 SPA Blueprint (architect-mode artifact)
- Phase 5 Production Readiness Blueprint: [`Phase5_Production_Readiness_Blueprint.md`](Phase5_Production_Readiness_Blueprint.md:1)

---
Implementation readiness: The repository has a complete, non-speculative master guide that consolidates Phase 0–5 authoritative blueprints and defines concrete checkpoints and owners to transition from planning to implementation. This document is the single source of truth for orchestration and execution gating.

End.