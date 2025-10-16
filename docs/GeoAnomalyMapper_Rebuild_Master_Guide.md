# GeoAnomalyMapper Operations Master Guide

This guide consolidates the practices, responsibilities, and guardrails that keep GeoAnomalyMapper production-ready. It reflects the system as deployed today and serves as the authoritative reference for engineering, geophysics, and operations teams.

## 1. Objectives
- Deliver reproducible, multi-modal geophysical anomaly mapping at scale with defensible scientific provenance.
- Provide a governed codebase backed by continuous integration, automated testing, and strict tooling enforcement.
- Maintain a world-class user experience across APIs and analytical notebooks.

## 2. Foundation
- **Tooling guardrails** – `black`, `ruff`, `mypy`, and pre-commit run in CI; merge gates require green pipelines.
- **Configuration governance** – `config/config.json` and environment overrides are version-controlled; configuration changes undergo the same review process as code.
- **Artifact retention** – Raw, interim, feature, and model artefacts are stored under hashed manifests with lifecycle policies and disaster recovery coverage.

## 3. Core Responsibilities
- **Geophysicist** – Owns scientific briefs, validates inversion and fusion outputs, and signs off on release metrics.
- **Pipeline Engineer** – Maintains ingestion agents, preprocessing recipes, and orchestration flows; guarantees deterministic builds.
- **Infrastructure Engineer** – Operates Terraform modules, Kubernetes clusters, storage accounts, and observability stacks.
- **QA Lead** – Curates regression suites, golden datasets, and chaos drills; enforces release checklists.

## 4. Execution Workflow
1. **Planning** – Capture requirements in scientific briefs and architecture notes; update docs alongside code.
2. **Implementation** – Develop in feature branches with unit and integration tests.
3. **Validation** – Run the full Carlsbad benchmark suite, pyshtools harmonic checks, and reproducibility comparisons.
4. **Review** – Geophysicist, Pipeline Engineer, and QA Lead jointly review pull requests and release candidates.
5. **Release** – Tag semver release, publish artefacts, and update observability dashboards.
6. **Post-release** – Monitor metrics, review anomaly feedback, and schedule improvement sprints.

## 5. Risk Management
- **Data availability** – Download agents implement retries, circuit breakers, and manifest-based replays; S3 caches serve as fallback.
- **Numerical stability** – Solver diagnostics, mesh quality checks, and adaptive damping mitigate divergence.
- **Security** – Secrets rotate automatically, supply-chain scans produce SBOMs, and container images are signed prior to deployment.
- **Operational cost** – Autoscaling policies, workload quotas, and usage dashboards keep compute and storage within budget.

## 6. Documentation Set
- **README** – Quickstart, highlights, and architecture overview.
- **Configuration Guide** – Authoritative documentation for `config.json`, YAML recipes, and environment overrides.
- **Developer Guide** – Extension points for utilities, paths, and error handling.
- **Troubleshooting Manual** – Structured response plans for network, authentication, integrity, processing, and validation issues.
- **Physics Weighting Guide** – Full description of the fusion weighting system with calibration advice.
- **Scientific Core Architecture** – Modeling, inversion, uncertainty, and governance blueprint.

## 7. Release Checklist
1. Lint, type-check, and run tests in CI.
2. Execute `make stac`, `make download`, `make harmonize`, `make features`, `make train`, `make infer`, and `make vectorize` on a clean environment.
3. Validate Carlsbad benchmark metrics (F1 ≥ 0.70, precision ≥ 0.74, recall ≥ 0.68).
4. Publish documentation updates, ensuring every configuration or interface change is reflected.
5. Record release notes in `CHANGELOG.md` and archive artefact manifests.
6. Obtain sign-offs from Geophysicist, Pipeline Engineer, QA Lead.

## 8. Continuous Improvement
- **Metrics reviews** – Weekly review of observability dashboards, anomaly feedback, and compute usage.
- **Backlog refinement** – Quarterly planning to prioritise new datasets, algorithm upgrades, and infrastructure evolution.
- **Knowledge sharing** – Regular tech talks, scientific debriefs, and runbook updates keep the team aligned and audit-ready.

*Updated: October 2025*
