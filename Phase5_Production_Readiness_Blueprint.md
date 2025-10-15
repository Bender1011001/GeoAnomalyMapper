# Phase 5 Production Readiness Blueprint

## 0. Alignment with Prior Phases
- Reinforce repository and tooling governance defined in [Phase0_Execution_Blueprint.md](Phase0_Execution_Blueprint.md:26), ensuring Phase 5 additions reuse the same linting, type checking, pre-commit, and GitHub Actions foundations.
- Provision infrastructure that satisfies the ingestion, preprocessing, and orchestration requirements documented in [Phase2_Data_Pipeline_Backbone_Outline.md](Phase2_Data_Pipeline_Backbone_Outline.md:101), including scalable storage backends, worker profiles, and pipeline sequencing.
- Extend the scientific observability, provenance, and validation metrics highlighted in [docs/Phase3_Scientific_Core_Blueprint_Outline.md](docs/Phase3_Scientific_Core_Blueprint_Outline.md:41) to production telemetry and alerting.
- Assume the Phase 4 SPA blueprint mandates production builds of the single-page application with CDN distribution, API gateway routing, and feature flag support managed through shared infrastructure.

## 1. Infrastructure as Code (IaC)
### 1.1 Terraform Project Layout (`iac/`)
- Root structure: `iac/environments/{dev,staging,prod}`, `iac/modules/{networking,security,compute,storage,database,observability,identity}`, `iac/policies/` for OPA/Sentinel rules, and `iac/scripts/` for helper automation per [Phase0_Execution_Blueprint.md](Phase0_Execution_Blueprint.md:15).
- Module standards: each module exposes versioned `variables.tf`, `outputs.tf`, and documented usage with examples aligned to Terraform registry expectations.
- Shared providers pinned with checksums; enforce formatting via `terraform fmt` check integrated into pre-commit hooks.

### 1.2 Environments, Variables, and State
- Use AWS S3 remote state with DynamoDB table for state locking per environment; bucket paths `gam-terraform-state/{dev,staging,prod}`.
- Leverage Terraform workspaces for environment segregation coupled with per-environment `tfvars` and secrets pulled dynamically from AWS Systems Manager Parameter Store.
- Maintain environment-specific override modules for scaling (e.g., node counts, RDS sizes) while keeping networking and security policies centralized.

### 1.3 Core Resource Provisioning
- Networking: create dedicated VPC with segmented subnets (public, private, data) and transit gateway readiness; integrate AWS Network Firewall for ingress/egress controls.
- Compute: default to Amazon EKS for orchestrating API, worker (Dask/Celery), and SPA workloads; provide fallback module for AWS ECS Fargate if lightweight tasks are needed.
- Data tier: provision Amazon RDS for PostgreSQL with PostGIS extensions, utilizing Multi-AZ deployments, automated snapshots, and read replicas for analytics.
- Object storage: configure S3 buckets for raw data, processed artifacts, and CDN assets with lifecycle policies matching Phase 2 retention expectations and cross-region replication for DR.
- Cache and messaging: deploy Amazon ElastiCache (Redis) for job orchestration, plus Amazon SQS for pipeline events aligned with Dask task distribution.
- Observability: incorporate Amazon Managed Prometheus, Amazon Managed Grafana, and Amazon OpenSearch Service for centralized metrics, dashboards, and log search aligning with [docs/Phase3_Scientific_Core_Blueprint_Outline.md](docs/Phase3_Scientific_Core_Blueprint_Outline.md:43).
- Edge and delivery: provision AWS CloudFront and AWS WAF for SPA asset delivery and API protection, referencing Phase 4 SPA requirements.

### 1.4 Secrets Management and Policy Enforcement
- Integrate HashiCorp Vault for long-lived secrets with AWS Secrets Manager for application consumption; populate Kubernetes Secrets via external secrets operator.
- Enforce policy as code: run Terraform OPA checks (Conftest) and Sentinel policies gating deployments (e.g., disallow public S3 buckets, enforce encryption).
- Automate secret rotation for database credentials, API keys, and TLS certificates; document rotation cadence in runbooks.

## 2. Containerization Strategy
### 2.1 Multi-stage Dockerfiles
- API: base on `python:3.11-slim`, multi-stage build installing dependencies with pip-tools lockfiles, run under non-root user, leverage distroless base for runtime.
- Worker (Dask/Celery): derive from API image to ensure dependency parity, add Dask/Celery packages, configure entrypoints referencing `gam/core/orchestrator` contracts in [Phase2_Data_Pipeline_Backbone_Outline.md](Phase2_Data_Pipeline_Backbone_Outline.md:103).
- SPA: use `node:20-alpine` builder for Phase 4 front-end, final stage served via `nginx:alpine` with immutable asset caching headers.
- Support GPU workloads by providing optional CUDA-based images for scientific engines consistent with [docs/Phase3_Scientific_Core_Blueprint_Outline.md](docs/Phase3_Scientific_Core_Blueprint_Outline.md:29).

### 2.2 Security and Hardening
- Enforce non-root execution, minimal OS packages, CIS benchmark scans, and read-only root filesystem where practical.
- Integrate vulnerability scanning (Trivy, Grype) in CI; fail builds on critical issues.
- Sign images using Sigstore cosign; store attestations alongside SBOM artifacts.

### 2.3 Image Lifecycle and Registries
- Host images in Amazon ECR with repository-per-service naming; apply immutable tags (`<service>:<semver>-<git-sha>`).
- Maintain separate registries or prefixes for dev, staging, prod; restrict prod registry writes to release pipelines only.
- Document local vs production differences: local Compose stack uses development images with hot reload, production relies on pinned versions and environment-specific config mounts.

## 3. CI/CD Pipeline Expansion
### 3.1 GitHub Actions Workflow Enhancements
- Extend `.github/workflows/ci.yml` from [Phase0_Execution_Blueprint.md](Phase0_Execution_Blueprint.md:43) to include Python, Node, and Terraform jobs with dependency caching.
- Add dedicated workflows: `security.yml` (Snyk, Trivy, dependency review), `docker-publish.yml` (multi-arch builds), `terraform-plan.yml` (plan on PR, apply on approval).
- Require SBOM generation via Syft, store artifacts in release attachments and ECR.

### 3.2 Delivery Pipelines
- Define environment promotions: merge to `main` triggers staging deploy via GitHub Environments with approvers; Git tags (`v*`) trigger production deploy after change advisory board approval.
- Use progressive delivery strategies: blue/green for API/EKS services, canary for workers, CloudFront versioned distributions for SPA.
- Automate rollbacks with stored manifests (Helm charts/Kustomize) and `kubectl rollout undo`; log rollback procedures in runbooks.

### 3.3 Terraform Governance
- Plans executed on PR with policy checks; applies restricted to staging and production using protected runners and manual approvals.
- Capture Terraform drift detection nightly; notify infrastructure channel on drift events.

## 4. Observability & Reliability
### 4.1 Telemetry Stack
- Logging: ship structured JSON logs to OpenSearch with index lifecycle management; tag logs with correlation IDs propagated from Phase 2 pipeline orchestrator.
- Metrics: scrape via Prometheus (API latency, worker throughput, ingestion success rate) building on existing `monitoring/prometheus/gam-metrics.yml`.
- Tracing: instrument FastAPI and worker code with OpenTelemetry, exporting to AWS X-Ray or Tempo-compatible backends.

### 4.2 Dashboards, Alerts, and SLOs
- Grafana dashboards covering pipeline health, model convergence metrics (per [docs/Phase3_Scientific_Core_Blueprint_Outline.md](docs/Phase3_Scientific_Core_Blueprint_Outline.md:44)), infrastructure capacity, and SPA user experience.
- Alert routing: integrate Alertmanager with PagerDuty for Sev1/Sev2 and Slack for informational events; define escalation policies.
- SLO catalog: API p95 latency, job completion rate, data freshness SLA, SPA availability; attach error budget policies.

### 4.3 Health, Resiliency, and Chaos
- Configure Kubernetes liveness/readiness probes aligned with service SLIs; include synthetic checks for SPA endpoints and API workflow tests.
- Schedule chaos drills (pod disruption, node failure, dependency outage simulations) quarterly; document findings and remediations.

## 5. Security & Compliance
### 5.1 Identity and Access
- Integrate Amazon Cognito or existing IdP for JWT issuance; configure API gateway authorizers and SPA token refresh flows.
- Apply Kubernetes RBAC aligned with service accounts; enforce least privilege IAM roles for workloads.
- Implement audit logging for admin actions, data access, and configuration changes.

### 5.2 Secrets and Encryption
- Eliminate plaintext prod environment variables; inject secrets at runtime via AWS Secrets Manager and Vault dynamic credentials.
- Enforce encryption at rest (EBS, S3, RDS) and TLS in transit (ACM certificates, mTLS for internal services where needed).
- Document key rotation schedule and automation (KMS CMKs, database credentials, signing keys).

### 5.3 Network and Perimeter Defense
- Utilize security groups, NACLs, and network firewalls to isolate tiers; restrict outbound traffic from private subnets.
- Deploy AWS WAF rules covering OWASP Top 10, rate limiting, and geo restrictions as required.
- Implement supply chain controls: dependency pinning, Dependabot alerts, third-party component review board.

## 6. Deployment & Operations Runbook
### 6.1 Deployment Order and Promotion
- Sequence: Infrastructure updates (Terraform) → Database migrations (Alembic gating) → Worker updates → API rollout → SPA distribution.
- Document promotion checklist per environment including pre-deploy validation, post-deploy smoke tests, and feature flag toggles.

### 6.2 Operational Roles and Procedures
- Define roles: Foundation Engineer (infrastructure), Pipeline Operator (data workflows), Scientific Owner (model validation), Support Engineer (incident response).
- Provide runbooks for scaling (EKS nodegroups, worker concurrency), incident triage, and disaster recovery aligning with Phase 2 pipeline dependencies.

### 6.3 Backup and Recovery
- Automate RDS PITR, weekly full snapshots, and cross-region copies; test restores quarterly.
- Snapshot S3 artifact buckets; maintain manifest of critical scientific datasets for reconstruction.
- Store configuration backups (Terraform state, Helm charts) in secure versioned storage; test full environment restores annually.

## 7. Assumptions, Risks, Open Questions
- Assumptions: AWS is the primary cloud; GitHub Actions remains CI/CD engine; Phase 4 SPA artifacts follow Node build pipeline; no additional compliance frameworks beyond best practices.
- Risks: Infrastructure sprawl increasing operational cost, complexity of multi-environment state management, delays in secrets rotation tooling, potential observability noise leading to alert fatigue.
- Open questions:
  - Confirm preferred secrets manager integration pattern (Vault agent vs external secrets controller) with security team.
  - Validate capacity targets for EKS clusters based on Phase 2 worker throughput projections.
  - Determine change management process for Terraform applies (single vs batched approvals).
  - Align on PagerDuty escalation policy ownership across scientific and infrastructure teams.

```mermaid
flowchart LR
    dev[Source Control]
    ci[CI Pipelines]
    scans[Security Scans]
    images[ECR Images]
    sbom[SBOM Storage]
    terraform[Terraform Apply]
    staging[Staging EKS]
    prod[Prod EKS]
    spa[CloudFront SPA]
    data[RDS PostGIS]
    observability[Prometheus Grafana OpenSearch]
    alerts[PagerDuty Slack]

    dev --> ci --> scans --> images
    scans --> sbom
    images --> staging --> prod
    images --> spa
    terraform --> staging
    terraform --> prod
    prod --> data
    prod --> observability --> alerts
    staging --> observability