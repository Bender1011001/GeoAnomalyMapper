# Validation-First Blind Known-Void Workflow

This repository can now run a validation-first workflow that separates blind
candidate generation from withheld-label scoring. The harness enables blind
known-void validation, but it does not prove field accuracy until real holdout
sites are run and scored.

## Separation of Inputs

- Public runner input: `validation_examples/public_manifest_fixture.json`
  - Site identifiers, names, public centers, run area, processing mode, and
    candidate CSV paths or no-download SAR search parameters.
  - Must not include known void geometry, expected depths, ground-truth labels,
    or paths to withheld labels.
- Public real holdout scaffold:
  `validation_examples/public_manifest_real_holdout_scaffold.json`
  - Broad public target regions plus matched comparand regions.
  - Includes deterministic Sentinel-1 SLC search windows and selection counts.
  - Does not identify site class, truth geometry, cave-passage geometry, or
    expected depths.
- Public campaign scaffold:
  `validation_examples/public_manifest_campaign_scaffold.json`
  - Template-only small multi-site campaign example for calibration, primary
    holdout-candidate, and audit-only entries.
  - Uses neutral `pair_id`/`group_id`, public split names, public strata,
    public site category, acquisition stratum, terrain/land-cover/lithology
    descriptors, campaign tier, provider/comparison arms, and broad public
    context.
  - It is intentionally smaller than the preregistered primary holdout target
    recommendation. The campaign registry should expand copied manifests toward
    24 broad positive regions plus 24 matched null controls only after
    calibration review, parameter lock, and private label-custody setup.
- Withheld scorer input: `validation_examples/withheld_labels_fixture.json`
  - Site class (`positive` or `negative`) and known void labels.
  - Used only by the score stage after candidate outputs are frozen.
- Withheld-label template: `validation_examples/withheld_labels_template.json`
  - Private custodian scaffold for `site_class`, `site_subclass`, label
    provenance, evidence tier, custody records, scoring eligibility, and
    release/custody metadata.
  - This template is intentionally truth-bearing and must be copied to private
    storage before real values are added. It must not be mixed into public
    manifests or report packages.
- SAR inventory output: `sar_inventory.json`
  - Search-only product metadata from public manifest sites.
  - Records search parameters, time windows, auth mode without secret values,
    product IDs/names, provider, processing level, beam mode, orbit metadata
    when available, size, timestamps, and selected products.
- Product lock output: `product_lock.json`
  - Deterministically freezes selected product IDs from the inventory so later
    real execution can use the same selection or detect drift.
- Parameter set: `validation_examples/validation_parameter_set_template.json`
  - Records thresholds, resolution profile, PINN settings, scoring tolerances,
    provider preferences, comparison arms, split policy, model card, and analysis
    plan.
  - Must not include labels, truth geometry, expected site class, or withheld-label
    paths.
  - Has a deterministic canonical hash and short ID. Approval metadata is not
    part of the canonical hash, so a calibrated set can be marked approved
    without changing the scientific parameter identity.
- Campaign registry: `validation_examples/campaign_registry_template.json`
  - Records the public campaign ID, protocol/preregistration references, public
    manifest hash, parameter-set hash/ID, approved-parameter requirement, split
    policy, provider arms, scoring tolerances, metric definitions, target counts,
    immutable artifact hashes, and explicit `draft` / `approved` / `locked`
    status.
  - Must not include private labels, truth geometry, field evidence, secret keys,
    credential values, or withheld-label paths.
  - Has a deterministic canonical hash and short registry ID. Status, approval,
    and lock metadata are not part of the canonical hash, so a reviewed registry
    can be approved or locked without changing its preregistered scientific
    content identity.
- Frozen candidate output: `frozen_candidates.csv`
  - A copied or generated snapshot of pipeline candidates, compatible with the
    existing `detected_anomalies.csv` schema.

## Public vs Private Campaign Fields

Public manifests are runner inputs and may preserve no-label campaign metadata
needed for stratified analysis and reproducibility:

- `split` / `split_designation` for calibration, primary holdout, optional
  extension, or audit-only bookkeeping.
- Neutral `pair_id` and `group_id` values that do not encode positive/control
  truth.
- `public_strata` / `public_stratum`, `public_site_category`,
  `acquisition_stratum`, `terrain_descriptor`, `land_cover_descriptor`,
  `lithology_descriptor`, `campaign_tier`, and broad public-context notes.
- `prior_run_status`, `audit_only`, `prior_run_audit_only`, and
  `public_scoring_status` for transparent handling of calibration or audit-only
  sites.
- Public acquisition settings, provider IDs, comparison arms, and placeholder
  provider notes.

Public manifests must not contain truth-bearing or custody-bearing fields. The
validator recursively rejects label/truth key aliases such as `site_class`,
`positive`, `negative`, `label`, `labels`, `known_voids`, `truth_geometry`,
`ground_truth`, `known_cave_geometry`, `expected_depth_m`,
`exact_void_coordinates`, `private_label_path`, `withheld_labels_path`,
`label_provenance`, `evidence_tier`, `custody_record`, `scoring_eligibility`, and
`release_metadata`. This rejection applies inside nested metadata objects and
provider notes, not only at the target top level.

Private withheld-label files are scorer-only inputs. They may include site class,
site subclass, label provenance, evidence tier, exact known-void locations,
tolerances, custody record, scoring eligibility, and release/custody metadata.
Keep them outside the runner path, do not commit real private files, and do not
provide private-label paths in public manifests. Template files can be validated
with `--allow-templates` before copying them to a private location.

## Safe Fixture Smoke Test

These commands use only committed local fixtures and do not download SAR data or
start PINN training:

```bash
python blind_validation.py validate-public --manifest validation_examples/public_manifest_fixture.json
python blind_validation.py run --manifest validation_examples/public_manifest_fixture.json --output-dir data/blind_validation/fixture_run
python blind_validation.py score --run-manifest data/blind_validation/fixture_run/run_manifest.json --labels validation_examples/withheld_labels_fixture.json --output data/blind_validation/fixture_score.json
python blind_validation.py baseline-report data/blind_validation/fixture_score.json --output-json data/blind_validation/fixture_baseline_report.json --output-text data/blind_validation/fixture_baseline_report.txt
python blind_validation.py package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --baseline-json data/blind_validation/fixture_baseline_report.json --baseline-text data/blind_validation/fixture_baseline_report.txt --output-dir data/blind_validation/fixture_report_package
```

The fixture intentionally includes one positive hit and one negative-control
false positive so the score report exercises both sides of the metrics.

The same workflow can be reached through the stable top-level research-product
CLI surface:

```bash
python geoanomaly.py validation validate-public --manifest validation_examples/public_manifest_fixture.json
python geoanomaly.py validation run --manifest validation_examples/public_manifest_fixture.json --output-dir data/blind_validation/fixture_run
python geoanomaly.py validation score --run-manifest data/blind_validation/fixture_run/run_manifest.json --labels validation_examples/withheld_labels_fixture.json --output data/blind_validation/fixture_score.json
python geoanomaly.py validation package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --output-dir data/blind_validation/fixture_report_package
```

Use `python geoanomaly.py commands` to print canonical validation-first examples.

## Release Hardening Smoke Test

Release hardening is intentionally validation-first and no-download. The top-level
health command checks local prerequisites and example/test command availability
without loading withheld labels, downloading SAR products, or starting PINN
training:

```bash
python geoanomaly.py health --json --skip-gpu
python geoanomaly.py preflight --json --skip-gpu
```

After local editable install, the packaged console entry point is equivalent:

```bash
python -m pip install -e .
geoanomaly health --json --skip-gpu
geoanomaly commands
```

The health report includes:

- Python version and platform metadata.
- Core module and declared dependency availability using import discovery.
- Local data/output directory existence and writability. Use `--create-dirs` only
  when empty local output folders should be created.
- `.env` presence and expected Earthdata key names without printing values.
- Free disk estimate from the standard library when available.
- Optional GPU availability; pass `--skip-gpu` for CPU-only CI and smoke tests.
- Safe fixture workflow commands and `python -m unittest discover -s tests`.

Release smoke commands for a fresh clone:

```bash
python -m pip install -e .
geoanomaly health --json --skip-gpu
python -m compileall -q blind_validation.py geoanomaly.py tests
python -m unittest discover -s tests
```

The committed CI workflow `.github/workflows/validation-first-ci.yml` runs with
empty Earthdata credential variables and must not be modified to inject secrets
for fixture validation. Real-data workflows should remain separate, explicit, and
reviewed because they may search/download remote products or trigger long
processing.

## Clean Environment Reproducibility

Use a fresh virtual environment when validating install metadata or reproducing a
release-smoke failure. These commands are intentionally no-download/no-training
with respect to SAR products and PINN execution; only Python package installation
uses the package index.

Windows command prompt / PowerShell:

```powershell
python -m venv .venv-clean
.venv-clean\Scripts\python -m pip install --upgrade pip
.venv-clean\Scripts\python -m pip install -e .
.venv-clean\Scripts\python -m compileall -q blind_validation.py geoanomaly.py json_utils.py tests
.venv-clean\Scripts\python -m unittest discover -s tests -p test_release_hardening.py
.venv-clean\Scripts\python -m unittest discover -s tests -p test_blind_validation.py
```

POSIX shell equivalent:

```bash
python -m venv .venv-clean
.venv-clean/bin/python -m pip install --upgrade pip
.venv-clean/bin/python -m pip install -e .
.venv-clean/bin/python -m compileall -q blind_validation.py geoanomaly.py json_utils.py tests
.venv-clean/bin/python -m unittest discover -s tests -p test_release_hardening.py
.venv-clean/bin/python -m unittest discover -s tests -p test_blind_validation.py
```

Dependency name note: the declared package is `scikit-gstat>=1.0.0`, while the
Python import checked by the health report is `skgstat`. This package/import name
split is expected and is recorded in both `requirements.txt` and
`pyproject.toml`.

## Calibration Lock Workflow

Treat calibration as a pre-holdout artifact. Create or copy a parameter set,
calibrate it only on calibration data, then freeze and approve it before any
withheld holdout labels are exposed.

Create a draft parameter set template:

```bash
python blind_validation.py init-parameters --validation-id real_holdout_broad_regions_v1 --name broad_region_quick_v1 --output data/blind_validation/parameters/broad_region_quick_v1.json
```

Validate and print the canonical hash/ID:

```bash
python blind_validation.py validate-parameters --parameter-set data/blind_validation/parameters/broad_region_quick_v1.json --allow-templates
```

After calibration review, set approval metadata in the copied JSON, or generate an
approved template only if the exact default parameters are the reviewed set:

```bash
python blind_validation.py init-parameters --validation-id real_holdout_broad_regions_v1 --name broad_region_quick_v1 --output data/blind_validation/parameters/broad_region_quick_v1_approved.json --approved --approved-by validation_custodian --approved-at-utc 2026-01-01T00:00:00Z
python blind_validation.py validate-parameters --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved
```

Compare draft and approved copies, or compare a candidate change against the
locked reviewed set:

```bash
python blind_validation.py compare-parameters --reference data/blind_validation/parameters/broad_region_quick_v1_approved.json --candidate data/blind_validation/parameters/broad_region_quick_v1_candidate.json --output data/blind_validation/parameters/parameter_comparison.json --fail-on-changed
```

Record the approved parameter set in inventories, product locks, and run
manifests:

```bash
python blind_validation.py preflight --manifest validation_examples/public_manifest_real_holdout_scaffold.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --output-dir data/blind_validation/holdout_inventory --write-lock
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --output-dir data/blind_validation/holdout_dry_run
```

At scoring time, require the same approved hash. This check happens before labels
are loaded, so changed or unapproved parameter sets fail without touching withheld
truth:

```bash
python blind_validation.py score --run-manifest data/blind_validation/holdout_real_run/run_manifest.json --labels path\to\private_withheld_labels.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --output data/blind_validation/holdout_score.json
```

## Campaign Registry Lock Workflow

The campaign registry is the preregistration binding for the multi-site blind
campaign. It freezes the public manifest identity, parameter-set identity,
approved-parameter requirement, split policy, provider/comparison arms, scoring
tolerances, metric definitions, target counts, and immutable artifact hashes
before no-download inventory, product locking, and later execution. It remains a
public/no-label artifact and must never carry labels, field evidence, secrets, or
paths to withheld-label files.

Validate the committed no-label template:

```bash
python blind_validation.py validate-campaign-registry --registry validation_examples/campaign_registry_template.json --allow-templates
```

Create a draft registry from a copied public manifest and approved parameter set:

```bash
python blind_validation.py init-campaign-registry --validation-id real_holdout_broad_regions_v1 --campaign-id real_holdout_broad_regions_registry_v1 --name broad_region_campaign_v1 --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --output data/blind_validation/campaign_registry.json
```

Compare the registry against the current manifest and parameter set; use
`--fail-on-drift` in automation so edits fail before inventory or execution:

```bash
python blind_validation.py compare-campaign-registry --registry data/blind_validation/campaign_registry.json --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --output data/blind_validation/campaign_registry_compare.json --fail-on-drift
```

After independent review, approve and lock the registry. The status transition
updates approval/lock metadata while preserving the canonical registry hash for
the scientific content:

```bash
python blind_validation.py approve-campaign-registry --registry data/blind_validation/campaign_registry.json --output data/blind_validation/campaign_registry_approved.json --approved-by validation_custodian --approved-at-utc 2026-01-01T00:00:00Z
python blind_validation.py lock-campaign-registry --registry data/blind_validation/campaign_registry_approved.json --output data/blind_validation/campaign_registry_locked.json --locked-by validation_custodian --locked-at-utc 2026-01-01T00:00:00Z
python blind_validation.py validate-campaign-registry --registry data/blind_validation/campaign_registry_locked.json --require-locked
```

Use the locked registry in no-download inventory and product-lock creation. This
records `campaign_registry_id` and `campaign_registry_hash` in both inventory and
product lock, and fails on manifest/parameter drift before search-only inventory
metadata is written:

```bash
python blind_validation.py preflight --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --campaign-registry data/blind_validation/campaign_registry_locked.json --output-dir data/blind_validation/holdout_inventory --write-lock
```

Use the same locked registry for dry-run or later explicit real execution. The
runner records registry identity in `run_manifest.json` and reproducibility
metadata, and `--require-locked-campaign-registry` rejects draft or merely
approved registries:

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --campaign-registry data/blind_validation/campaign_registry_locked.json --require-locked-campaign-registry --output-dir data/blind_validation/holdout_dry_run
```

### Campaign-level execution plan, status, and no-label evidence

Use the campaign-level driver after the copied public manifest, approved
parameter set, locked registry, and product lock have been reviewed. The default
workflow is no-download/dry-run-safe: planning and status commands emit strict
JSON, do not load withheld labels, do not start SAR downloads, and do not start
PINN training.

Create a deterministic per-target execution plan from the public manifest,
locked registry, approved parameter set, and product lock:

```bash
python blind_validation.py campaign-plan --manifest validation_examples/public_manifest_real_holdout.json --campaign-registry data/blind_validation/campaign_registry_locked.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --product-lock data/blind_validation/holdout_inventory/product_lock.json --output-dir data/blind_validation/campaign_execution --output data/blind_validation/campaign_execution_plan.json
```

The plan records one command per target using `run --target-id ...`, records the
locked registry/parameter/product-lock identities, and marks provider support,
selected products, output paths, and skip/block reasons. Audit-only targets are
excluded from primary execution by default and appear as `skipped_audit_only`.
Provider arms without current end-to-end real execution support, such as
Umbra/X-band placeholders, are skipped or flagged instead of being silently run.
For audit planning only, use `--include-audit-only` or
`--allow-unsupported-provider-arms`; do not use those flags to convert audit-only
or unsupported placeholder arms into primary locked validation.

Run the dry-run-safe campaign plan. This creates/resumes per-target dry-run
manifests and frozen candidate CSV snapshots without downloads or training:

```bash
python blind_validation.py campaign-run --plan data/blind_validation/campaign_execution_plan.json --output data/blind_validation/campaign_execution_status.json
```

Check status at any time without rerunning targets:

```bash
python blind_validation.py campaign-status --plan data/blind_validation/campaign_execution_plan.json --output data/blind_validation/campaign_execution_status.json
```

Package a no-label campaign evidence summary for execution readiness review:

```bash
python blind_validation.py campaign-package --plan data/blind_validation/campaign_execution_plan.json --status data/blind_validation/campaign_execution_status.json --sar-inventory data/blind_validation/holdout_inventory/sar_inventory.json --product-lock data/blind_validation/holdout_inventory/product_lock.json --output data/blind_validation/campaign_no_label_evidence.json
```

The campaign package is intentionally not a scoring package. It includes plan,
status, manifest/registry/parameter/product-lock hashes, per-target run-manifest
hashes when present, candidate counts, and claim-boundary text, but it excludes
withheld labels and score outputs.

### Robustness and ablation planning

Reviewer-facing robustness work is represented as a public, no-label robustness
plan before any heavy processing. The committed template is
`validation_examples/robustness_ablation_plan_template.json`. It covers the
expected ablation families for threshold sensitivity, minimum anomaly voxel
thresholds, morphology iterations and connected-component topology, depth-prior
or PINN regularization sensitivity, SAR preprocessing and vibrometry quality
filters, top-k candidate cutoffs for physically extreme candidate volume, null
region/random-spatial baselines, and repeat product/date stability groups.

Validate the template without downloads, inventory searches, withheld labels, or
training:

```bash
python blind_validation.py validate-robustness-plan --plan validation_examples/robustness_ablation_plan_template.json --allow-templates
```

For a real campaign, copy the template, remove `template_only`, set the final
`validation_id`, `campaign_id`, public artifact references, and variant matrix,
then validate the copied plan without `--allow-templates`. Robustness plans are
strict public artifacts: the validator rejects labels, known voids, truth
geometry, field evidence, secret values, private-label paths, and withheld-label
paths. The plan hash and short ID are canonicalized after normalizing defaults,
so check them with the validation command after editing.

The policy block must keep these safety gates unless an independent protocol
change is reviewed:

- `labels_withheld_from_runner: true`
- `no_downloads_or_training: true`
- `default_execution_mode: dry_run_plan_only`
- `parameter_changes_after_holdout_unblinding: forbidden`
- `ablation_use_rule: calibration_only_unless_preregistered_before_holdout_unblinding`

Ablations are calibration-only by default. A variant with holdout scope is valid
only when the robustness plan was preregistered and locked before holdout
unblinding, with `preregistration.status` set to
`preregistered_before_holdout_unblinding` or
`locked_before_holdout_unblinding` and `holdout_unblinding_status` still showing
that labels are not unblinded. Do not use post-hoc holdout ablations to tune
parameters after withheld labels are known.

Create a deterministic dry-run-only ablation execution plan linked to the same
public manifest, locked campaign registry, and approved parameter set:

```bash
python blind_validation.py robustness-plan --plan data/blind_validation/robustness_plan.json --manifest validation_examples/public_manifest_real_holdout.json --campaign-registry data/blind_validation/campaign_registry_locked.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --output-dir data/blind_validation/robustness_runs --output data/blind_validation/robustness_execution_plan.json
```

The resulting strict JSON records the robustness plan hash/ID, campaign registry
hash/ID, parameter-set hash/ID, selected targets, variant matrix, and dry-run
commands containing `run --robustness-plan ... --variant-id ...`. It does not
execute real processing. Calibration targets are selected by default; use
`--include-audit-only` only for audit planning, and use
`--include-preregistered-holdout` only for holdout ablations that were
preregistered before holdout unblinding.

Individual dry-run manifests can record the selected robustness variant without
altering processing by itself:

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --campaign-registry data/blind_validation/campaign_registry_locked.json --require-locked-campaign-registry --robustness-plan data/blind_validation/robustness_plan.json --variant-id void_threshold_025 --output-dir data/blind_validation/robustness_runs/void_threshold_025
```

The top-level wrapper delegates the same commands through
`python geoanomaly.py validation validate-robustness-plan`,
`python geoanomaly.py validation robustness-plan`, and
`python geoanomaly.py validation robustness-summary`.

Real campaign execution is opt-in and must be planned and run explicitly. The
planner refuses synthetic fallback, refuses template/draft/drifted inputs, and
requires `--product-lock` plus `--confirm-real-downloads-and-training` before a
real-execution plan can be written:

```bash
python blind_validation.py campaign-plan --manifest validation_examples/public_manifest_real_holdout.json --campaign-registry data/blind_validation/campaign_registry_locked.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --product-lock data/blind_validation/holdout_inventory/product_lock.json --output-dir data/blind_validation/campaign_real_execution --output data/blind_validation/campaign_real_execution_plan.json --execute-real --confirm-real-downloads-and-training
python blind_validation.py campaign-run --plan data/blind_validation/campaign_real_execution_plan.json --execute-real --confirm-real-downloads-and-training --output data/blind_validation/campaign_real_execution_status.json
```

Run those real commands only after disk headroom, selected product sizes,
provider limitations, and non-synthetic settings are independently reviewed.

Registry validation and comparison recursively reject label-like keys such as
`labels`, `known_voids`, `site_class`, `withheld_labels_path`,
`private_label_path`, and field-evidence/secret-like keys. It also rejects
path-like string values that point to withheld-label files. Keep private custody
and withheld-label release records outside the registry.

## Real Holdout Dataset Procedure

1. Choose public broad regions and matched comparand regions.
   - Start from `validation_examples/public_manifest_real_holdout_scaffold.json`
     for a ready-to-edit broad scaffold.
   - Or copy `validation_examples/public_manifest_real_template.json` for a
     single-site template.
   - Keep all ground truth, site class, cave geometry, expected depths, and
     private map links out of the public manifest.
2. A validation custodian keeps withheld labels separately, outside the runner
   path, by copying `validation_examples/withheld_labels_template.json` to a
   private location and filling site class plus known-void locations there.
3. The runner receives only the public manifest.
4. The runner/custodian freezes an approved parameter set before holdout scoring.
5. The custodian creates, validates, approves, and locks a no-label campaign
   registry binding the public manifest, approved parameter set, split policy,
   metrics, provider arms, target counts, and immutable artifact hashes.
6. The runner builds a no-download SAR inventory and product lock, recording the
   parameter-set hash/ID and campaign registry hash/ID.
7. The runner creates a campaign-level execution plan/status summary and
   no-label evidence package. This dry-run-safe stage can resume existing
   per-target outputs and excludes audit-only/unsupported provider arms by
   default.
8. Later, after product selection is frozen, the runner executes real validation
   explicitly and freezes candidate CSVs.
9. The scorer verifies the approved parameter-set hash, then combines the frozen candidates with private withheld labels and
   writes a JSON score report.

Template setup for a custom manifest:

```bash
copy validation_examples\public_manifest_real_template.json validation_examples\public_manifest_real_holdout.json
python blind_validation.py validate-public --manifest validation_examples/public_manifest_real_holdout.json
```

Validate the committed broad-region scaffold:

```bash
python blind_validation.py validate-public --manifest validation_examples/public_manifest_real_holdout_scaffold.json
```

Validate the committed campaign scaffold without downloads or private labels:

```bash
python blind_validation.py validate-public --manifest validation_examples/public_manifest_campaign_scaffold.json --allow-templates
```

Use the campaign scaffold only as a public-safe starting point. The campaign
registry workflow should copy it, remove `template_only`, assign final neutral
public IDs, expand the target list, and lock split/pair/group/acquisition strata
without adding private labels. Runner and preflight commands may consume the
copied public manifest only after registry comparison passes; scoring must wait
until frozen outputs exist and the private custodian supplies a separate
withheld-label file.

Validate the private withheld-label template schema before copying to private
custody storage:

```bash
python blind_validation.py validate-labels --labels validation_examples/withheld_labels_template.json --allow-templates
```

Do not run scoring against the template. Real scoring labels must be supplied by
the custodian only after candidate outputs are frozen and parameter/product locks
are verified.

### Mammoth prior-run audit-only handling

Mammoth is useful for audit and calibration because it has prior-run history in
the project, but that prior exposure means it must not be treated as a primary
holdout scoring site. If a Mammoth-area public entry is included, mark it with
public no-label metadata such as `split="audit"`,
`split_designation="audit_only_prior_run"`, `campaign_tier="audit_calibration_only"`,
`prior_run_status="prior_run_public_target_audit_only"`, `audit_only=true`,
`prior_run_audit_only=true`, and
`public_scoring_status="not_primary_holdout_scoring_site"`. These fields document
exclusion from primary holdout scoring without revealing a positive/control label
or any known cave geometry.

## No-Download SAR Preflight Inventory

The inventory command is search-only by design. It does not call product
download APIs, does not run vibrometry, and does not start PINN training. It may
load local `.env` values to authenticate ASF/CMR search, but it records only auth
mode and secret presence metadata, never token or password values.

Run a no-download inventory for every public manifest target and write a product
lock in the same output directory:

```bash
python blind_validation.py preflight --manifest validation_examples/public_manifest_real_holdout_scaffold.json --output-dir data/blind_validation/holdout_inventory --write-lock
```

For a locked campaign, include both approved parameters and the locked campaign
registry:

```bash
python blind_validation.py preflight --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --campaign-registry data/blind_validation/campaign_registry_locked.json --output-dir data/blind_validation/holdout_inventory --write-lock
```

Equivalent alias:

```bash
python blind_validation.py inventory --manifest validation_examples/public_manifest_real_holdout_scaffold.json --output data/blind_validation/holdout_inventory/sar_inventory.json --lock-output data/blind_validation/holdout_inventory/product_lock.json
```

For an auth-independent parser smoke test that does not load `.env`:

```bash
python blind_validation.py preflight --manifest validation_examples/public_manifest_real_holdout_scaffold.json --output data/blind_validation/no_dotenv_inventory.json --no-dotenv
```

That last command may report search failures if ASF authentication or the
`asf_search` dependency is unavailable, but it still writes a safe inventory
record when an output path is provided.

## Higher-Quality SAR Comparison Scaffolding

Public manifests can now express provider preferences and comparison arms. Use
`comparison_arms` on a target to plan Sentinel-1 C-band versus X-band Spotlight
comparisons without downloading commercial products or exposing credentials.

Template:

```bash
copy validation_examples\public_manifest_sar_comparison_template.json validation_examples\public_manifest_sar_comparison_holdout.json
python blind_validation.py validate-public --manifest validation_examples/public_manifest_sar_comparison_holdout.json --allow-templates
```

Supported provider IDs in validation metadata are:

- `sentinel1_asf`: search-only ASF Sentinel-1 C-band SLC inventory, real lock
  supported for current locked execution.
- `umbra_open_data`: X-band Spotlight metadata placeholder in blind-validation
  inventories. Current end-to-end lock/download execution is not wired through
  this harness.
- `capella_commercial`, `iceye_commercial`, and
  `xband_spotlight_placeholder`: commercial or generic placeholder arms. Do not
  put credentials, order details, API keys, or private procurement data in public
  manifests.

When an arm is not search-supported, preflight writes a
`placeholder_not_searched` record with provider, band, arm, limitations, and
`placeholder_only=true`; it does not download or contact commercial ordering
systems. Current locked real execution remains Sentinel-1 only and is guarded by
product-lock verification before any download.

Score and baseline reports propagate `provider`, `provider_label`,
`comparison_arm`, and `comparison_group`. Aggregates now include
`by_provider`, `by_provider_label`, `by_comparison_arm`, and metadata counts when
score inputs carry those fields.

## Freeze Product Selections

`preflight --write-lock` writes `product_lock.json` directly. To rebuild a lock
from an existing inventory, use:

```bash
python blind_validation.py lock-products --inventory data/blind_validation/holdout_inventory/sar_inventory.json --output data/blind_validation/holdout_inventory/product_lock.json
```

To detect drift against a previous lock:

```bash
python blind_validation.py lock-products --inventory data/blind_validation/holdout_inventory/sar_inventory.json --previous-lock data/blind_validation/holdout_inventory/product_lock.json --output data/blind_validation/holdout_inventory/product_lock_next.json --fail-on-changed-selection
```

Selection is deterministic from search results using the
`start_time_desc_product_id_asc` policy: most recent product start time first,
then product ID/name as a stable tie-breaker.

Dry-run freeze without downloads or training:

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --campaign-registry data/blind_validation/campaign_registry_locked.json --require-locked-campaign-registry --output-dir data/blind_validation/holdout_dry_run
```

## Execute Real Validation Later

Real execution is opt-in only and should be done only after the public manifest
and product lock are frozen. For locked real validation, use
`--require-product-lock --product-lock ...`. The runner verifies that the lock
matches the public manifest hash, has selected product metadata for every
`real_slc` target, and then passes locked Sentinel-1 product metadata into the
acquisition path.

The current real pipeline can enforce exactly one locked Sentinel-1 product per
target before any download is attempted. If a lock selects multiple products for
a target, the runner fails clearly before download. Regenerate the lock with
`sar_search.selection_count` set to `1`, or refactor acquisition/processing for
multi-product locked execution.

The preflight inventory and product lock include `estimated_download_size_mb` and
`estimated_download_size_gb` fields when ASF product sizes are available. Compare
those estimates against `python geoanomaly.py health --json --skip-gpu` disk
metadata before allowing real execution. Missing size metadata should be treated
as unknown disk risk, not as zero size.

Real execution also requires an explicit confirmation flag in addition to
`--execute-real`. This prevents accidental downloads or long PINN training from a
copy-pasted command. Use `--confirm-real-downloads-and-training` only after the
public manifest, approved parameters, product lock, selected-product size
estimate, free disk space, and no-synthetic-fallback policy have been reviewed.

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --output-dir data/blind_validation/holdout_real_run --execute-real --confirm-real-downloads-and-training --require-product-lock --product-lock data/blind_validation/holdout_inventory/product_lock.json
```

For a locked campaign, include the registry gate as well:

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --campaign-registry data/blind_validation/campaign_registry_locked.json --require-locked-campaign-registry --output-dir data/blind_validation/holdout_real_run --execute-real --confirm-real-downloads-and-training --require-product-lock --product-lock data/blind_validation/holdout_inventory/product_lock.json
```

Synthetic fallback remains disabled by default for real validation. Only use it
for explicit positive-control experiments, not blind real-world validation. It
cannot be combined with `--require-product-lock`:

```bash
python blind_validation.py run --manifest validation_examples/public_manifest_real_holdout.json --output-dir data/blind_validation/synthetic_control --execute-real --confirm-real-downloads-and-training --allow-synthetic-fallback
```

Run manifests now include reproducibility metadata: public manifest hash,
campaign registry hash/ID when present, product lock hash when present, code
fingerprints for key modules, command arguments with secret-looking values
redacted, Python/platform metadata, random seed policy, and explicit
no-synthetic-fallback status.

## Score After Withheld Labels Are Supplied

Scoring is the only stage that reads withheld labels. Run it after candidate
outputs are frozen and after the label custodian supplies a private labels file:

```bash
python blind_validation.py score --run-manifest data/blind_validation/holdout_real_run/run_manifest.json --labels path\to\private_withheld_labels.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --require-approved-parameters --output data/blind_validation/holdout_score.json
```

The repository template is only a schema scaffold. Real private withheld labels
must be user-supplied by an independent validation custodian after runner outputs
are frozen. Do not derive withheld labels from model outputs, public manifests, or
candidate CSVs, and do not commit private holdout label files.

## Aggregate Baseline Reports

After one or more score JSON files exist, generate a structured baseline report
and concise text summary:

```bash
python blind_validation.py baseline-report data/blind_validation/holdout_score.json --output-json data/blind_validation/holdout_baseline_report.json --output-text data/blind_validation/holdout_baseline_report.txt
```

Equivalent alias:

```bash
python blind_validation.py summarize-scores --score data/blind_validation/holdout_score.json --output-json data/blind_validation/holdout_baseline_report.json
```

The baseline report aggregates precision/recall-like metrics, hit rate by split,
site class, provider, provider label, and comparison arm, false-positive
candidates per area, rank-of-first-hit distribution, localization/depth error
summaries, confidence calibration bins where candidate confidence scores are
present, and metadata counts for provider/comparison-arm fields.

When score JSONs were produced from fixture or calibration ablation runs that
recorded `robustness_variant` or `ablation_variant` metadata, generate a
robustness summary without loading labels again:

```bash
python blind_validation.py robustness-summary data/blind_validation/score_variant_a.json data/blind_validation/score_variant_b.json --output data/blind_validation/robustness_summary.json
```

The robustness summary groups existing score metrics by variant ID, variant
family, variant arm, provider, comparison arm, null-baseline group, and repeat
stability group. It is an aggregation of already-written score JSONs; it does not
run SAR processing, start training, inspect public manifests for truth, or load
withheld labels.

## Package a Validation Report

After the score JSON and optional baseline report exist, package the evidence for
review. The package command copies and redacts supplied text/JSON artifacts,
writes `validation_summary.json`, writes `file_hash_manifest.json`, and writes a
human-readable `methods_limitations_claim_boundary.md` report. It records both
source artifact hashes before redaction and packaged artifact hashes after
redaction.

Minimal required package:

```bash
python blind_validation.py package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --output-dir data/blind_validation/fixture_report_package
```

Full package with a parameter set, no-download SAR inventory, product lock,
baseline reports, command logs, and notes:

```bash
python blind_validation.py package-report --public-manifest validation_examples/public_manifest_real_holdout.json --parameter-set data/blind_validation/parameters/broad_region_quick_v1_approved.json --sar-inventory data/blind_validation/holdout_inventory/sar_inventory.json --product-lock data/blind_validation/holdout_inventory/product_lock.json --run-manifest data/blind_validation/holdout_real_run/run_manifest.json --score-json data/blind_validation/holdout_score.json --baseline-json data/blind_validation/holdout_baseline_report.json --baseline-text data/blind_validation/holdout_baseline_report.txt --command-log data/blind_validation/command_log.txt --notes "Custodian supplied labels after run outputs were frozen." --output-dir data/blind_validation/holdout_report_package
```

Equivalent top-level wrapper:

```bash
python geoanomaly.py validation package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --output-dir data/blind_validation/fixture_report_package
```

Report packages include explicit evidence labels such as
`EVIDENCE_PUBLIC_NO_LABEL_INPUT`, `EVIDENCE_FROZEN_CANDIDATE_RUN_MANIFEST`,
`EVIDENCE_WITHHELD_LABEL_SCORE_OUTPUT`, and
`EVIDENCE_AGGREGATED_BASELINE_METRICS_JSON`. The claim boundary is repeated in
both JSON and text outputs: results are candidate detections unless independently
field-verified; a validation report package is not a confirmed-void, hazard, or
drill-target claim.

Do not pass `.env`, private tokens, or raw credential logs to the package command.
The package command redacts secret-like keys, token/password/bearer strings, and
current environment secret values from copied text and JSON artifacts, but the
preferred workflow is still to avoid collecting secrets in validation artifacts in
the first place.

Exact local fixture chain from manifest validation through report package:

```bash
python blind_validation.py validate-public --manifest validation_examples/public_manifest_fixture.json
python blind_validation.py run --manifest validation_examples/public_manifest_fixture.json --output-dir data/blind_validation/fixture_run
python blind_validation.py score --run-manifest data/blind_validation/fixture_run/run_manifest.json --labels validation_examples/withheld_labels_fixture.json --output data/blind_validation/fixture_score.json
python blind_validation.py baseline-report data/blind_validation/fixture_score.json --output-json data/blind_validation/fixture_baseline_report.json --output-text data/blind_validation/fixture_baseline_report.txt
python blind_validation.py package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --baseline-json data/blind_validation/fixture_baseline_report.json --baseline-text data/blind_validation/fixture_baseline_report.txt --output-dir data/blind_validation/fixture_report_package
```

## Metrics Written to Score Reports

The scorer writes structured JSON with deterministic target-level and summary
metrics, including:

- positive-site hit/miss
- negative-site false-positive flags
- candidates per square kilometer when `area_km2` is available
- rank of first hit
- horizontal distance error in meters
- depth error when the withheld label has expected depth and candidate depth is
  available
- site precision-like and recall-like summaries
- candidate precision-like and known-void label recall-like summaries
- confidence and rank summaries
- per-target scoring assumptions

## Candidate Location Fallbacks

Existing `detected_anomalies.csv` rows are supported. The scorer resolves
locations in this order:

1. `centroid_m` interpreted as meters relative to the public target center.
2. Explicit x/y meter fields, if present.
3. Candidate latitude/longitude converted around the public target center.
4. Pixel centroid plus domain width and grid dimensions, if available.
5. Public target center fallback when no usable location exists.

Fallbacks are recorded in `scoring_assumptions` and candidate summaries so lower
confidence georeferencing is visible in the score output.

## Independent Field Verification Evidence

Scored candidate detections are not field verification. Confirmation of a real
void, object, hazard, or drill target requires independent field evidence outside
the model workflow, such as survey-grade mapping, borehole logs, engineering
drawings with provenance, instrumented geophysics, or site-owner documentation.

Use `validation_examples/field_verification_template.json` as a private evidence
intake scaffold for post-score review. It is intentionally separate from public
runner manifests and private withheld labels. The template should be copied to a
private evidence repository, completed by a field-verification custodian, and
referenced in human review notes only after redaction. Do not fabricate field
evidence and do not treat a validation report package as independent field
evidence.

## Claims Boundary

Passing fixture tests proves that blind-run separation and deterministic scoring
work. It does not prove real-world field accuracy. Field claims require real
holdout sites, frozen outputs, withheld labels, and independent review of the
resulting score report.
