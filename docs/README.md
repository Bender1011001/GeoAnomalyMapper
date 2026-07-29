# Documentation index

Start with the [project README](../README.md) — what the system is, the
ground-truth validation, and the quickstart.

## Read these to understand the system

| Doc | What it's for |
|---|---|
| **[CRITIQUE.md](CRITIQUE.md)** | Adversarial self-review: every known weakness in the project, with fix status, plus the publication-safety checklist (what may and may not be claimed). Read this before trusting anything else. |
| **[DISCOVERY_SOP.md](DISCOVERY_SOP.md)** | Binding governance for every detection capability: triage budgets, auto-explain before human review, provenance requirements, coordinate-redaction ethics, and the N=3 pre-registered kill switch. |
| **[VALIDATION_FIRST_WORKFLOW.md](VALIDATION_FIRST_WORKFLOW.md)** | Design of the blind-validation harness (`blind_validation.py` + `geoanomaly.py`): blind candidate generation kept separate from withheld-label scoring. |

## Read these to use a specific capability

| Doc | What it's for |
|---|---|
| **[CORONA.md](CORONA.md)** | Tutorial for `archaeo_intel/corona.py` — free 1960s ~2 m CORONA imagery, from quicklook to georeferenced GeoTIFF. The most reusable standalone module here. |
| **[SPEC_interesting_classifier.md](SPEC_interesting_classifier.md)** | Build spec for the general "worth a glance" ranker, including eight hard-won failure lessons worth reading even if you never run the code. |

## The evidence

**[notebook/](notebook/)** — the lab notebook. Append-only research record with
every experiment, including the dead ends, the retractions, and the null
results. This is where the README's "validated" and "failed, removed" claims
are cashed out.

## Package documentation (lives beside the code)

- [deformation_intel/](../deformation_intel/README.md) — the validated core
- [archaeo_intel/](../archaeo_intel/README.md) — archaeology surface-proxy channel
- [interesting_intel/](../interesting_intel/README.md) — general imagery ranker
- [tools/insar_prototype/](../tools/insar_prototype/README.md) — HyP3 fast-deformation channel

## Figures

`img/` holds the figures embedded in the project README. Regenerate them from
source data with `python tools/make_readme_figures.py`.
