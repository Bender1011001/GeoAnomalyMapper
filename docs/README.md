# docs/ — index

## Living documents (maintained)

- **[RESEARCH_TRACKS.md](RESEARCH_TRACKS.md)** — the research program lab
  notebook: charter, pre-registrations, append-only verdict ledger, and a
  maintained **status summary at the top** (read that first). Every hunt,
  track, and sweep verdict since 2026-07-15 is recorded here.
- **[DISCOVERY_SOP.md](DISCOVERY_SOP.md)** — binding governance for every
  detection capability: triage budgets, auto-explain before human review,
  provenance requirements, ethics/coordinate redaction, and the N=3
  pre-registered kill switch.
- **[CORONA.md](CORONA.md)** — tutorial for `archaeo_intel/corona.py`:
  free 1960s ~2 m CORONA imagery, from quicklook to georeferenced GeoTIFF.
- **[VALIDATION_FIRST_WORKFLOW.md](VALIDATION_FIRST_WORKFLOW.md)** — design
  of the blind validation harness (`blind_validation.py` + `geoanomaly.py`):
  blind candidate generation separated from withheld-label scoring.

Package documentation lives beside the code:
[../README.md](../README.md) (project),
[../CONTEXT.md](../CONTEXT.md) (current state + trap diary),
[../deformation_intel/README.md](../deformation_intel/README.md),
[../archaeo_intel/README.md](../archaeo_intel/README.md),
[../tools/insar_prototype/README.md](../tools/insar_prototype/README.md).

## Dated reports (historical records — do not edit)

- **[NATIONAL_SCAN_REPORT.md](NATIONAL_SCAN_REPORT.md)** (2026-07-11) —
  12-AOI full-archive OPERA deformation scan with detector v5.
- **[WESTHUNT_BALIKH_REPORT.md](WESTHUNT_BALIKH_REPORT.md)** (2026-07-13) —
  Balikh-Khabur steppe survey: 35 verified candidates, 5 ring structures
  (coordinates redacted to ~11 km).
- **[RESCAN_V2_REPORT.md](RESCAN_V2_REPORT.md)** (2026-07-14) — 33-tile
  rescan after the two recall-bug fixes: 118 hits vs 15, 25/25
  catalog-validated (coordinates redacted).
- `westhunt_verified.json`, `rescan_v2_results.json` — machine-readable
  (redacted) results behind those reports.
- **experiment_records/** — raw JSON verdicts, including the
  failed-approach control runs (vibrometry Carlsbad/plains controls) and the
  Wink OPERA validation; these files are why the README's claims table can
  say "backed by a ground-truth experiment".

## Where the rest of the evidence lives (local-only, gitignored)

`data/research/` holds candidate lists (`*_LOCAL.json`, coordinate-redaction
rule), per-tile sweep outputs, and `scripts/` — 149 preserved
experiment scripts referenced by name in the RESEARCH_TRACKS ledger.
