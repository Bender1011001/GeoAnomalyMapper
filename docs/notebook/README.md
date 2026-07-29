# The lab notebook

**This is a research record, not documentation.** It is append-only, dated, and
deliberately includes everything that did not work. If you are trying to
understand what the system *does*, read the [project README](../../README.md)
instead — this directory is for anyone who wants to check the working.

Why it is public: the claims in the README ("validated", "failed, removed") are
only worth anything if the evidence behind them is inspectable. This is that
evidence.

## Start here

**[RESEARCH_TRACKS.md](RESEARCH_TRACKS.md)** — the main ledger (~2,200 lines).
It has a maintained status summary at the top; read that first. Every hunt,
sweep, and verdict since 2026-07-15 is recorded, including pre-registrations
written *before* results were known, and null results recorded with the same
detail as positive ones.

Representative entries, if you want a sample rather than the whole thing:

- The four-region arid sweep — 33,458 detections, 357 tiles, resolved to zero
  unexplained. Includes the coverage gaps, honestly labelled.
- Closure-phase disturbance detection — a validated ~6-week early-warning lead,
  *and* the retraction of an initial novelty claim that turned out to be wrong
  (closure-phase soil-moisture retrieval is a mature field; we said otherwise
  first, then corrected it).
- The Lake Cahuilla "lost ship" search — a null result, run anyway because a
  measured null beats an assumed one.

## Dated reports (historical records — do not edit)

| Report | Date | What it covers |
|---|---|---|
| [NATIONAL_SCAN_REPORT.md](NATIONAL_SCAN_REPORT.md) | 2026-07-11 | 12-AOI full-archive OPERA deformation scan, detector v5 |
| [WESTHUNT_BALIKH_REPORT.md](WESTHUNT_BALIKH_REPORT.md) | 2026-07-13 | Balikh–Khabur steppe survey: 35 verified candidates, 5 ring structures |
| [RESCAN_V2_REPORT.md](RESCAN_V2_REPORT.md) | 2026-07-14 | 33-tile rescan after two recall-bug fixes: 118 hits vs 15, 25/25 catalog-validated |
| [INTERESTING_FINDINGS.md](INTERESTING_FINDINGS.md) | ongoing | Output of the general "worth a glance" ranker |
| [closure_phase_writeup.md](closure_phase_writeup.md) | 2026-07 | Technical note on the closure-phase detector, with references |

`westhunt_verified.json` and `rescan_v2_results.json` are the machine-readable
results behind those reports.

**Coordinates in every public artifact here are rounded to ~0.1° (~11 km).**
Unprotected candidate sites in conflict zones must not be published as a precise
target list. Precise data stays local.

## experiment_records/

Raw JSON verdicts, including the **failed-approach control runs** — the
vibrometry Carlsbad-vs-barren-plains controls that killed that pipeline, and the
Wink OPERA validation that established the current one. These files are the
reason the README's claims table is allowed to say "backed by a ground-truth
experiment."

## What is not here

`data/research/` (gitignored, local only) holds candidate lists, per-tile sweep
outputs, and ~149 preserved experiment scripts referenced by name in the ledger.
They are excluded because they contain unredacted coordinates and are far too
large for a repository.
