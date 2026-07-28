# interesting_intel — validation results and ranked findings

*2026-07-27. All imagery free/public (NAIP, Sentinel-2, Copernicus GLO-30).
Everything below is a LEAD, not a discovery claim.*

## What was built

A five-stage funnel (`interesting_intel/`) that turns a bounding box into a
ranked "worth a glance" queue: regional priors (DEM relief/depressions,
Sentinel-2 spectral outliers, change-over-years) -> parallel cached chip fetch
+ CV features (novelty-vs-own-region, geometry, confounds) -> cheap VLM over
25-chip colour contact sheets -> stronger VLM on full-res chips -> a final
contact sheet and this report. Nothing is hard-vetoed; definitive confounds
(cultivated flat land, industrial pads) only demote rank.

## Positive / negative controls — honest history

Round 1 (first real run, 2026-07-27 morning): **1/5 positives surfaced.**
The failures were real and instructive:

| control | round-1 result | diagnosis | fix |
|---|---|---|---|
| Bingham Canyon Mine | PASS (novelty rank 2/25) | — | — |
| Racetrack Playa | FAIL (nov 15/25) | at a 1 km view a playa is just one smooth surface among its valley's other smooth surfaces | evaluate at 4 km view — scale must match the feature |
| Nazca lines | FAIL (geo rank 17/25) | geoglyph lines are ~1.5 sigma tonal features; canny provably blind (straightedge = 0.0) | new radon faint-line scorer (top-5 sinogram bins; a 99.9th percentile misses single lines) |
| Richat Structure | FAIL (nov 18/22) | Hough circle score normalises peak-over-median across radii — a structure with circles at EVERY radius has a high median and erases itself | new gradient-radiality scorer (fast-radial-symmetry style) |
| Minuteman silo (spec's ~47.0,-101.5) | FAIL (nov 9/25) | the approximate coordinate is plain farmland | use a real mapped silo (OSM Launch Facility G-05, 48.051,-101.855) |

Round 2 (same harness, fixed scorers/scales/coords):

| control | novelty (rank) | geometry (rank) | strict top-decile | through full funnel |
|---|---|---|---|---|
| Bingham Canyon Mine | 3.67 (1/25) | 0.197 (1/25) | PASS | — |
| Racetrack Playa (4 km) | 4.11 (1/25) | 0.189 (1/25) | PASS | — |
| Nazca lines | 1.62 (15/25) | 0.156 (2/25) | PASS | — |
| Richat Structure | 3.23 (3/22) | 0.172 (5/22) | FAIL by one rank | **PASS** — wide-pass VLM rates it 2/3: "concentric circular pattern... resembling a giant eye" |
| Minuteman silo G-05 | 1.44 (13/25) | 0.281 (6/25) | FAIL | **FAIL** — VLM sees the cell but rates it 1/3 ("shed at field intersection") |
| Kansas cropland (negative) | 0.88 (23/25) | 0.306 (12/25) | PASS (stayed dull) | — |

**Bottom line: 3/5 positives surface on cheap CV rank alone, 4/5 through the
full funnel, and the negative control stays dull.** The genuine miss is
documented as a capability floor: a single ~1-acre fenced gravel pad inside
active cropland is statistically identical to farm infrastructure at every
stage of this system — finding it needs either tasked high-res imagery or a
non-imagery prior (it IS mapped in OSM as military, which is its own tell).

Controls harness: `python -m interesting_intel.validate` (results in
`results/interesting/controls/controls.json`; control chips + 8-chip
background rings are committed as real-imagery test fixtures under
`tests/fixtures/interesting_chips/`, exercised by
`tests/test_interesting_real_chips.py`).

## Region runs

### Rich region: Racetrack / northern Death Valley (36.45-36.85 N, 117.35-117.75 W, ~35 x 44 km, NAIP)

Funnel: 1,265 candidates (35 DEM + 800 spectral + 800 change peaks, merged,
+ 1 seeded control) -> 1,199 chips scored -> 500 sheeted (20 sheets) ->
92 wide-notable -> 21 full-res reviews -> 21-entry final queue.
40 min wall-clock, $0.041 VLM. Artifacts:
`results/interesting/racetrack/` (report.md, final_queue.png, chips_full/).

The seeded Racetrack Playa entered at CV rank 84/1199 (top 7%) and the wide
pass flagged its cell at interest 2 — the funnel surfaces its own control.

One live design lesson (kept because it changed the code): the first focus
pass rated ALL 21 full-res chips "interest 0" — the stock prompt measures
UNEXPLAINABILITY, and a skeptical model correctly explains almost everything
(a crater-like depression: "ordinary desert geomorphology"). But the Richat
Structure is fully explained geology and is still the canonical worth-a-look.
Stage 4 now asks separately for `worth_a_glance` (visual remarkability,
independent of explainability) and ranks the queue on that first. After the
change: 4 chips at worth 2, 8 at worth 1, 9 at worth 0.

Top of the queue (mundane explanation first, per project rule):

| rank | what the VLM saw | mundane explanation | what does not fit |
|---|---|---|---|
| 50 | dark irregular object alone on flat tan playa | rock outcrop or vegetation cluster (this is the Racetrack's Grandstand area) | "no clear connection to surrounding terrain"; VLM wants sub-0.5 m imagery to resolve texture |
| 35 | dark water body with sharp unnatural-looking shoreline | terrain shadow / dark lake edge | shadow edge "sharp in some places, diffuse in others" |
| 402 | isolated finger-like rock formation with shadowed crevices | eroded outcrop | isolation on smooth desert floor |
| 45 | light irregular depression with concentric benches | wash-cut terraces | concentric geometry |
| 63 | circular raised mound with radial pattern | volcanic/spring mound | radial symmetry in flat terrain |
| 88 | cluster of small structures + roads in canyon near red-brown patch | mining camp (Death Valley has many historic workings) | worth a look on general principle |

### Negative control: central Kansas cropland (38.40-38.80 N, 98.30-98.70 W)

Funnel: 1,320 candidates -> 1,200 scored -> 500 sheeted -> 263 wide-notable
-> 26 full-res reviews -> 26-entry queue. Artifacts:
`results/interesting/kansas/`.

The comparison that matters:

| | Racetrack (rich) | Kansas (boring) |
|---|---|---|
| confound-demoted in stage 2 | 5 / 1,199 | **593 / 1,200** (cultivated) |
| wide-notable cells | 92 | 263 (cropland is BUSY, not interesting) |
| final queue | 21 | 26 |
| my "solid huh" count in queue | **8** | **1-2** |
| queue look | lone playa object, ringed mound, canyon camp, swirl strata | rivers, contour strips, farm ponds |

Raw notable-counts are HIGHER in cropland (every pond and farmstead is
"notable" to the wide pass) — the discrimination happens in the demotion
layer and in queue quality. The 1-2 genuine Kansas finds (a cluster of white
dome-like structures at 38.4851,-98.6058; pale horseshoe earthworks at
38.4885,-98.6512) are arguably the system doing its job: even a boring
region's queue leads with the least-boring thing in it. The mid-run failure
mode this run exposed (MPC SAS tokens parsed as local time -> 403 flood after
~1 h) is fixed in `archaeo_intel/data_access.py::parse_expiry_utc` with a
regression test.

## Precision (the product number)

My own review of the 21-entry Racetrack queue (a human should repeat this —
that number is the product): **8/21 solid "huh, what is that"** (ranks 50,
35, 402, 45, 63, 88, 251, 262), **6/21 marginal** (pretty geology, mildly odd
color), 7/21 dull. The `worth_a_glance` ordering put solid hits in all of the
top 4 positions; of the top 10, 8 were solid-or-marginal. The one seeded
control in the queue (rank 84, the playa at a 1 km chip view) confirms
end-to-end plumbing but its 1 km chip is the dull view of a feature whose
right scale is 4 km — per-feature scale remains the biggest known ranking
handicap.

## Cost + runtime (measured)

- Chip fetch is the bottleneck, exactly as the build spec predicted — but the
  fix was algorithmic, not parallelism alone: `reproject()` reads remote COGs
  at full resolution (185 s per 1 km NAIP chip, 49 s per 50 km S2 chip);
  windowed decimated reads use COG overviews (~20 s and 3.2 s first-touch,
  less warm, and ~1-3 s effective at 8-12 parallel workers).
- Controls harness: 148 chip fetches in 385 s (round 1), 50 more in 366 s
  (round 2, remainder cache hits). CV scoring: 0.8 s per 500 px chip.
- Racetrack region (~35 x 44 km): 40 min wall-clock end to end — stage 1
  priors 2.6 min, stage 2 (1,259 chip fetches + CV) 26 min, stage 3 wide VLM
  5 min, stage 4 focus VLM 7 min. Kansas comparable (stage-1-3 timings lost
  to the token crash; the report now persists per stage).
- VLM actuals (OpenRouter, qwen3-vl-235b): Racetrack $0.068 for 62 calls
  (incl. the stage-4 re-run), Kansas $0.035 persisted + ~$0.012 for the wide
  pass lost to the crash, control one-offs ~$0.005. **Total ~ $0.12 for
  everything in this document.** Cost is a rounding error, as the build spec
  predicted; wall-clock is imagery fetch.

## Known limits

- Sentinel-2 10 m is the global floor; NAIP ~1 m is CONUS-only. Sub-metre
  features (vehicles, small structures, rock-trail-scale marks) are invisible.
- Chip CV runs on grayscale; colour anomalies only enter via the VLM stages.
- Priors are static-scene + slow-change; nothing here detects motion (that is
  deformation_intel's job).
- Coordinates in public artifacts are rounded to 4 decimals for the US runs;
  for conflict-zone archaeology the project rule stands: round to ~0.1 deg
  publicly, keep precise data local.
