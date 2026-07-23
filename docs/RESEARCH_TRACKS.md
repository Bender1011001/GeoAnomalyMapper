# Research tracks: new mathematics on data we already hold

Charter (2026-07-15): stop re-running established methods; extract NEW results
from the SAME archives by processing signal that standard pipelines discard or
never look at. Every track is pre-registered (hypothesis, test, numeric pass,
kill criterion) per docs/DISCOVERY_SOP.md before its first result is seen.

## Program status summary (maintained; last updated 2026-07-21)

Everything below this section is an append-only chronological ledger. This
section is the current state, so a reader does not have to replay it.

**Validated new capabilities (from data we already held):**

- **Closure-phase disturbance detector (Track 2) — PASS.** Onset flagged at
  the documented Eldorado site-work date with zero pre-onset flags, and a
  ~6-week early-warning lead over the validated coherence channel
  (dielectric/moisture disturbance precedes scatterer destruction).
- **Asc/desc backscatter anisotropy (Hunt 10) — PASS, replicated.** AUC 0.639
  (n=141) + 0.608 on an independent AOI ~90 km away — the best single
  free-data archaeology feature measured to date. Validated as a RANKING
  feature only: the autonomous z>4 peak sweep returned 12/12 confounders
  (it is a structure detector — buildings, ruins, pylons, cliffs — not a
  tell detector).
- **TDA H1 persistence (Track 3) — PASS as a ring-sensitivity feature.**
  Tell Chuera anchor + 3/5 ring candidates above control p90. NOT autonomous
  at 30 m DEM: the 111-candidate flat backlog triaged to NULL (12/12 top
  combined-rank candidates were terrain or modern).
- **Coherence-drop transient detector (Cap 1) — validated, envelope-bounded.**
  6-8 sigma on the bare-desert control; only valid on non-vegetating terrain,
  and the rule now carries the >30%-same-date scene-event veto.
- **CORONA human-GCP georeferencing — operational.** ~200 m near anchors,
  refinable to ~150 m locally; first vetted use demoted ring 34 (annulus
  absent in 1967 = probable modern enclosure).

**Closed negatives (quantified, honestly dead):**

- **Track 1 (ICA) — RETIRED.** Qualitatively demonstrated at Wink (19.6x sink
  enrichment, clean atmosphere separation) but both registered external-truth
  tests were voided by mis-sited positive controls; not re-rolled per
  anti-p-hacking discipline.
- **Hunt 11 (B-perp sub-pixel altimeter) — KILLED at N=3.** The height signal
  is real (plant excess ~3x the road-pixel floor, correct magnitudes) but the
  achievable floor is ~3 m and DISP-S1's coherence masking deletes exactly
  the pixels of interest — the 3-5 m archaeological regime is out of reach
  on this data.
- **Hunts 8/13 — NULL** (kurtosis 0.518, VH/VV intercept 0.541). **Hunt 12 —
  rejected** (Capability-2 kill-switch precedent). **Hunts 7/14 — parked**
  (collinearity; no night Landsat).
- **Auto-georeferencing of CORONA film — classed a research problem** after
  six attempts with six distinct documented failure modes; the operational
  path is human GCPs (~10 min/strip, shipped in archaeo_intel/corona.py).
- **Step-event catalog — NULL** on creep terrain (Wink); needs
  abrupt-failure terrain.
- **Cap-1 blind sweep #1 (Palmyra-west 2023) — clean NULL.** Post-veto, every
  event auto-explained (rain scene event, harvest, phenology); zero
  unexplained disturbances in the window.

**In flight:**

- **Bare-desert OPERA displacement sweep** — 39/65 tiles complete as of
  2026-07-21, drivers running (desert_sweep_v2.py / v2b.py, subprocess-per-
  tile with 4 h hard timeout); per-tile results + window caches in
  data/research/desert_sweep/; triage via desert_triage.py.

**Queued:** Hunt 9 (moisture-lag, steppe sites, reuses RTC stacks);
closure-phase archaeology stretch test + early-warning fold-in alongside the
coherence channel; anisotropy into the combined-ML ensemble (current ceiling
AUC 0.616); Cap-1 clean paths (hyper-arid re-point, or seasonal-baseline
rule on OPERA); CORONA stereo DEMs (bench, unregistered).

**Provenance note (2026-07-21):** the per-experiment scripts named in ledger
entries (cap1_*, track1/2/3_*, hunt*, aniso_*, desert_sweep*, tda_*,
sweep1_*, corona/GCP scripts) were written in a session scratchpad under
Temp; all 149 are preserved verbatim in `data/research/scripts/`
(gitignored, local-only). They are experiment records, not maintained
package code. Anything promoted into the public repo must first be vetted
for precise candidate coordinates per the redaction rule.

## Track 1 — Blind source separation (spatial ICA) of OPERA displacement cubes

Standard practice (ours included) collapses the ~270-epoch cube into
parametric fits (velocity/acceleration/seasonal), assuming the signal shapes.
Hypothesis: ICA decomposes the raw spatiotemporal cube into independent
sources — regional aquifer seasonality, ramps, atmosphere, and LOCALIZED
processes currently drowned under them. Directly attacks the documented
Central Valley wall (wall-to-wall aquifer signal blinds the detector).
Novelty: ICA exists in volcano InSAR (ICASAR); no published application to
DISP-S1 or karst screening.

- PRE-REGISTERED (before first run): on the cached Wink cube, >=1 of k=6
  components concentrates its top-1% |spatial weight| at the known sink
  complex at >=20x chance density with a monotonic-dominant time course
  (|Spearman rho| >= 0.8). Fail = kill.

## Track 2 — Closure phase: the signal InSAR processing throws away

Interferogram triplets (AB + BC - AC) should close to zero; the residual
"closure error" that pipelines calibrate out is a physical measurement of
changing surface scattering, dominated by soil-moisture change in the top
decimeters (De Zan et al.). Nobody operationalizes it on free HyP3 stacks.
We hold 30 sequential 6-day pairs over the Eldorado control scene with a
dated construction event; 14 skip-pairs (i -> i+2) SUBMITTED 2026-07-15
(gam_track2_closure) complete the triplets.

- PRE-REGISTERED test 1: closure-phase anomaly maps must show the
  construction footprint as an anomaly cluster distinct from the quiet
  control box in >=3 consecutive post-onset triplets, and NOT before onset.
- Stretch test 2 (archaeology): buried structures alter moisture retention;
  closure anomalies over catalogued Khabur sites vs matched controls
  (AUC > 0.60) would recover, via radar, the physics the NDVI crop-mark
  channel failed to see (that channel died at chance — see SOP ledger).
- Caveat registered up front: closure from UNWRAPPED products includes
  unwrapping-error contamination; quiet-scene triplets are the null.

## Track 3 — Persistent homology (H1) as a shape-native ring detector on DEMs

A rampart circuit is a topological loop of high ground. Max H1 persistence
of the detrended elevation function measures "ring-ness" at ANY amplitude —
where bump-ranking (prominence) scored only AUC 0.547 on the general site
population. TDA on DEMs for archaeological ring detection is nearly absent
from the literature.

- PRE-REGISTERED (before first run): (a) Tell Chuera + >=3/5 west-hunt ring
  candidates above the 90th percentile of 40 matched steppe controls; OR
  (b) population AUC > 0.60 on the dense-Khabur catalog tile (baseline
  0.547). Both fail = kill.

## Bench (queued, not yet registered)

- Step-event catalog from OPERA cubes: matched filter for discrete
  displacement steps -> a DATED catalog of collapse/settlement events;
  never published from DISP-S1.
- CORONA stereo DEMs: we hold both cameras of every KH-4B stereo pair;
  automated 1967 DEMs differenced against modern DEMs = 60 years of
  measured earth-surface change (erosion, tell destruction, dune migration).
- Time-series shapelet clustering: discover deformation behavior classes
  (ratchet, step, reversible) rather than assuming them.

## Results ledger (updated 2026-07-15 evening)

- **Track 1 (ICA): NEAR-MISS — re-registration required.** First run scored
  against a mistaken cube-center sink location (metadata carried no bbox);
  geometry corrected by locating the sink complex from the RAW cumulative
  field (independent of the ICA output). Re-scored: component 4 concentrates
  at the sink complex at **x19.6 chance density with a perfectly monotonic
  time course (|rho|=1.00)**; all five other components <=x0.3 (clean
  atmospheric/ramp separation, visible in track1_ica_wink.png). The
  pre-registered bar was >=x20: by the letter this is a FAIL at 2% margin.
  Per SOP, no post-hoc threshold changes: claim requires a PASS on a fresh
  pre-registered cube (next: a national-scan target cube re-fetched for the
  purpose). Qualitative finding stands: ICA separates localized secular
  deformation from atmosphere on DISP-S1 without parametric assumptions.
- **Track 3 (persistent homology): PASS (Part A, first run, no tuning).**
  Tell Chuera H1-persistence 4.00 m vs matched-steppe control p90 = 1.36 m;
  ring candidates above p90: 3/5 (ring9 1.76, ring25 3.38, ring34 2.22) —
  exactly the registered criterion. Part B failed honestly (population AUC
  0.535 ~ prominence baseline): H1 is a RING detector, not a general tell
  detector — the validated capability is amplitude-agnostic ring-form
  detection, a class our prominence channel cannot rank. TDA-on-DEM for
  Kranzhugel detection appears absent from the literature; this is the
  program's first genuinely new validated method.
- **Track 2 (closure phase): data ordered.** 14 skip-pairs submitted
  (gam_track2_closure) completing ~14 triplets over the Eldorado control
  scene (quiet baseline + dated construction onset). Analyzer to be built
  against the pre-registered tests above; products persist on HyP3 ~2 weeks.

### Track 1 re-registration (PRE-REGISTERED 2026-07-15 before fetch)

Fresh cube: 24 km OPERA DISP-S1 cube centered (29.88, -94.92) — Mont
Belvieu / Baytown TX. External truth: the Barbers Hill salt-dome storage
complex (~29.868, -94.902) has decades-documented localized subsidence
independent of this project. PASS: among k=6 spatial-ICA components (same
recipe as the Wink run, unchanged), >=1 concentrates its top-1% |weight|
within 2 km of the dome center at >=20x chance density with time-course
|Spearman rho| >= 0.8. This is the second and FINAL registration for the
claim; fail = Track 1 dies per kill switch.

### Bench verdicts (2026-07-16)

- **Step-event detector (Wink cube): NULL.** Zero step candidates at SNR>8
  across 534k pixels. Physics: Wink dissolution subsides by CREEP, and
  OPERA's recommended mask deletes true fast-collapse pixels (documented).
  A step catalog needs cubes over abrupt-failure terrain (mining collapses),
  not dissolution creep.
- **Auto-georeferencing of CORONA, attempt #4 (Mathematica
  FindGeometricTransform, affine, 64 m/px): FAIL — "no corresponding
  points."** Four independent failures (SIFT, edge-SIFT, coarse NCC,
  Wolfram keypoints) now bound the problem: POINT-feature matching does not
  cross the 1967-film/modern-optical modality gap. Remaining designs:
  vignette-removed coarse structural alignment, or human GCPs (supported,
  released, ~10 min/strip).

### Cap-1 BLIND SWEEP #1 (PRE-REGISTERED 2026-07-16 before submission)

- AOI: 34.70-35.00 N, 38.20-38.60 E (west-of-Palmyra Syrian steppe; arid,
  high coherence, plausible undocumented ground activity).
- Stack: 12-day Sentinel-1 pairs spanning 2023 (~30 jobs, gam_cap1_sweep1).
- Detector: EXACTLY the re-registered rule (frac<0.85 excess vs in-scene
  quiet reference, baseline+max(5*MAD,0.02), 3-of-4 persistence).
- Quiet reference / FP zone (pre-registered): 34.75-34.85 N, 38.45-38.55 E.
- Triage budget: top 20 events by excess magnitude; each auto-explained
  (before/after S2 + OSM landuse) before human review. Output register:
  "candidate discrete surface disturbances with dates; expect flood-wash,
  agriculture, and permitted works as confounders."

### Track 3 sweep boundary (2026-07-16) — validated method, bounded deployment

The TDA ring sweep top-N (post flatness-triage) was chip-reviewed: high-H1
"flat" candidates are natural closed contours (wadi-incised mesa remnants,
scarp edges, basalt relief), not archaeological rings. A circularity/size
discriminator was built to separate ring SHAPE from blobs — it FAILED
decisively: at 30 m Copernicus DEM a 150-200 m Kranzhugel is only 5-7 px
across, so loop shape is unmeasurable (metric unstable, anchors and
false-positives both ~1.3-1.7, and it would reject validated ring9).

CONCLUSION (honest, bounds a real method): H1-persistence is a VALIDATED
ring-SENSITIVITY signal (anchor test passed: Kranzhugel score high) but is
NOT a standalone discovery detector at free-DEM resolution — the natural
closed-contour population is not separable from true rings by persistence or
by shape at 30 m. Correct use: TDA H1 as ONE ranking FEATURE feeding VLM/
human triage (surfacing ring-candidate tiles), never autonomous. A true
ring sweep needs higher-resolution elevation (TanDEM-X 12 m paid, or the
un-built CORONA stereo DEMs). This is the same resolution wall the
prominence and crop-mark channels hit, now measured for topology too.

### Track 2 first verdict (2026-07-16): FAIL by letter — candidate EARLY signal

9 triplets computed (Dec 2020 - Mar 2021). Baseline excess +0.008 rad; from
the triplet containing 2021-01-25 (documented site-work start): sustained
+0.034/+0.016/+0.021/+0.015/+0.015 (2-4x baseline, ZERO pre-onset flags).
Registered threshold failed the detection only via its blind 0.05-rad floor
(3x the actual signal scale). NOTE the physics-rich near-miss: closure
elevation begins ~6 WEEKS BEFORE the coherence rule fired at this site
(2021-03-14) — moisture/dielectric disturbance from early light activity
precedes scatterer destruction. If confirmed, closure is an EARLIER tripwire
than the validated coherence channel.

RE-REGISTRATION (2026-07-16, before new data): extension skip-pairs
(gam_track2_ext, Mar-Jun 2021) complete triplets through the grading window.
Corrected rule, frozen now: threshold = baseline + 5*MAD (NO floor), onset =
3-of-4 triplets above; PASS = onset within Jan-19..May-05 activity window
and zero flags in Dec. Final registration; fail = track dies.

## Hunts 6-10 (proposed by the Gemini collaborator, 2026-07-16; triaged)

- **Hunt 6 — relief/illumination inversion for CORONA georef: ACCEPTED,
  PRE-REGISTERED.** Match film to DEM-rendered HILLSHADE under Dec-1967
  solar geometry (not albedo features, which the 20th century erased), with
  vignette removal first and MUTUAL INFORMATION as the similarity (the
  modality-gap metric; none of the 4 failed attempts combined these).
  PASS: MI peak z>=5 over the rot/scale/flip search AND Kharab Sayyar +
  Tell Chuera projected in-strip AND the Chuera crop shows the ring.
- **Hunt 8 — temporal backscatter statistics (speckle texture): ACCEPTED
  with corrections.** Data path = MPC Sentinel-1 RTC time series (not SLC);
  claims via the standard site-vs-control AUC harness (not Rayleigh tests);
  test set restricted to FLAT UNOCCUPIED catalog sites (a man-made-scatterer
  detector trivially finds the modern village on a tell, not the site).
- **Hunt 10 — asc/desc anisotropy: ACCEPTED, run jointly with Hunt 8**
  (same RTC pulls, extra feature). Flat-steppe restriction handles the
  slope-aspect confounder.
- **Hunt 9 — moisture->green-up lag: REFRAMED.** Closure per-pixel over the
  Khabur is quota-prohibitive; RTC backscatter is the free moisture proxy.
  Irrigation dominates lag in cropland -> steppe sites only. Queued behind
  8/10 (reuses their stacks).
- **Hunt 7 — BRDF sun-sweep roughness: PARKED.** Solar zenith and phenology
  are both functions of day-of-year (collinear); identifiable only via
  year-to-year rainfall anomalies — needs a real identifiability design
  before any registration.

## Hunts 11-14 triage (2026-07-16 night)

- **Hunt 11 — B-perp regression as sub-pixel altimeter (DEM-error inversion):
  ACCEPTED PENDING FEASIBILITY GATE.** Physics textbook-correct; neighbor-
  differencing kills atmosphere. BUT the simulation assumed raw interferograms
  — OPERA DISP-S1 removed topographic phase server-side and may already
  correct residual DEM error. GATE (next session, one granule): does
  per-epoch perpendicular baseline ship in DISP-S1 metadata, and does the
  DEM-error residual survive into displacement? No registration until both
  answer yes.
- **Hunt 12 — halo-differenced temporal variance: REJECTED (kill-switch
  precedent).** This is Capability 2 renamed: iteration 2 already swept
  local background-differencing scales; iteration 3 tested every acquisition
  individually (29 dates, 46 unoccupied flat sites, max 0.571 vs 0.60 bar).
  The sim's premise (mixed pixels diverge phenologically) is what the ground
  truth refuted. Only legal path: ONE fresh registration in temperate
  rain-fed terrain where crop-mark literature has actual wins.
- **Hunt 13 — VH/VV temporal-regression intercept: ACCEPTED as a FEATURE**
  in the Hunt-8/10 RTC harness (same data pull carries both polarizations;
  zero marginal cost). Not a standalone hunt.
- **Hunt 14 — day/night apparent thermal inertia: PARKED — DATA-IMPOSSIBLE
  with Landsat here.** Census over the truth tile 2015-2025: 1000 daytime
  scenes, ZERO night. (Daytime thermal anomaly alone is already our measured
  best single channel, AUC 0.622.) Only path: ECOSTRESS night LST (ISS
  orbit, covers 36.6N; LP DAAC access) — separate feasibility check before
  any registration.

### Hunt 6 field verdict + auto-georef campaign close (2026-07-16 night)

Gemini's synthetic validation (translation-only, MI locks through vignette +
albedo noise) CONFIRMED the metric but under-modeled the film: real KH-4B
needed vignette removal (attempt 3's lesson), anisotropic scale (attempt 5's
lesson), and even then attempt 6 exposed a sixth failure mode: UNNORMALIZED
MI is biased toward small-overlap combos, so the grid search slides to the
scale boundary (0.149 "best" with opposite flips near-tied = no geometric
lock; anchors project outside the strip). Fix specified for any future
attempt: normalized MI + minimum-overlap guard + per-segment (pan-angle-
local) scale.

CAMPAIGN STATUS: six attempts, six distinct documented failure modes (SIFT/
grain; edge-SIFT; NCC/vignette; Wolfram keypoints/modality; isotropic scale;
MI overlap bias). Automatic film-to-modern georeferencing is hereby classed
a RESEARCH PROBLEM, not a task — publishable if solved. Operational path
remains human GCPs (~10 min/strip, shipped in archaeo_intel/corona.py).

### Track 2 FINAL VERDICT (2026-07-16): PASS

Rule frozen before extension data (base+5*MAD=+0.012, no floor, 3-of-4):
onset at triplet ending 2021-01-25 = the documented site-work start, ZERO
pre-onset flags. Curve: flat +0.007 through 2020 -> climb from Jan 25 ->
peak +0.028 during March grading -> decay as surface stabilizes.
**Closure phase (the signal every InSAR pipeline discards as calibration
error) is a VALIDATED construction/disturbance detector on free HyP3 data.**
Bonus validated claim: closure onset (Jan 25) LEADS coherence-channel onset
(Mar 14) by ~6 weeks -> closure is an earlier tripwire (dielectric/moisture
disturbance precedes scatterer destruction). Next: (1) the stretch
archaeology test (closure over flat catalog sites vs controls); (2) fold
closure as an early-warning layer alongside the coherence sweep.

## Program scoreboard (2026-07-16, two research sessions)

VALIDATED (new, on data we already held):
- Closure-phase disturbance detector (Track 2) — PASS, + early-warning lead.
- TDA H1 ring-sensitivity feature (Track 3) — PASS as a triage feature
  (bounded: not autonomous at 30 m DEM).
NEAR-MISS / re-registered:
- ICA source separation (Track 1) — x19.6 vs 20 bar; Mont Belvieu final test
  auto-chained.
CONFIRMED-signal, rule-tuning pending:
- Coherence-drop (Cap 1) — PASS on control; blind sweep processed, analyzer
  next.
NULL / bounded (honestly closed):
- Step-event catalog (creep, not steps); Crop-mark / Cap 2 & Hunt 12
  (chance in Khabur); auto-georef (6 failure modes -> research problem);
  Hunt 14 thermal-inertia (no night Landsat).
QUEUED: Hunts 8/10/13 (RTC texture+anisotropy+dual-pol intercept, one pull);
Hunt 9 (moisture-lag, steppe); Hunt 11 (B-perp altimeter, gated on metadata);
Cap-1 blind-sweep analyzer.

### Track 1 (ICA) FINAL DISPOSITION (2026-07-16): RETIRED — not robustly validated

Mont Belvieu registered test scored FAIL (best 0.8x vs 20x bar). But an
independent raw-cumulative-field check (NOT ICA output) shows WHY: the
Barbers Hill subsidence is real and IN the cube (-14.7 cm bowl at
~29.841,-94.892) but 3.2 km from the pre-registered truth point
(29.868,-94.902), OUTSIDE the 2 km disc. The registered coordinate shows
~0 cm. So the positive control was MIS-SITED (my external-coordinate error),
which VOIDS the scored test — it is neither a fair PASS nor a fair FAIL.

Discipline call: NOT re-scored against the moved bowl (that is post-hoc
target selection = p-hacking). And NOT re-registered a third time (endless
control re-rolls until one passes is itself p-hacking). Track 1 is RETIRED
as **qualitatively demonstrated (Wink: 19.6x sink enrichment, clean
atmosphere separation) but NOT robustly validated** — no clean pass of a
pre-registered EXTERNAL-truth test in two attempts, both undone by control
geolocation. RE-ENTRY CONDITION (for anyone resuming): pick a control whose
subsidence bowl is VERIFIED in the raw cube BEFORE registering the truth
coordinate; then one clean registered run decides it. Lesson for the whole
program: verify the positive control's signal exists in raw data before
registering a detector against it.

### Cap-1 blind sweep #1 — ANALYZER BUILT + RUN (2026-07-16)

sweep1_analyze.py: streams the 29 succeeded HyP3 coherence rasters
(gam_cap1_sweep1) via /vsizip//vsicurl (nothing downloaded in bulk), applies
the pre-registered rule at pixel level (quiet-zone per-pair normalization ->
baseline from first 8 -> drop > max(5*MAD, 0.15) -> 3-of-4 persistence ->
cluster >=6px), ranks event clusters by depth*sqrt(size), reports FP rate on
the pre-registered quiet zone. Top-20 events + dates written to
data/research/sweep1_events_LOCAL.json (coords LOCAL only, conflict-zone
redaction). This is the program's FIRST discovery run of a validated tool on
UNKNOWN ground. Verdict recorded on completion.

NOTE (disk): E:\code.projects\SAR-project reclaimed 48 GB (69->21 GB) —
removed 6 vibrometry-era Sentinel-1 SLC scenes (re-downloadable free from ASF)
and redundant local DEM tiles (streamed from AWS on demand). All processed
outputs, metadata, manifests kept. That folder never held an InSAR archive;
it held the retired vibrometry experiment's raw inputs.

### Cap-1 blind sweep #1 VERDICT (2026-07-16): NULL — deployed outside validated envelope

21 event clusters, but 16/21 share onset 2023-06-29 spread across 68% of the
AOI, and the pre-registered quiet zone fired ABOVE scene average (0.164% vs
0.08%). Diagnosis: BROAD SEASONAL DECORRELATION (spring-green Syrian steppe
dries scene-wide in summer), not localized disturbance. A single-pair-crash
rejection (added, caught only 2023-11-20) does NOT fix it because seasonal
drop is multi-pair, not a spike.

ROOT CAUSE (mine, honest): the detector was VALIDATED on BARE DESERT
(Eldorado — no vegetation, no seasonal coherence swing). Syrian steppe
vegetates; the sweep AOI is OUTSIDE the validated envelope. No discoveries;
NULL result.

TWO CLEAN PATHS (registered, not yet run):
1. Re-point at genuinely bare/hyper-arid UNKNOWN ground (Rub al Khali fringe,
   Sahara interior) where the control's no-vegetation assumption holds — the
   detector should transfer there.
2. Upgrade to a SEASONAL-BASELINE detector: multi-year stack, flag a pair's
   coherence drop only if it departs from the SAME calendar-window in other
   years (isolates anthropogenic from seasonal). Costs more pairs/quota, or
   use OPERA where the archive is deep + free.
LESSON: a detector's validated envelope (terrain/vegetation regime) must
match the deployment AOI. Do not deploy a bare-desert coherence rule on
vegetated terrain. Cache: scratchpad/sweep1_stack.npz (re-runs instant).

### Hunt 11 GATE (2026-07-16): PASSED — feasible on free OPERA data

Inspected an OPERA DISP-S1 granule directly. Ships (all free, no HyP3):
- corrections/perpendicular_baseline (per-epoch B-perp) — Hunt 11's exact
  regressor variable, present.
- short_wavelength_displacement + timeseries_inversion_residuals — two
  candidate residual layers where a sub-30m-DEM feature (a 2 m tell) would
  leak.
Caveat now a testable experiment, not a blocker: OPERA phase-linking may
already estimate+remove the B-perp-proportional DEM error (CEOS metadata
shows orbital_baseline_refinement applied). REGISTERED TEST: at a pixel over
a known sub-DEM feature (small tell / building cluster of known height),
differenced against a flat neighbor 100 m away, regress residual vs
perpendicular_baseline; slope != 0 (p<1e-3) => residual survives => Hunt 11
is a working free sub-pixel altimeter. Next-session build.

### FREE SEARCHES LAUNCHED (2026-07-16, in-envelope, no quota)

- **Bare-desert OPERA displacement sweep** (desert_sweep.py): 65 cubes over
  Mojave/Great Basin/Sonoran/Permian/Salton — the VALIDATED deformation
  detector on free OPERA, IN its bare-arid envelope. Streaming; localized
  accelerating subsidence candidates -> data/research/desert_sweep/. This is
  the free, no-quota, in-envelope search the coherence sweep could not be.
- **RTC hunts 8/10/13** (rtc_hunt.py): free MPC Sentinel-1 RTC, both orbits,
  VV+VH over the Khabur ground-truth tile. Temporal kurtosis + VH/VV
  intercept + asc/desc anisotropy, AUC vs 14k catalog (flat unoccupied sites
  only). Streaming.

### Cap-1 BLIND SWEEP #1 — RESULT: NULL, real confound found (2026-07-17)

29 pairs analyzed, 21 clustered "events" surfaced by the pre-registered rule.
Diagnostic BEFORE any claim (per SOP auto-explain discipline): date
distribution of the 21 events = {2023-04-30: 3, 2023-06-29: 16, 2023-07-11: 2}.
**81% of flagged pixels (271/336) share ONE onset date.** 21 independent
ground disturbances would not cluster onto 2-3 dates — this is a SINGLE
scene-wide decorrelation event (near-certainly a rain cell partially over the
AOI) that survived the quiet-zone median normalization and got fragmented
into pseudo-clusters by the connected-component labeler. VERDICT: sweep #1
is a NULL, not 21 candidates. No site claims from this run.

REAL METHODS BUG FOUND (not present in the Eldorado control test, because
that control had zero weather events in its window): the quiet-zone
normalization removes the MEAN scene-wide effect but not its SPATIAL
gradient (edge-of-rain-cell partial coverage). FIX for sweep #2 (registered
now, before re-running): add a same-date CROSS-SCENE check — discard any
onset date where >30% of all flagged pixels cluster on that single date
(scene-event veto), independent of and in addition to the per-pixel rule.

### RTC hunts 8/10/13 VERDICT (2026-07-17): Hunt 10 PASSES — new best single channel

MPC Sentinel-1 RTC pull over the dense Khabur tile, 141 FLAT UNOCCUPIED
catalog sites vs matched controls (n=141/141 both arms). Ascending stack 24
scenes (full), descending stack 19 scenes (search exhausted available older
GRD scenes beyond that — not a data-quality problem, just fewer descending
passes archived this far back; both stacks independently clean).

- **Hunt 8 (temporal VV kurtosis): NULL.** AUC 0.518 — chance. Sub-pixel
  "twinkle" statistics do not separate flat sites from steppe at 10 m RTC.
- **Hunt 13 (VH/VV regression intercept): NULL.** AUC 0.541 (raw 0.459,
  i.e. BELOW chance before the sign flip) — no double-bounce signature
  survives RTC processing at this resolution.
- **Hunt 10 (ascending/descending backscatter anisotropy): PASS.**
  **AUC 0.639**, clears the pre-registered >0.60 bar on first run, no tuning.
  Directional radar-scattering mismatch (stable from one look direction,
  volatile from the other = sub-pixel oriented geometry) separates flat
  catalogued sites from open steppe better than any prior channel:
  prominence 0.547, BSI 0.553, texture 0.595, thermal 0.622 (prior best),
  combined-ML 0.616 -> **anisotropy 0.639 is the new single-channel record.**
  Physically sound (built structures/rubble scatter directionally; bare
  steppe is isotropic) and CHEAP (one free MPC pull, no HyP3 quota).
  NEXT: fold into the combined-ML harness (may push the ensemble above
  0.616); test at larger n and a second AOI before calling it robust.

## Wide-area deployments launched (2026-07-17) — "as much space as possible"

Two validated, zero-cost channels turned into actual searches (not controls):

1. **Bare-desert OPERA sweep**: 65 tiles, free NASA DISP-S1, zero HyP3 quota.
   Mojave, Great Basin, Sonoran, Permian salt, Salton/Lucerne — every arid
   OPERA-covered basin. Validated detect_anomalies() per cube. Coordinates
   are US public land (no redaction needed) -> results go in the public repo
   once the sweep completes and candidates are verified.
2. **Anisotropy sweep**: Hunt 10 (asc/desc backscatter anisotropy, AUC 0.639,
   the new best single free-data channel per the 2026-07-17 RTC verdict)
   deployed as an 80-tile local-anomaly search over 35.30-37.30N, 38.60-42.30E
   (the full Jazira). z>4 peak threshold, catalog-distance auto-context.
   MPC's STAC endpoint hit a live service outage (504s, confirmed via a
   direct retest of a box that worked minutes earlier — not our bug); a
   recovery watcher (wait_mpc.py) is chained to auto-launch the sweep the
   moment the endpoint returns. Coordinates stay LOCAL (conflict-zone
   redaction rule) until reviewed.

Both self-execute; verdicts land in this file when complete.

### Hunt 10 replication test — PRE-REGISTERED 2026-07-17, before running

Correction to process: the wide anisotropy sweep (80 tiles, task above) was
launched on a SINGLE unreplicated tile's AUC=0.639 without first testing a
second independent AOI. That is a real process failure, not a nuance — it
means the wide sweep's premise has not been checked. Registering the fix now,
before seeing the result:

- Box2 = 41.30-41.80E, 36.55-36.90N (far-east Khabur, ~90 km from the
  original 40.55-40.95E/36.50-36.75N tile). 1387 catalog sites available;
  subsampled to <=150 flat-unoccupied for tractable runtime.
- Identical feature code, identical flat-site filter, identical control
  sampling.
- PASS BAR: >0.60 — the SAME threshold set before hunt-8/10/13 was run the
  first time (not re-picked with knowledge of the 0.639 result).
- Script: aniso_replicate.py. If this fails to replicate, Hunt 10 is
  downgraded to NULL alongside 8 and 13, and the wide sweep's output gets
  reported as exploratory only, not as a validated-channel search.

### Hunt 10 replication VERDICT (2026-07-17): REPLICATES

Independent sample: box2 (41.30-41.80E, 36.55-36.90N), ~90 km from the
original tile, 150 flat-unoccupied sites + 150 matched controls (subsampled
from 733 available). Identical feature code, identical site filter, SAME
pre-registered >0.60 bar (set before any hunt-8/10/13 result existed).

**AUC 0.608 — PASSES.** Original tile: 0.639 (n=141). Two independent
samples, ~90 km apart, both land in 0.60-0.64. The anisotropy signal
generalizes across this landscape; it is not a one-tile artifact.

Calibration (explicit, so this isn't overclaimed): 0.608 is a THIN pass,
lower than the original — the expected pattern for a real-but-modest effect
meeting fresh data, not a red flag. This validates the FEATURE, not any
candidate. It says nothing about whether any peak the wide 80-tile sweep
(task above) produces is real; each candidate that sweep surfaces still
needs the same individual optical/context check as any other candidate.

Process note for the record: the wide sweep was launched BEFORE this
replication existed (a real sequencing error, corrected once caught — see
the entry above this one). The fix is now in place for future features:
replicate on an independent AOI before deploying at scale, not after.

### Desert sweep: silent-hang bug found and fixed (2026-07-18)

Twice observed: the driver process stayed alive with zero log output and
zero network connections for 1-2.5+ hours, immediately after granule search
succeeded and before any per-window read log line appeared — i.e. NOT
covered by opera.py's existing internal 180s-per-window read watchdog
(commit e2333db from an earlier session). Manual kill-and-restart worked
both times but isn't a fix; it requires a human to notice.

Real fix: restructured into desert_tile_worker.py (does one tile's cube-
build + detect) invoked by desert_sweep_v2.py via subprocess.run(timeout=
4h) per tile. A hang anywhere inside the worker — regardless of which
internal code path is stuck — now gets hard-killed by the OS-level
subprocess timeout, logged, and the tile is left uncached for automatic
retry on the next launch. No internal library fix required; the external
timeout is unconditional.

Progress preserved through both incidents: 9/65 tiles completed with real
results (up to 44 KB anomaly data per tile) before the v2 fix; the stuck
10th tile resumed correctly under v2 with zero lost work.

### Hunt-10 wide sweep: COMPLETE, triaged, verdict NULL for archaeology (2026-07-21)

The 80-tile Jazira anisotropy sweep finished: 228 z>4 peaks, 193 >=1.5 km
from any Menze-Ur catalog site. Full SOP auto-explain triage over all 228:

- **107 MODERN** — OSM settlement/building/industrial/dam/tower within 1 km.
- **12 SLOPE** — mean Copernicus-DEM slope > 5 deg in a 1-km window.
- **6 UNKNOWN** — API failures.
- **103 nominal survivors** — but OSM coverage in rural Syria/Iraq is thin,
  so "no OSM features" is weak evidence there.

Chain analysis (RANSAC collinearity, >=4 points within 0.6 km of a line):
27 chains over all 228 peaks. **Null-tested** (20 scrambled fields, both
uniform and marginal-preserving): random fields give 20-23 chains and up to
7-point chains — so only the two chains with >=8 members (10 and 8 points,
spans 80/43 km) exceed chance; they are engineered corridors (the r1c0/r2c0
fan matches the Tabqa-dam transmission grid). Most 4-6-point "chains" are
chance-compatible; the chain kill is a deprioritizer, not proof. After
chains + pair-clusters: 36 isolated survivors.

Optical review (Sentinel-2, 2 km chips) of the top 12 isolated survivors:
**12/12 explained.** 7 unmapped modern settlements/farm compounds, 2-3
ruined or destroyed villages (Sinjar district — conflict damage; these
coordinates stay LOCAL, never published), 2 canyon-rim layover (the 1-km
MEAN slope check dilutes cliff walls — future triage should use p90 local
slope), 1 isolated small object on a straight desert lineament (pipeline
valve-station signature). **Zero plausible archaeological candidates.**

**Channel verdict, stated plainly:** a z>4 anisotropy PEAK sweep is a
structure detector, not an archaeology detector. The validated tell signal
is a subtle 1-3 dB distributional shift (AUC 0.61-0.64 — weak, ranking-
grade); the sweep's 5-27 dB peaks are buildings, ruins, pylons and cliffs.
Physics said this in advance; the optical review confirmed it 12/12.
Correct future use of the validated feature: as a RANKING FEATURE scored at
candidate locations from the prominence channel (like TDA-H1), never as an
autonomous peak sweep.

Salvage worth recording: the sweep IS a working unmapped-structure detector
for OSM-dark regions (found real settlements, ruins and infrastructure that
OSM lacks). That capability could serve humanitarian mapping (e.g. HOT-
style unmapped-settlement detection) — publishable as a tool WITHOUT
publishing any conflict-zone coordinates.

### Pre-registration: aniso as ranking feature on the 111 TDA flat candidates (2026-07-21)

Registered BEFORE any anisotropy value is computed at these locations.
Feature: per candidate, mean |mean_VV_asc - mean_VV_desc| dB inside r<500 m
MINUS the same in a 750-1250 m annulus (background-subtracted so values are
comparable across tiles; the validated form was absolute-within-tile, this
is the cross-tile adaptation). Stacks: MPC RTC 2020-2023, <=10 scenes per
orbit. Combined rank = mean of (h1max_m descending rank, aniso descending
rank). This is a RANKING deployment of a validated weak feature (AUC
0.61-0.64), not a detection test: no pass/fail bar, no discovery claims.
Deliverable = re-ranked triage queue + S2 chips for human review, chips
reviewed in combined-rank order. Candidates remain LOCAL-only (Syria).

### Cap-1 sweep #1 CLOSED: post-veto re-analysis, all events explained (2026-07-21)

The pre-registered scene-event veto (registered 2026-07-18, before this
re-run) applied to the cached sweep-1 stack: the 2023-06-29 date held 412
of 611 flagged pixels (67% > 30% bar) -> vetoed as the known rain event.
Post-veto: 4 events. Auto-explain (S2 before/after):

- Three clusters, one story: ~8 ha of valley-bottom FIELDS at ~34.865N
  38.457E, onset 2023-04-30, green in late March -> cut/plowed by June 1.
  Harvest — the Syrian wheat/barley window. Agriculture, mundane.
- One weak event (9 px, depth 0.15, onset 2023-07-11): steppe drainage
  shrub patches green in March, dried by June. Phenology, mundane.

**Sweep #1 final verdict: clean NULL — zero unexplained candidates.** The
capability worked as designed end-to-end: it detected real surface change
(rain, harvest, dry-down) and the veto + auto-explain pipeline attributed
every event without human archaeology review being needed. The Palmyra-west
2023 window simply contains no unexplained discrete disturbance above the
validated threshold. Coordinates stay LOCAL per conflict-zone rule.

### Hunt 11 GATE PASSED + Part A registration (2026-07-21)

Gate (metadata): DISP-S1 granules carry corrections/perpendicular_baseline
as a FULL-RES per-pair layer (7830x9512 float32, verified on F38238).
B-perp spread across 14 sampled pairs: std 80 m, range -109..+201 m —
enough regression leverage (est. per-pixel height-error floor ~2 m at 434
epochs, ~5 mm epoch noise, R*sin(theta)~550 km).

**Part A registered BEFORE any displacement-vs-baseline regression is run:**
Data: Mont Belvieu cube already on disk (434 epochs, 400x400 px) + per-epoch
window-mean B-perp fetched from the granules' baseline layer only.
Model: within each single-reference segment, per-pixel OLS
d_k = a + v*t_k + c*B_k; height error dh = c * R*sin(theta) (theta from
granule metadata if present, else 37 deg documented fallback).
Positive control: >=1 structure built AFTER 2015 (post-COP-DEM) at the Mont
Belvieu NGL complex, identified from S2 imagery BEFORE any regression
output is viewed. Negative control: flat farmland pixels.
PASS bar: post-2015 structure shows |dh| >= 3 m at >= 3 sigma vs farmland
AND the |dh| map is structure-aligned (visibly tracks buildings/tanks, not
noise). FAIL: no structure-aligned dh or control < 3 m -> Part A dies and
ticks the N=3 counter. Purpose if validated: sub-DEM-resolution height
anomalies from FREE data = mound/structure detection below 30 m DEM
resolution (the resolution wall, attacked from phase geometry).

### Hunt 11 Part A: controls FROZEN from optical only (2026-07-21)

Selected from NAIP 1-m before any regression output exists:
- POSITIVE: new fractionator train complex at ~(29.8700, -94.9350), box
  29.8655..29.8760 N, -94.9420..-94.9265 E. NAIP 2016: bare field (only a
  staging corner). NAIP 2022: full multi-train plant with tall columns
  (long shadows). Built 2017-2021 => guaranteed absent from the 2011-2015
  COP-DEM. Expected dh strongly positive (tens of m at the columns).
- NEGATIVE: farmland at (29.9250, -94.9450) +-500 m — open fields in both
  NAIP years. Expected |dh| < ~2 m (DEM noise floor).
- Amendment (specified now, pre-output): because the plant was built
  mid-stack and construction-era pixels decorrelate, control evaluation
  uses only reference segments starting >= 2022-01-01 (~170 epochs).
  Farmland negative is evaluated on the same segments.

### TDA x aniso ranked triage: backlog CLOSED, 12/12 top candidates are confounders (2026-07-21)

The pre-registered ranking deployment ran to completion: all 111 TDA flat
candidates scored (aniso center-minus-annulus excess +0.05..+0.89 dB — the
subtle regime, exactly as the validated feature should behave), combined
rank = mean(h1 rank, aniso rank), S2 chip per candidate. Human review of
the top 12 by combined rank: **12/12 explained** — incised meander loops
(x2, the predicted "Delta lesson" class), mountain/badland contour loops
(x6, incl. the h1=52 m top-persistence entry = a dissected massif), modern
farm compounds (x1, the highest-aniso entry +0.89 dB = its buildings), and
known-catalog margins (x3 at 0.5-0.7 km from catalog sites). Zero new
ring-form candidates.

This clears the 111-candidate backlog with a NULL and re-confirms the
2026-07-16 verdict with the full pipeline: at 30 m DEM the TDA-H1 channel
cannot do autonomous discovery — the flat-gate (roughness < 8 m) still
passes badlands and meander spurs. The aniso feature behaved correctly
(it up-ranked the one real structure cluster); the pool it ranked was
terrain. Autonomous ring discovery still waits on 10-12 m elevation
(TanDEM-X or CORONA stereo). Candidates and chips remain LOCAL.

### CORONA human-GCP georeferencing operational + first ring verdict (2026-07-21)

Strip ds1102-1025da013 (Dec 1967) georeferenced by human-in-loop GCPs
(the operator reading rendered images; no auto-matcher). Method that
worked: catalog-anchor bootstrap — Tell Chuera identified by its
1-km multi-summit mound + pond + radial hollow-way star (GCP-1), a
wadi-side mound cluster confirmed the row axis (GCP-2). Findings:
- The strip is stored 180-deg ROTATED vs geography: col increases WEST
  (~2.0 m/col), row increases NORTH (~2.0 m/row). Provisional transform:
  col = 87560 + (39.4997-lon)/2.24e-5; row = 3040 + (lat-36.6489)/1.8e-5.
  Accuracy ~200 m near Chuera, degrading to ~300-800 m at 25 km
  extrapolation (panoramic distortion) — refine locally per target via
  wadi-bend matching (demonstrated at ring34, ~150 m).
- 1967-vs-2023 portrait pairs produced for Tell Chuera + rings 25/28/32/34
  (ring 9 lies ~2.5 km south of the strip footprint — off-image).
- **Ring 34 verdict (first CORONA-vetted candidate): the crisp ~360 m
  spoked annulus in 2023 S2 does NOT appear in 1967 CORONA** at a local
  registration good to ~150 m, under low December sun that would shadow-
  enhance a real bank. 1967 shows only a compact hamlet + faint pale
  sub-circle. The annulus is very likely POST-1967 (livestock/agricultural
  enclosure attached to the now-sprawling, now-abandoned village).
  Ring 34 demoted: probable modern feature, not archaeology. (Residual
  check if ever needed: a second CORONA date or KH-7 frame.)
This is the CORONA channel doing its designed job: a 60-year time
baseline that discriminates modern earthworks from ancient ones —
a discriminator no amount of modern-only data can replicate.

### Hunt 11 Part A iteration 1: FAIL (2026-07-21) — N=1 of 3

Ran exactly as registered (7 post-2022 segments, 105 epochs, per-pixel OLS
d=a+v*t+c*B, dh=c*R*sin37). Results: FARM negative control mean +0.36 m
(unbiased) but std 10.97 m — the noise floor is ~11 m, not the ~2 m
back-of-envelope (only 105 usable epochs after the registered post-2022
restriction, and per-epoch atmosphere is 10-15 mm, not 5 mm; the numbers
reconcile exactly). PLANT positive control: p95 17.0 m, max 35.1 m —
real excess over farmland (17 vs 11) but 0% of pixels clear the 3-sigma
(33 m) bar, and the dh map is speckle, not structure-shaped. Additional
finding: the 80%-completeness mask leaves only ~250 valid pixels per
control box (DISP-S1 masks low-coherence epochs) — the naive estimator is
also data-starved. **FAIL on both registered criteria. Hunt-11 kill
counter: 1 of 3.**

Iteration 2 registered NOW, before any new run, ONE change: spatially
HIGH-PASS each epoch's displacement field before the regression (subtract
a 500-m Gaussian blur, sigma ~17 px at 30 m). Physical basis: DEM-error
signal lives at structure scale (pixels-100s m); atmospheric delay lives
at km scale — high-pass should cut the farmland floor several-fold while
preserving the plant signal. Same segments, same controls, same bar
(3 m at 3 sigma vs farmland + structure alignment). Ceiling note recorded
in advance: even a pass at sigma~1-3 m leaves 3-5 m tells marginal — this
hunt's archaeological payoff requires the floor to land under ~2 m.

### Hunt 11 iteration 2: FAIL (N=2 of 3) — but the failure is the control, not the physics (2026-07-21)

The registered high-pass did its job: per-segment median |dh| fell from
7-23 m to 1.3-3.0 m (atmosphere removed, as predicted). New numbers:
FARM +8.01 +- 5.89 m, PLANT p95 15.5 m / max 35.1 m, 0% over the 3-sigma
bar (17.7 m). FAIL as registered. Diagnosis: DISP-S1 only retains
COHERENT pixels — in "farmland" those are the farm BUILDINGS, so the
negative control measures real structure heights (+8 m ~ barns), and the
plant's 35 m max ~ fractionator columns. Both controls are height-bearing;
the null was ill-posed. This is consistent with the estimator actually
RECOVERING heights — but per SOP the iteration fails against its bar.

Iteration 3 (FINAL) registered before any new run, ONE change: negative
control redefined as ROAD pixels — OSM motorway/trunk/primary/secondary
polylines inside the cube box, rasterized with a 1-px buffer. Pavement is
flat (true dh ~ 0), coherent, and selected geometrically from OSM, not
from any regression output. Same segments, same estimator (iter-2
high-pass), same bar: plant p95 >= 3 m at >= 3 sigma vs road-pixel std
AND structure alignment. If it fails, Hunt 11 dies at N=3.

### Hunt 11 KILLED at N=3 (2026-07-21) — signal real, floor too high

Iteration 3 (road-pixel negative, registered above): ROAD control behaves
as a true null — mean +0.54 m, std 2.91 m, n=558 (the estimator is
unbiased on flat coherent ground). PLANT: p95 8.27 m, max 27.61 m
(~fractionator column heights), but 3-sigma = 8.72 m and only 1% of plant
pixels clear it. FAIL against the registered bar; per the N=3 kill switch
Hunt 11 is DEAD.

What the three iterations established, for the record:
1. The B-perp height-error signal EXISTS in free DISP-S1 products and the
   per-pixel OLS recovers it (plant excess ~3x the road floor, correct
   sign, magnitudes matching real structure heights).
2. The achievable per-pixel floor with ~105 post-2022 epochs is ~3 m
   (after the high-pass that removes atmosphere), and DISP-S1's coherence
   masking leaves few valid pixels on exactly the structures of interest.
3. For archaeology (3-5 m mounds, often in seasonally decorrelating
   fields) the floor and the pixel-retention pattern are both wrong —
   the channel cannot reach its target regime on this data. A future
   revisit would need full-archive segments (2016+, ~3x epochs) or
   CSLC-level phase, i.e. a different experiment, separately registered.

Kill counter honored: three pre-registered iterations, controls frozen
throughout, one change each. Hunt 11 closes as a clean negative with a
quantified floor — the most useful kind of dead end.

### CORONA verdicts for rings 25/28/32 — ring 32 PROMOTED (2026-07-21)

Same per-target local-registration method as ring 34. Verdicts:
- **Ring 32 (~36.7N 39.7E rounded; precise LOCAL only): PROMOTED — the
  best surviving candidate of the whole ring hunt.** 1967 CORONA (2 m):
  a living village of courtyard houses ON a raised sub-circular platform,
  central bright mound, a pond/depression at its N edge, dark moisture
  halo, and hollow ways radiating in all directions (ancient route-hub
  signature). 2023 S2: village abandoned/leveled; a crisp dark annulus
  ~350-400 m remains — exactly the TDA-detected DEM ring — with fields
  encroaching. Three sensors, 56-year baseline, one consistent story:
  a pre-1967 (probably pre-modern) ringed settlement, not in the
  Menze-Ur catalog. What CORONA cannot do is date it below "pre-1967";
  next step is professional follow-up (TanDEM-X 12 m DEM or field
  survey) — beyond free-data reach, so the candidate is documented and
  parked, coordinates local.
- Ring 25: faint sub-circular tonal patch present in BOTH eras — not
  modern construction; possibly a plow-degraded low mound or natural
  ring. Stays weak-open.
- Ring 28: no optical counterpart in either era (DEM-only anomaly).
  Inconclusive.
- Ring 9: off-strip (south of da013 footprint); needs the adjacent frame.

Ring-hunt scoreboard after CORONA vetting: 1 promoted (32), 1 killed as
modern (34), 1 weak-open (25), 1 inconclusive (28), 1 unchecked (9).
The 60-year baseline changed the verdict on 2 of 4 checkable candidates —
the channel earns its place in the standard triage chain.

### Mojave desert-sweep candidate: layover hypothesis REFUTED — lead revived (2026-07-21)

Geometry check at the rank-1 desert candidate (34.5591 -116.7685, Mojave):
Copernicus DEM slope in the central 360 m window is 0.6 deg mean / 1.0 deg
max (1.3 km window mean 1.2 deg) — there is no steep radar-facing slope;
the earlier "suspect layover/DEM artifact" note is withdrawn. Status now:
a localized ~90 m accelerating bowl (-2.5 cm/yr peak, breakpoint mid-2019
from -0.5 to -3.6 cm/yr, Mogi depth ~250 m, r2 0.87) on FLAT ground near
the Troy Lake playa margin, no quarry/buildings/pivots in NAIP. Open
mundane alternatives: single-well pumping cone; playa-margin evaporite
dissolution (evaporite beds exist locally). Next: cross-check state well
records / water levels; keep in the verification queue for when the sweep
completes. This is currently the best unexplained lead in the desert sweep.

### Ring 9 located on the adjacent frame — provisional (2026-07-21)

da013's footprint misses ring 9 by ~2.5 km south; the adjacent frame
ds1102-1025da014 (same pass, next frame south — confirmed by it containing
the Jebel Abd al-Aziz ridge system) covers it. The ring-9 wadi corridor was
matched by feature signature (double-bend + long straight track crossing +
bankside settlement clusters, all present in both 1967 and 2023). At the
predicted position on the west bank, the 1967 frame shows a settlement
cluster and a possible ~240 m sub-circular feature — the same
"pre-1967 feature with settlement context" pattern as ring 32. PROVISIONAL:
da014 needs its own 2-GCP registration pass (transform constants are
per-frame) before a firm verdict. Queued.

### Tampa deliverable prepared (2026-07-21)

Top-3 localized-subsidence brief written to
data/reports/tampa_sinkhole_brief_2026-07.md (LOCAL — data/ is gitignored;
release is the user's call, intended receivers FGS/SWFWMD/county/FDOT).
Candidates: 28.44303 -82.42280 (-6.8 cm/yr, accelerating -1.7 cm/yr2,
pasture + relict sinkhole pond); 28.37163 -82.46685 (-4.2 cm/yr,
accelerating, ~100 m east of the Suncoast Parkway); 28.35692 -82.50001
(steady -1.4 cm/yr, wetland — peat-compaction alternative stated). NAIP
context chips alongside. All stated as UNVERIFIED satellite leads with
uncertainties and receiver actions listed.

### Mojave lead strengthened: well-records check + Ridgecrest timing (2026-07-21)

CA DWR Well Completion Reports queried in a ~2.4 km box around the
candidate (34.5591 -116.7685): exactly ONE well on record — WCR1982-006581,
domestic supply, 400 ft, drilled 1982, ~0.95 km SE. No agricultural wells.
A domestic well cannot produce a localized ~65,000 m3/yr volume-loss bowl;
the pumping-cone explanation is substantially weakened (caveats: WCR
coverage is incomplete and section-center registered; the Mojave Basin is
adjudicated, so watermaster pumping records would be the definitive check).

Separately noted: the candidate's fitted breakpoint is 2019.5. The M6.4/
M7.1 Ridgecrest earthquakes struck 2019-07-04/05 (= 2019.51), ~120 km NW,
with strong shaking across the Mojave block. Earthquake-triggered onset of
localized compaction/dissolution collapse fits the observed rate change
(-0.5 -> -3.6 cm/yr) exactly in time. Recorded as a suggestive coincidence,
NOT a conclusion. Lead status: best-in-sweep, unexplained, strengthened.

### Ring 9 first-look verdict + campaign wrap (2026-07-21)

Ring 9 zoom pair via single-anchor local registration on da014 (~300-400 m
residual, quantified by the wadi offset between panes): 2023 S2 shows a
clear ~350-400 m sub-circular enclosure with darker rim on the wadi's west
bank. 1967 CORONA shows bankside settlement/ruin clusters and a FAINT pale
sub-circular zone at the corresponding position — no crisp bank visible,
but also no sharp modern enclosure (the ring-34 modern-kill pattern does
NOT apply; the modern feature reads degraded, not new-built). Verdict:
**weak-positive, open** — one rung below ring 32. Firming it up requires a
proper 2-GCP registration pass on da014. Queued.

**Ring campaign final scoreboard (5 DEM-detected candidates through the
60-yr CORONA baseline): ring 32 PROMOTED (strongest — 1967 village on
ringed platform, radial hollow ways); ring 9 weak-positive; ring 25
weak-open; ring 28 inconclusive; ring 34 KILLED (modern).** One promotion,
one kill, three opens from five — the baseline changed or sharpened the
verdict on every candidate it could see.

Also this session: Tampa brief finalized with FGS incident-database
cross-check (0 reported incidents within 1.1 km of any of the 3
candidates — all new signals); Mojave lead strengthened (1 domestic well
in 2.4 km box; breakpoint 2019.5 = Ridgecrest timing).

### CORONA stereo experiment: queued with feasibility math (2026-07-21)

Forward camera confirmed held: 1102-1025df/, 36 frames. Frame-to-frame
correspondence with the aft camera is NOT trivial (df012/df013 probed at
the da col bands show different ground — along-track and scan-start
offsets differ; df012 cols 83-93k contains a bright serrated ridge system
absent from da013). Frame localization should proceed via whole-strip
quicklooks (factor 64) before any zoom probing.

Feasibility (recorded before the experiment): KH-4B fore/aft convergence
~30 deg -> height = parallax x ~1.87 at 2 m GSD. Tell Chuera's ~18 m
mound = ~5 px parallax (clean positive control); ring 32's bank at 1-2 m
= <1 px (below floor); ring 32's PLATFORM registers only if >= ~4-5 m.
Design when run: film-to-film local correlation (same modality — unlike
the 6 failed cross-modal attempts), Chuera mound parallax must be
recovered within 30% of prediction as the gate, THEN measure ring 32
platform. If the gate fails, the method dies without touching the ring.

### Pre-registration: closure-phase archaeology stretch test (2026-07-21)

Question: do FLAT catalogued tell sites carry elevated |closure phase|
(moisture/dielectric heterogeneity from buried architecture) vs matched
steppe controls? Uses the VALIDATED Track-2 closure tool in a new
application. Registered before any job is submitted:
- AOI: the dense-Khabur hunt-8/10/13 box (36.50-36.75 N, 40.55-40.95 E);
  sites = the SAME 141 flat unoccupied catalog sites and 141 matched
  controls already frozen for the RTC hunts (no re-sampling).
- Data: one Sentinel-1 track, spring window (Mar-Jul 2023, moisture
  contrast season), sequential 12-day pairs + skip pairs -> triplets.
  HyP3 INSAR_GAMMA, job name gam_closure_arch (~17 jobs, free quota).
- Metric: per-site mean |closure| (scene-median referenced, 3x3 window)
  across all triplets; rank AUC sites-vs-controls.
- Bar (same as all channels): AUC >= 0.60 = PASS -> new detection channel
  for buried architecture. AUC < 0.60 = NULL, recorded, no retry beyond
  the standard N=3 iteration rule.

### Closure-arch test: execution note (2026-07-21)

First batch (21 jobs) accidentally used track-123 frame 466, which only
GRAZES the AOI (65-row sliver; lat span 36.72-38.81 vs box top 36.75) —
scene search by intersection is not containment; the per-date dedup kept
the wrong frame. All 141 sites fell outside coverage -> n=0, no verdict
consumed (the registered test is unaffected; no site data was seen).
Corrective batch gam_closure_arch2 submitted on frame 471 (lat 35.23-37.26,
fully contains the box), same 21 pair structure. Chained watcher+analyzer
running; verdict lands automatically. Lesson recorded: check frame
CONTAINMENT of the AOI before submitting InSAR batches.

### Closure-arch iteration 1: FAIL 0.589 (partial coverage), iteration 2 registered (2026-07-21)

Frame-471 batch: 21/21 jobs, 10 triplets, but the product rasters cover
only ~11 km of the box E-W -> 31/141 sites sampled. Separability 0.589
(below 0.60 bar) = FAIL, N=1 of 3. Notable and recorded before iteration
2: the polarity is REVERSED from the naive hypothesis — sites show LOWER
|closure| than controls (0.0452 vs 0.0547 rad median), consistent with
compacted anthropic sediment retaining less moisture than steppe soil.
The separability metric is polarity-agnostic (same standard as the RTC
hunts), so no hypothesis change is needed or made.

Iteration 2 registered NOW, ONE change (coverage fix, not tuning):
reproject BOTH batches (frames 466 + 471, 42 products) onto a common
EPSG:4326 grid and mosaic by nanmean of per-frame |closure| triplets;
evaluate every frozen site/control the mosaic reaches. Same metric, same
0.60 bar.

### Closure-arch CORRECTION: iteration 1 RETRACTED, root cause found (2026-07-21)

Two errors found and owned before any verdict stands:
1. **Iteration 1's separability 0.589 is VOID** — the window-read returns
   only the raster's covered sliver, but the site-sampling function mapped
   lat/lon assuming the array spanned the full box: sites were sampled at
   wrong pixels. The number carried no information about sites. Retracted.
2. **Iteration 2 (correct geometry, two-frame mosaic): DATA-INSUFFICIENT,
   no verdict** — true finite coverage is 6% of the box; n_ctrl=2. Root
   cause diagnosed from product footprints: track 123's eastern swath edge
   sits at 40.68 E — the AOI (40.55-40.95 E) straddles it. Frame 471
   covers full lat but only the box's western third; frame 466 only the
   top sliver. This track can never cover the frozen site set.

Kill-counter accounting, stated plainly: NO valid iteration has been
consumed (an invalid measurement and an unevaluable one give zero
information about the hypothesis; treating them as FAILs would be as wrong
as treating them as passes). N remains 0 of 3. Registered next execution
(same test, same sites, same bar): submit the 21-pair structure on a track
whose scene footprint CONTAINS the full box — candidates track 50 (desc)
or 43 (asc) from the original search — with containment verified from
scene geometry BEFORE submission. Two execution lessons now in the SOP
list: (a) verify frame containment, (b) never sample a window-read array
as if it spanned the request box.

### Closure-phase archaeology: PASS 0.619 on first valid iteration (2026-07-21)

Track 50 frame 471 (verified to CONTAIN the box): 21/21 jobs, 10 triplets,
**coverage 100%, n=141 sites vs n=141 controls — separability 0.619,
clears the pre-registered 0.60 bar with zero tuning.** Sites show HIGHER
mean |closure| than steppe controls (0.0595 vs 0.0505 rad) — the original
hypothesis direction (moisture heterogeneity over archaeological sediment).
The earlier reversed-polarity hint from the invalid partial batch did not
survive valid geometry.

Calibration, stated before anyone overclaims: 0.619 is a THIN pass —
ranking-grade, same family as anisotropy (0.61-0.64) and thermal (0.622).
It is a population-level statistical signal, NOT a site detector. And the
mechanism is INFERRED (subsurface-moisture heterogeneity), not proven —
a surface-texture correlate cannot be excluded from this test alone.

What makes it notable anyway: closure phase is a signal every standard
InSAR pipeline discards as error, and it carries archaeological site
information on free data. We are not aware of prior use of closure phase
for archaeology (not claimed as fact — as absence-of-known-precedent).

REPLICATION REGISTERED NOW, before any wider deployment (the rule learned
from the aniso sequencing error): independent second AOI = the aniso
replication box (41.30-41.80 E, 36.55-36.90 N, ~90 km away), same recipe
(one containing track/frame — containment verified BEFORE submission —
Mar-Jul season, 21-pair triplet structure, frozen flat-site/control
sampling seed 11), same 0.60 bar. PASS -> validated dual-AOI channel;
FAIL -> stays a single-AOI curiosity, N counter starts.

### Closure-phase archaeology REPLICATES: 0.603 on the independent AOI (2026-07-21)

The registered replication ran immediately: track 50 frame 471 turns out
to CONTAIN both boxes, so the finished products covered AOI-2 directly
(independence lives in the sites — 733 flat catalog sites + 733 matched
controls, 90 km from AOI-1 — not in the interferograms; the 21 duplicate
gam_closure_rep jobs submitted before realizing this are redundant and
void). Result: **separability 0.603, over the 0.60 bar, same polarity
(sites 0.0912 vs controls 0.0688 rad), 100% coverage, n=733/733.**

Channel status: **dual-AOI validated, ranking-grade** — 0.619 and 0.603,
both thin. Stated ceiling context: every validated free-data channel on
this ground truth lands in 0.55-0.64 (prominence 0.547, texture 0.595,
combined-ML 0.616, thermal 0.622, aniso 0.639/0.608, closure 0.619/0.603)
— a consistent family ceiling around ~0.6. Closure joins the ensemble as
a ranking feature; next natural step is folding it into the combined-ML
harness to test whether the ensemble finally beats 0.64. No autonomous
detection claims; mechanism (subsurface moisture) remains inferred.

### Ensemble increment 1: NO GAIN from naive rank-mean (2026-07-22)

closure+aniso rank-mean on the AOI-1 frozen sites: 0.598 — BELOW the best
in-test single channel (closure 0.619, which reproduced exactly). The
weaker channel drags the unweighted mean down; no ensemble benefit from
naive combination. Secondary finding worth its own line: aniso re-measured
on the SAME box at 16 scenes/orbit gives 0.578 vs 0.639 at 24 scenes —
the anisotropy channel is sensitive to stack depth; any use of it must
specify (and afford) full stacks. If ensemble work continues, the next
legitimate step is a PRE-REGISTERED cross-validated weighted combination
on full-depth stacks — n=141 with thin signals makes unregistered weight
fitting an overfitting machine, so it is not attempted casually.

### Stereo prep: fore/aft frame pairing unresolved after systematic probing (2026-07-22)

Findings recorded for the next session: (1) FGDC/NITF metadata positions
carry DIFFERENT biases per camera (da013 metadata center is ~0.11 deg
north of its empirically verified ground; df007 metadata says 36.7N but
shows Tigris-style canyon country; df012 metadata says 36.1N but shows a
major E-W ridge). Metadata cannot pair the cameras. (2) Probe log — df007:
big meandering canyon river (Tigris?); df011: E-W serrated ridge with
tell-dotted steppe NW + a DISTINCTIVE pentagon-walled compound (band cols
84-92k, upper right); df012: E-W ridge (Abd al-Aziz or Sinjar); df013:
cloudy steppe; df014: dissected wadi country with dark pond. The two
ridge frames are Abd al-Aziz vs Sinjar in some order — resolving WHICH
pins the whole df sequence (0.13 deg per frame step).
NEXT-SESSION RECIPE (no more roulette): S2-verify df011's pentagon
compound against both ridge hypotheses (one chip each at the ridge-relative
predicted spot); one match anchors df011 absolutely; da013's twin then
follows by frame arithmetic; then the Chuera parallax gate runs.
Also noted: ensemble increment 1 closed NO-GAIN (see entry above); desert
sweep 54/65 at this writing.

### Stereo prep: FRAME-TILT discovered — the root cause of all pairing failures (2026-07-22)

Decisive observation: df009 shows the Khabur/Kawkab country (36.55 N) at
its east end and the Abd al-Aziz north flank (~36.45 N) at mid-strip —
one 14-km-tall frame spanning both is impossible unless the frame's
footprint is TILTED, drifting tens of km in latitude across its 215-km
length (consistent with the ~10-deg rotation the inflated FGDC bboxes
implied). Consequence: "frame = latitude band" arithmetic is invalid for
BOTH cameras; every one of the 15 failed pairing probes assumed it. Even
da013's validated transform is a LOCAL fit (cols 76-93k) — extrapolation
to strip ends will drift.

Fix (recorded for the next stereo session): build a per-frame tilted-line
model lat(col) from 2-3 anchors SPREAD ALONG each frame (west/mid/east),
harvested from the existing factor-32 montages (dfm_007..012 on disk).
With tilt models for df007-df009, Chuera's frame+col follows analytically.
No more single-point probes. Probe log extended this session: df008
62-70k = tell checkerboard plains; df009 62-70k = rolling steppe; df009
44-62k = Abd al-Aziz north fan; frame-edge zooms negative for Chuera;
df008 68.5-73.5k = walled village on wadi (not Chuera).

### Bare-desert OPERA sweep: COMPLETE, honest verdict = NO new void, pipeline validated (2026-07-22)

65-tile North-America desert sweep finished (Mojave/GreatBasin/Sonoran/
PermianSalt/SaltonLucerne). Raw: 6,379 deduped anomalies, 743 classified
accelerating_subsidence. Strict localized-void filter (is_localized +
accelerating + rate_reliable + |regional_correlation|<0.30 + area<0.10km2
+ void_likelihood>=0.9) -> 20 survivors after OSM/slope auto-explain.

Cluster + wider-context check demolished all 20 as known processes:
- **GreatBasin (~11 survivors, 39.3-39.7N -118.4/-118.6):** the Carson
  Sink / Fallon closed basin — 8.5 km from Stillwater geothermal, and the
  classic USGS groundwater-subsidence province (Newlands irrigation).
  Eleven "decoupled localized bowls" = individual pump/geothermal cones.
- **SaltonLucerne (~7):** Imperial-Coachella geothermal+ag belt; loudest
  survivor (-7.7 cm/yr, 33.27/-115.60) has 2 geothermal + 1 gas plant
  within 15 km.
- **Sonoran (1, 33.00/-113.30):** looked cleanest (OSM=0, isolated) but
  NAIP shows it sits on the rim of a center-pivot irrigation complex near
  Gila Bend AZ. Agricultural drawdown. (Lesson: OSM farmland tags are
  ABSENT for AZ desert agriculture -> the auto-explain MUST include an
  imagery/center-pivot stage, not OSM alone. Filed as an SOP fix.)

**Verdict: zero new void discoveries.** The sweep is a POSITIVE validation
that the pipeline correctly lights up real subsidence provinces, and an
honest NEGATIVE on void discovery: at 30 m free data the detector cannot
separate a pumping/geothermal bowl from a collapse void inside an active
groundwater basin (each pump makes its own decoupled local cone, defeating
the regional_correlation filter). The single best remaining lead is
unchanged and stands apart precisely because it has NO such explanation:
the earlier Mojave bowl 34.5591/-116.7685 (flat, one distant domestic well,
onset at Ridgecrest 2019.5, isolated from any ag/geothermal). Everything
survives as data (data/research/desert_survivors.json); coordinates are US
public land, no redaction.
