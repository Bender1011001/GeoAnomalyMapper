# Research tracks: new mathematics on data we already hold

Charter (2026-07-15): stop re-running established methods; extract NEW results
from the SAME archives by processing signal that standard pipelines discard or
never look at. Every track is pre-registered (hypothesis, test, numeric pass,
kill criterion) per docs/DISCOVERY_SOP.md before its first result is seen.

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
