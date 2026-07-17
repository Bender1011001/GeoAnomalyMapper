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
