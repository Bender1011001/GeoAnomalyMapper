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
