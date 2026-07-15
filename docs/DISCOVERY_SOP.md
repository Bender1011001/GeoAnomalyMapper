# Discovery-Module SOP (Standard Operating Procedure)

Governance for every new detection capability in GeoAnomalyMapper. Derived
from measured failures (vibrometry retraction, fusion 0/14 retraction,
Casa Grande thermal null, Fairfield warehouse false alarm, LiDAR meander-bluff
ghosts, Chuera ranking starvation, prompt-v2 systematic misclassification) and
measured successes (Wink ground truth x2, Menze-Ur 25/25 catalog validation).

## Part 1 — Universal governance (binding before any pixel is processed)

1. **Triage budget rules the sweep.** Human eyeball capacity (chips/night)
   dictates sweep size and thresholds. Engine outputs a ranked Top-N queue;
   the metric is Precision@N, never theoretical recall. Candidates must be
   deduplicated across overlapping tiles (merge radius >= half the feature
   scale) and persist across reruns.

2. **Auto-explain before human triage.** Every candidate is automatically
   cross-referenced against (a) OSM landuse polygons (quarry, construction,
   military, industrial, landfill), AND (b) a before/after Sentinel-2 optical
   pair straddling the anomaly date — OSM lags by years; the optical change
   check is what actually resolved Fairfield. Explained hits die silently.

3. **Provenance or it didn't happen.** Every surviving candidate serializes
   scene IDs, acquisition timestamps, orbit/track numbers, processing
   parameters, and thresholds into results.json beside the coordinates.

4. **Ethics & disclosure, decided in advance.**
   - Archaeology: candidate coordinates in looting-vulnerable regions are
     NEVER published publicly; findings route to national heritage
     authorities. Public write-ups redact coordinates.
   - Infrastructure risk: acceleration/void signatures under infrastructure
     go to the jurisdictional regulator first, not social media.
   - Output register is always "candidate", never "discovery"/"proof".
     Novelty is only determinable inside a ground-truth catalog footprint.

5. **Kill switch (N=3, pre-registered).** A capability dies after 3 failed
   tuning iterations against its frozen control set. An *iteration* is ONE
   pre-registered parameter change with a predicted effect, re-scored on the
   SAME controls. Changing the control set resets nothing (no target
   shopping). Negative-control zones are chosen BEFORE the first run,
   geologically matched to the sweep terrain.

## Part 2 — Capability 1: Coherence Transient Tracking ("Ghost Trail")

**Output register:** "Candidate discrete surface disturbance. Precision
unknown until optical validation; expect false positives from permitted
construction, aeolian transport, wadi flash-floods."

- **Data path (corrected):** HyP3 `INSAR_GAMMA` jobs on short-baseline
  (6-12 day) Sentinel-1 pairs; each product bundle includes a coherence
  GeoTIFF (`*_corr.tif`). There is no "raw SLC coherence matrix" to pull —
  SLCs are processed server-side. The Wink pipeline already ingests these
  products; reuse it. OPERA DISP-S1 (displacement time series) has no
  per-pair coherence and is NOT this data path.
- **Quota math:** ~10k HyP3 credits/month. A 12-month baseline at 12-day
  pairing is ~30 jobs per AOI. Plan 1-2 AOIs at a time; a "desert sweep" is
  a quota fantasy.
- **Physics — drop-then-recover:** disturbed soil re-coheres in weeks.
  `drop -> recovers at new state` = one-time event (digging/grading; the
  target). `drop -> stays down` = ongoing process (vegetation flush, active
  mining, moisture) — classified and filtered, not flagged.
- **Confound normalization:** score each pixel's drop against the
  scene-median drop of that same pair (regional null — rain decorrelates
  whole scenes). Require persistence across >= 2 subsequent pairs; confirm in
  both ascending and descending tracks where available.
- **Named positive control:** the Gemini Solar Project construction start
  (Moapa Valley, NV — massive dated disturbance of stable desert, publicly
  documented start; verify exact AOI/date from public filings before job
  submission). Success = onset timestamped within +/-1 acquisition cycle.
- **FP ceiling:** < 5 persistent flags / 1,000 km^2 / year on a
  pre-registered undisturbed negative zone in the same desert.

## Part 3 — Capability 2: Temporal Phytological Variance ("Crop Mark")

**Output register:** "Candidate subsurface structural impedance. Minimum
resolvable feature ~10-20 m (rampart circuits, moats, platforms — not rooms
or walls). Expect false positives from paleochannels, salinity gradients,
plow lines, pivot-irrigation geometry."

- **Data engineering:** Sentinel-2 L2A only; SCL cloud/shadow masking per
  scene; reject scenes < 30-40% clear. Statistics: raw temporal variance AND
  stress-timed contrast (dry-year vs wet-year means, years ranked by regional
  scene-median NDVI).
- **Control design (corrected):** the decisive positive control is a FLAT,
  plowed-over catalog site (Menze-Ur mound-height = 0/NA, area >= 2 ha) under
  active cultivation — a mounded tell confounds subsurface signal with
  slope/drainage of the mound itself. Big-tell tier is retained only as an
  easy sanity tier. Matched negative controls: same crop, same irrigation
  district, no known structure.
- **Success metric:** footprint-vs-background AUC >= 0.8 on the control set.
  (Context: the measured all-channel ceiling for the general small-site
  population here is AUC 0.62 — a channel that fails its OWN big-site
  control is dead on arrival for sweeps.)

### Capability-2 control ledger (live)

- **Iteration 1 (2026-07-14): FAILED.** Densest cultivated Khabur tile,
  42-scene 2019-2025 spring NDVI stack, 6 big-tell + 63 flat sites.
  AUC: raw_std 0.478/0.483 (big/flat), stress_contrast 0.497/0.487 — chance.
- **Iteration 2 (pre-registered, running):** single change — background-
  removal scale (250 m -> 600 m -> none); prediction: AUC rises if the 250 m
  detrend was erasing site-scale signal. If flat-tier AUC stays ~0.5 across
  scales, one iteration remains before the capability is declared dead.
