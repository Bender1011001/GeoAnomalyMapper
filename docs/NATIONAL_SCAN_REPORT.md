# National Deformation Scan — Consolidated Report (2026-07-10)

12 high-value AOIs (~24×24 km each), OPERA DISP-S1 time series 2016→2025,
36 epochs per target, analyzed with detector v4: calendar-true era stitching,
noise-adaptive thresholds, localized-vs-regional Mogi discrimination,
quiet-ground aquifer-correlation rejection, bootstrap depth ranges, and
early/late source-growth labels. Per-target details: `data/national_scan/
<target>/findings_v3.json`; summary: `data/national_scan/NATIONAL_REPORT_v3.json`.

## Results

| Target | Anomalies | Localized void candidates | Top candidate | Reading |
|---|---|---|---|---|
| Scranton anthracite, PA | 505 | **130** | −49.6 cm/yr @ **42 m** (41.3428, −75.7151) | Shallow depths match anthracite workings; most active mine-subsidence field in the set |
| Hutchinson salt, KS | 424 | 71 | −4.9 cm/yr @ 731 m (38.1257, −98.0626) | Active salt district; top depth deeper than local salt — treat depth with its CI |
| Central Valley, CA | 264 | 63 | −22.8 cm/yr @ 153 m | Known aquifer province; candidates here are pumping cones (no quiet ground for rejection — see caveats) |
| Houston, TX | 264 | 62 | −7.2 cm/yr @ 97 m (29.7170, −95.2482) | Fault-block + groundwater subsidence; industrial SE corridor |
| Carlsbad brine, NM | 114 | **48** | −3.6 cm/yr @ **394 m** (32.4687, −104.2920) | Depth lands in the Salado/Castile dissolution section; 11 km NNW of the remediated I&W well |
| Tampa/Spring Hill, FL | 238 | **45** | −17.2 cm/yr @ **111 m** (28.3465, −82.4136) | Sinkhole Alley; karst-depth accelerating bowls — flagship insurance-relevant result |
| Wink/Kermit, TX | 55 | 18 | −3.8 cm/yr @ **459 m** (31.7754, −103.1270) | Ground-truth site; top candidate in the validated sink complex at Salado depth |
| Pecos, TX | 128 | 13 | −6.9 cm/yr @ 169 m | Oilfield terrain correctly dominated by regional/uplift classes |
| Long Beach, CA | 54 | 5 | −5.9 cm/yr @ 160 m (33.7606, −118.2459) | Wilmington oilfield/harbor; small candidate set |
| The Villages, FL | 283 | 2 | −2.9 cm/yr @ 360 m | Mass "uplift" flags are a referencing-artifact suspect — do not interpret |
| Retsof, NY | 45 | **0** | — | 1994 mine collapse stabilized — clean result matches history |
| Bayou Corne, LA | 181 | **0** | — | 2012 sinkhole stabilized; swamp coherence limits |

**Totals: ~2,555 classified anomalies; 457 localized void candidates.**

## Validation anchors inside the scan

- Wink's top candidate sits in the independently validated sink complex, at the
  documented salt depth.
- Retsof and Bayou Corne — both *stabilized* historic collapses — correctly
  return **zero** candidates: the detector does not invent signal where ground
  truth says motion ended.
- Central Valley's aquifer province is correctly majority-classified regional.

## Caveats (read before acting on any candidate)

1. Candidates are **leads for investigation**, not confirmed voids. Ground
   methods (microgravity/ERT/records search) confirm; the satellite ranks.
2. **Aquifer rejection needs quiet ground.** In wall-to-wall subsiding basins
   (Central Valley, parts of Houston) there is no stable reference, so pumping
   cones can still score as candidates. External groundwater/well data is the
   remaining discriminator for those basins.
3. Mass-uplift fields (The Villages, Bayou Corne) are referencing artifacts
   until proven otherwise.
4. LOS single-geometry rates; depth estimates carry the bootstrap ranges in the
   per-target JSONs — use the ranges, not the point values.
5. Coverage-hole lists (fast-motion fingerprints) are lead-generation only;
   intersect with known salt/mine locations before spending HyP3 credits.

## What to do next with this

- Field-verifiable shortlist: Scranton (shallow, fast, dense), Tampa (insurance
  market), Carlsbad brine (regulatory hazard precedent).
- HyP3 10×2 follow-ups on coverage holes at Hutchinson/Wink/Carlsbad.
- Groundwater-context layer to de-confound the basin targets.
