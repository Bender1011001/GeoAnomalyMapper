# National Deformation Scan — Full-Archive Report (2026-07-11)

12 high-value AOIs (~24×24 km), NASA OPERA DISP-S1 time series 2016→2025,
processed with the **complete epoch archive** (149–434 epochs per target) and
detector v5: calendar-true era stitching, per-pixel temporal-coverage gate,
robust cluster fits, cluster-mean acceleration, rate-vs-cumulative
self-consistency, localized-vs-regional Mogi discrimination, and bootstrap depth
ranges. Every headline number below was independently verified against the raw
pixel series. Per-target detail: `data/national_scan/<target>/findings_v4_full.json`
and `map_final.png`; summary: `data/national_scan/NATIONAL_REPORT_final.json`.

> **This supersedes the earlier 36-epoch scan.** That sparse pass claimed 457
> candidates; most were artifacts from stitching drift on under-sampled epochs.
> The full archive plus reliability/coverage filtering reduces this to **35
> verified localized candidates** — a ~13× cull, with each survivor consistent
> with its raw cumulative displacement.

## Results (verified)

| Target | Epochs | Candidates | Top verified candidate | Reading |
|---|---|---|---|---|
| Scranton anthracite, PA | 194 | **16** | −24.5 cm/yr* @ 55 m (41.426, −75.600) | Active abandoned-mine subsidence; strongest field. *peak-pixel rate inflated; cluster ~−3 cm/yr |
| Wink/Kermit, TX | 269 | **6** | cluster to −88 cm cumulative (31.78, −103.13) | **Validation anchor** — recovers the documented sink complex (5/6 within 3.5 km of Sink 1) |
| Houston, TX | 434 | **5** | −1.9 cm/yr @ 76 m (29.854, −95.260) | Fault-block + groundwater subsidence |
| Tampa/Spring Hill, FL | 269 | **4** | −6.8 cm/yr @ karst depth (28.443, −82.423) | Sinkhole Alley; 2 accelerating karst movers (low OPERA coverage — HyP3 follow-up) |
| Long Beach, CA | 371 | **2** | −2.4 cm/yr abs @ 81 m (33.796, −118.248) | Tight subsidence bowls inside the injection-*uplifted* Wilmington oilfield |
| Hutchinson salt, KS | 254 | **1** | −2.3 cm/yr abs, −20.8 cm cum (38.029, −97.873) | One genuine salt-district bowl (was "71" on sparse data — artifacts) |
| Retsof, NY | 149 | **1** | ~−1 cm/yr, −13.6 cm cum (42.764, −77.799) | Real *residual* settlement at the 1994 salt-mine collapse; steady, decelerating |
| Carlsbad brine, NM | 254 | **0** | — | "48" on sparse data were artifacts; old top spot is actually *rising*. 206 masked coverage-holes = HyP3 targets |
| Bayou Corne, LA | 254 | **0** | — | Clean negative — stabilized 2012 collapse; swamp-coherence noise culled |
| The Villages, FL | 239 | **0** | — | Prior "uplift field" was a referencing artifact; quiet |
| Pecos, TX | 269 | **0** | — | Oilfield correctly dominated by regional/uplift classes |
| Central Valley, CA | 374 | **0** | — | Most extreme *real* subsidence in the set (−10 to −18 cm/yr abs) but broad-regional → correctly not localized voids; only 11.5% of pixels measurable (ag decorrelation) |

**Total: 35 verified localized candidates across 12 targets.**

## Validation & negative controls (the trust anchors)

- **Wink** (independently validated): the detector recovers the sink complex as a
  tight cluster of raw-consistent subsidence bowls (−22 to −88 cm cumulative)
  2–3.5 km from the documented Sink 1 — blind, through the full archive.
- **Bayou Corne** (stabilized 2012 collapse): **0** candidates — correct.
- **Central Valley** (aquifer province): **0** localized candidates — the
  localized-vs-regional discriminator correctly declines to call groundwater
  compaction a void field.
- The self-audit this pass discarded artifacts at Hutchinson (71→1), Carlsbad
  (48→0), Tampa (45→4), Scranton (130→16) and reclassified Retsof from "clean
  zero" to real residual settlement.

## Caveats (read before acting on any candidate)

1. Candidates are **leads for ground follow-up** (microgravity / ERT / records),
   not confirmed voids.
2. **Depths are the least-reliable output** — Mogi point-source fits scatter on
   complex/fast bowls (Wink 272–1046 m). Use the bootstrap ranges in the JSONs,
   and weight detection + rate over depth.
3. Rates are **relative to the AOI median** unless marked "abs"; in uplifting
   fields (Long Beach) a negative relative rate can be modest absolute motion.
4. Peak-pixel rates can exceed the cluster rate (Scranton −24.5 peak vs ~−3
   cluster) — trust the cluster.
5. Low-coverage targets (Tampa karst, Central Valley ag) undersample the fast
   movers OPERA masks — intersect with HyP3 short-pairs before spending credits.

## What to do next

- Field-verifiable shortlist: **Scranton** (dense, shallow, active),
  **Tampa** (insurance market), **Wink** (validated, for methodology).
- HyP3 10×2 short-pair follow-ups on Carlsbad's 206 coverage-holes and Tampa's
  low-coverage accelerating bowls.
- Groundwater-context layer to de-confound the basin targets (Houston, Central
  Valley).
