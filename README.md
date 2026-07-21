# GeoAnomalyMapper

**Ground-deformation intelligence from free satellite radar.** Detects, classifies,
and forecasts active subsurface processes — growing sinkholes, dissolving salt,
settling mine workings, induced subsidence — by mining NASA's InSAR archives with
a validated, test-covered analytics engine.

## What this system actually does (and doesn't)

Every capability claim below is backed by a ground-truth experiment. This project
previously explored other detection approaches; where they failed validation they
were removed, and where claims were never verified they are labeled as such. The
history is documented because it defines what the surviving system can honestly
claim.

| Approach | Ground-truth test | Status |
|---|---|---|
| **InSAR surface deformation (current system)** | Wink TX sinkholes (2× independent pipelines), Tampa "Sinkhole Alley", Central Valley discrimination | **Validated** — documented active subsidence recovered at 0.69 km; noise floor 0.19 cm/yr over 9.5 yr |
| Single-pass SAR Doppler "vibrometry" (Biondi-style) | Carlsbad Caverns vs barren-plains control, real Sentinel-1 SLC | **Failed & removed** — outputs were site-independent artifacts (both sites ~97% "void") |
| Public gravity + magnetics fusion (void detection) | Raw-data contrast at 14 known caves/mines | **Failed at feature scale** — 2–20 km resolution carries zero information about ~100 m voids |
| District-scale mineral prospectivity (legacy gravity line) | None reproducible in this repository | **Unverified** — historical claims (enrichment/hit-rate figures) are not backed by auditable artifacts; do not use until re-derived through the blind-validation harness |

**Honest physics:** this system sees ground that is **moving**. An actively
growing void, dissolving salt bed, or compacting aquifer produces measurable
surface motion (mm–cm/yr). A static, finished cavity (a stable cave, a completed
underground facility) produces none and is invisible to this — and to every other
public-data method we tested.

## Architecture — one system, layered channels

```
deformation_intel/
├── opera.py       OPERA DISP-S1 time-series ingestion (9.5-yr histories,
│                  lazy AOI windowed reads, 3-tier acquisition fallback,
│                  retry + on-disk caching, reference-era stitching)
├── timeseries.py  per-pixel velocity, ACCELERATION (collapse precursor),
│                  seasonal separation, regime-change detection, forecasting
├── sources.py     Mogi source inversion: bowl geometry -> source depth +
│                  volume-change rate (void vs aquifer discriminator)
└── detect.py      unified detector: cluster -> classify {accelerating /
                   steady / regional / seasonal / uplift} with confidence,
                   plain-language rationale, void_likelihood, forecast

tools/insar_prototype/   HyP3 short-pair stacks — the complementary channel
                         for FAST deformation that OPERA's quality masking
                         deletes (proven necessary at Wink)
archaeo_intel/           archaeology surface-proxy channel: DEM prominence
                         ranking -> VLM triage -> human review, plus the
                         CORONA 1960s film module (see below)
blind_validation.py      methodology-agnostic blind validation harness
                         (frozen candidates, withheld labels, hash-pinned
                         reproducibility) + geoanomaly.py CLI
```

The two motion channels are complementary by physics: OPERA gives mm-precision
9.5-year histories for slow/moderate motion (and therefore *prediction* —
acceleration and time-to-threshold), while on-demand HyP3 interferograms catch
fast, fresh deformation that decorrelates out of OPERA's masks.

## CORONA: free 1960s ~2 m spy imagery in five lines of Python

Probably the most immediately reusable piece of this repo:
[`archaeo_intel/corona.py`](archaeo_intel/corona.py) reads the CAST **CORONA
Atlas** open archive — 217 declassified US reconnaissance missions
(1960-72, KH-4B at ~1.8 m) — with no account, no bulk downloads:
whole-strip previews in ~2 s via HTTP range reads, full-resolution windowed
crops, a ground-control-point workflow with per-point residual QC, panoramic
distortion fitting, and warping straight to a QGIS-ready GeoTIFF. CLI
included. The frames pre-date mechanized agriculture and modern conflict, so
archaeological landscapes long erased on the ground are often crisply
visible. **Tutorial: [docs/CORONA.md](docs/CORONA.md).** Credit the
[CORONA Atlas project](https://corona.cast.uark.edu) (CAST, University of
Arkansas) when you use the data — and be kind to their bandwidth.

## Archaeology channel and the research program

[`archaeo_intel/`](archaeo_intel/README.md) applies the same
validated-detector discipline to landscape archaeology: DEM-prominence
ranking plus VLM triage, measured against a 14,324-site ground-truth catalog
(Menze & Ur 2012, Upper Khabur). The measured free-data ceiling is honest and
published in that README (best single feature AUC 0.639, fusion ceiling
~0.62): free data *ranks* candidates for human review; it does not discover
sites autonomously. Candidate coordinates in conflict zones are never
published (public artifacts round to ~11 km).

Since 2026-07-15 a pre-registered research program
([docs/RESEARCH_TRACKS.md](docs/RESEARCH_TRACKS.md), governed by
[docs/DISCOVERY_SOP.md](docs/DISCOVERY_SOP.md)) has mined signal that
standard pipelines discard. Validated so far: a closure-phase disturbance
detector with a ~6-week early-warning lead, SAR asc/desc anisotropy as the
best single archaeology ranking feature (replicated), TDA persistence as a
ring-form feature, and CORONA human-GCP georeferencing. Dead ends are
recorded with the same care as wins — see the status summary at the top of
that file.

## Validated results

- **Wink, TX (natural ground truth)** — the documented actively-subsiding
  salt-dissolution area east of Wink Sink 2 was blindly recovered by both
  channels: HyP3 stack (−8.6 cm/yr bowl, 0.69 km from the published location)
  and OPERA time series (dominant accelerating cluster, automatic 2023
  regime-change detection).
- **Tampa / Spring Hill, FL ("Sinkhole Alley")** — 42 localized, accelerating,
  Mogi-consistent subsidence candidates at karst depths, in the most
  sinkhole-prone (and sinkhole-insured) region of the US.
- **Central Valley, CA (discrimination test)** — the detector correctly labels
  the famous aquifer-compaction province as *regional* subsidence (201 of 264
  detections), rather than misreporting it as void collapse.

Records of these experiments (including the failed-approach control runs) are in
`docs/experiment_records/`; see [docs/README.md](docs/README.md) for the full
documentation index.

## Quickstart

```bash
pip install -e .
# Earthdata credentials (free account: https://urs.earthdata.nasa.gov)
cp .env.example .env   # fill EARTHDATA_USERNAME / EARTHDATA_PASSWORD

# health check (no downloads)
python geoanomaly.py health --json --skip-gpu

# unit tests for the analytics engine (fast, synthetic-signal round trips)
pytest tests/test_deformation_timeseries.py tests/test_deformation_sources.py \
       tests/test_deformation_detect.py -q
```

Building an AOI cube and detecting anomalies (Python):

```python
from deformation_intel.opera import build_aoi_cube
from deformation_intel.detect import detect_anomalies

cube = build_aoi_cube(31.769, -103.102, half_width_km=12.0,
                      cache_dir="window_cache")
for a in detect_anomalies(cube)[:10]:
    print(a.rank, a.classification, a.peak_velocity_cm_yr, "cm/yr",
          a.source_depth_m, "m", a.why)
```

## Data sources (all free)

- **OPERA DISP-S1** (NASA/JPL): validated L3 InSAR displacement time series over
  North America, 2016→present, 30 m — the primary channel.
- **ASF HyP3**: on-demand Sentinel-1 interferometry (monthly credit quota) — the
  fast-deformation channel.
- **Sentinel-1 SLC** via ASF (`slc_data_fetcher.py`) for custom processing.

## Known limitations

- LOS, single-geometry velocities (ascending/descending decomposition is on the
  roadmap); rates are line-of-sight, not pure vertical.
- The localized-vs-regional discriminator reduces, but does not eliminate,
  aquifer-pumping false positives; a groundwater/seasonality rejection layer is
  the next planned addition.
- Anomaly lists are *candidates for investigation*, not confirmed voids.
  Confirmation requires ground methods (microgravity, ERT, drilling).
- Run the deformation tests in a separate pytest process from any torch-based
  suites (native DLL load-order conflict on Windows).

## License

Proprietary — see [LICENSE](LICENSE).
