# deformation_intel — Unified Subsurface Deformation Intelligence

One system, multiple sensing channels, layered into ranked, classified,
explainable ground-deformation anomalies. Built to detect and **forecast**
active subsurface processes (sinkhole/void growth, mine settling, induced
subsidence) from free satellite radar.

## Why deformation (and not the earlier approaches)

Ground-truth validation retired two approaches at feature scale: single-pass
SAR Doppler vibrometry (couldn't tell Carlsbad from barren farmland) and
public gravity/magnetic fusion (too coarse — see the repo's audit notes). The
surviving, twice-validated capability is **surface-deformation InSAR**: a
growing subsurface void, dissolving salt, or collapsing workings moves the
ground, and that motion is measurable to mm/yr.

## Channels (each covers the others' blind spots)

- **OPERA DISP-S1 time series** (`opera.py`) — NASA/JPL validated PS+DS
  displacement, ~9.5-year history (2016→), mm-precision, the primary channel.
  Best for slow/moderate, *predictive* signals. Blind to the fastest movers
  (its quality mask deletes them) and to the most recent weeks (latency).
- **HyP3 short-pair stacks** (`tools/insar_prototype/`) — on-demand
  interferograms for fast/fresh deformation OPERA masks out. Complementary,
  not redundant — proven at Wink where OPERA had 0 valid epochs on the fastest
  bowls.
- **Ascending + descending** decomposition (planned) — separates true vertical
  settling from horizontal motion, killing a class of false positives.

## Layers

- `timeseries.py` — per-pixel robust velocity, **acceleration** (collapse
  precursor), annual-seasonal separation (groundwater vs monotonic collapse),
  **regime-change/breakpoint** detection, **time-to-threshold forecasting**,
  and OPERA reference-era **stitching** (resets → one continuous series).
- `sources.py` — **Mogi source inversion**: bowl geometry → source **depth**
  and **volume-change rate**. Turns "the ground is sinking" into "a source at
  ~D m is losing ~V m³/yr" — the physical discriminator between a shallow
  collapsing void and broad deep aquifer compaction.
- `detect.py` — the unification point: cluster → classify
  {accelerating_subsidence, steady_subsidence, seasonal_dominated, uplift} with
  a confidence, a plain-language *why*, a source estimate, and a forecast.
  Optional `context_samplers` attach rejection/context layers (groundwater,
  lithology/karst, land use) per anomaly without bloating the core.

Gravity/magnetics are deliberately **not** in the detector (measured zero
contrast at void scale); they remain regional context only.

## Data access is production-grade

`opera.py` streams AOI windows (not 360 MB full frames) with a three-tier
acquisition path proven necessary by ASF's split hosting: cloud lazy S3 →
authenticated HTTPS → download-extract-delete fallback, all with
retry+backoff and on-disk window caching so a run resumes instead of
restarting. A single 503 or a datapool-only frame no longer kills a 269-epoch
build.

## Validation

- **Wink TX (natural ground truth):** the documented active salt-dissolution
  subsidence complex is recovered as the dominant accelerating-subsidence
  cluster, with an automatic 2023 regime-change detection. Noise floor
  0.19 cm/yr over the 9.5-year record.
- **Unit tests** (`tests/test_deformation_*.py`): 26 synthetic-signal tests
  that plant known velocities/accelerations/breakpoints/Mogi sources and
  assert recovery within tolerance. They have already caught real bugs
  (spurious breakpoints, unit errors) during development.

## Status

Research capability, validated on one natural analog. Not yet a certified
monitoring service — needs multi-site validation, asc/desc decomposition, and
the context/rejection layers wired to real datasets before commercial use.
