# archaeo_intel — Archaeological Surface-Proxy Intelligence

Screens landscapes for *candidate* archaeological sites using only free public
satellite data, with the same discipline as `deformation_intel/`: every
analytic has a synthetic unit test, every channel must reproduce its positive
control before touching unknown ground, and every performance number below was
measured against real ground truth — the Menze & Ur 2012 Upper Khabur catalog
(14,324 sites, the most complete inventory of any Mesopotamian landscape).

Governance lives in [docs/DISCOVERY_SOP.md](../docs/DISCOVERY_SOP.md)
(triage budgets, auto-explain before human review, provenance, the N=3 kill
switch). Experiment-by-experiment verdicts live in
[docs/RESEARCH_TRACKS.md](../docs/RESEARCH_TRACKS.md).

## Modules

- `catalog.py` — Menze-Ur ground-truth catalog loader (Harvard Dataverse;
  beware: the CSV's columns are UTM-37N northing/easting despite lat/lon
  headers). Validation: Tell Brak matches at 117 m; 4/4 of this system's
  confident detections matched catalog sites at 41–373 m.
- `data_access.py` — lazy windowed reads (no bulk downloads) from Earth
  Search STAC (AWS) and Microsoft Planetary Computer (SAS signing, for
  collections like Landsat C2L2 thermal).
- `composite.py` — multi-temporal Sentinel-2 median composites
  (Orengo & Petrie style): cancels this year's crop/moisture/plough state,
  keeps persistent soil signatures.
- `detect.py` — the analytics: topographic `prominence` (the primary mound
  signal), robust z-scores, regional roughness, radial hollow-way alignment.
- `triage.py` — VLM chip triage (ancient / modern / natural) with prompt v3,
  A/B-validated on labeled anchor chips: 19/20 vs the old prompt's 10/20,
  which was *systematically* wrong on village-topped and bare excavated tells.
- `corona.py` — the CORONA 1960s ~2 m film archive: 2-second whole-strip
  quicklooks via HTTP range reads, full-res windowed crops, human-GCP
  georeferencing with residual QC, GeoTIFF export, CLI. Tutorial:
  [docs/CORONA.md](../docs/CORONA.md).

## Measured channel performance (free-data ceiling, honestly stated)

Population-scale AUC on the Khabur catalog (site vs matched control):

| Channel | AUC | Role |
|---|---|---|
| DEM prominence | 0.547 | Finds **large** mounds only; the ranking backbone (4/4 external validation) |
| BSI multi-temporal composite | ~0.55 | 2.3x contrast at big tells; not a small-site detector |
| Backscatter texture | 0.595 | weak |
| Landsat thermal composite | 0.622 | best single *raster* feature (100 m) |
| Combined ML (5-fold CV) | 0.616 | the measured free-data fusion ceiling |
| **Asc/desc SAR anisotropy** | **0.639 / 0.608 (replicated)** | best single feature; **ranking only** — as an autonomous peak sweep it detects buildings/ruins/cliffs, 12/12 confounders |
| TDA H1 persistence | ring-sensitivity feature | validated on ring forms; not autonomous at 30 m DEM |
| Closure phase (HyP3) | validated disturbance detector | ~6-week early-warning lead over coherence drop |

The anisotropy, TDA, and closure-phase analytics currently live as preserved
experiment scripts (`data/research/scripts/`, local-only), not yet as package
modules — see the RESEARCH_TRACKS status summary before reusing them.

What these numbers mean in practice: free data ranks and triages, humans
confirm. AUC ~0.6 channels are useful for *ordering* a review queue, never
for autonomous discovery claims. The measured resolution walls: 10 m optical /
30 m DEM cannot resolve 150–200 m ring forms or 3–5 m mounds reliably;
breaking them needs paid tasking (TanDEM-X) or the un-built CORONA stereo
DEMs. CORONA's unique value is the 60-year time baseline: it demoted a crisp
"ring" candidate by showing the annulus did not exist in 1967.

## Ethics and disclosure (binding, from the SOP)

- Candidate coordinates in looting-vulnerable regions are **never published**;
  public artifacts round to ~11 km, findings route to heritage authorities.
- Output register is always "candidate", never "discovery".
- Anything promoted from local experiment outputs into this public repo gets
  vetted for precise coordinates first.

## Tests

`pytest tests/test_archaeo_intel.py -q` — synthetic-signal round trips for
every analytic plus the Tell Brak positive-control reproduction.
