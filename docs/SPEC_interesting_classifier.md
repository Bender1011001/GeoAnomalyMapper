# BUILD SPEC: a general "worth a glance" classifier for satellite imagery

**Hand this whole file to the implementing agent. It is self-contained.**

---

## 0. What you are building, in one sentence

A ranked-shortlist system that looks at large volumes of free satellite/aerial
imagery and surfaces **anything a curious human would want to see** — not just
one phenomenon. Archaeology, weird geometry, industrial oddities, craters,
abandoned installations, unexplained scars, anything that is *out of place for
its surroundings*.

**The output is not a label. It is a queue.** Success = a human opens the top 100
and repeatedly says "huh, what is that?" Failure = the top 100 is noise, or is
all one boring category.

---

## 1. Context: what already exists (do not rebuild these)

Repo: `GeoAnomalyMapper`. Python. `pytest` suite, currently ~169 tests green.
Run tests with `python -m pytest tests/ -q`.

| Module | What it does | Reuse it |
|---|---|---|
| `archaeo_intel/data_access.py` | `stac_search(collection, bbox, ...)`, `read_grid(href, grid, w, h)` — fetches imagery from Microsoft Planetary Computer (NAIP 1 m, Sentinel-2 10 m, Copernicus DEM 30 m, Landsat) | **YES — this is your imagery layer** |
| `deformation_intel/context.py` | `agriculture_score`, `industrial_pad_score`, `is_cultivated_confound`, slope sampler, OSM proximity | **YES — these are working confound filters** |
| `deformation_intel/review.py` | `rank_for_review`, `contact_sheet` (labelled thumbnail grids), `build_review_package` | **YES — this is your review UX** |
| `deformation_intel/vlm_review.py` | OpenRouter client, `WIDE_PROMPT`/`FOCUS_PROMPT`, two-pass batching, cost estimator | **YES — this is your VLM layer** |
| `archaeo_intel/corona.py` | Declassified 1960s CORONA spy imagery (~2 m), georeferencing helpers | Useful for a **change-over-60-years** signal |
| `deformation_intel/sweep.py`, `opera.py` | InSAR ground-motion sweep (subsidence). | Only one *input signal*; do not let it define the project |

**Data sources available, all free, no Google:** NAIP (~1 m, US only, USDA —
*not* passed through commercial blurring), Sentinel-2 (10 m, global),
Copernicus DEM GLO-30, Landsat (1972→now), CORONA (~2 m, 1960s, partial global).

**OpenRouter is wired** (`OPENROUTER_API_KEY`). Live-priced options:
`qwen/qwen3-vl-235b-a22b-instruct` $0.21/M in (best open-weight),
`qwen/qwen3-vl-32b-instruct` $0.104/M (cheap), `anthropic/claude-opus-5` $5/M
(premium). Reviewing 60,000 chips individually on a cheap model ≈ **$10**; in
25-per-contact-sheet batches ≈ **$1.40**. Cost is a rounding error.

---

## 2. The actual architecture you should build

A **funnel**, cheap→expensive, with a human only at the end.

```
candidate locations (millions possible)
  └─ Stage 1  CHEAP NUMERIC PRIORS      (no imagery fetch)     → ~100k
  └─ Stage 2  IMAGERY FETCH + CV scores (1 chip per site)      → ~10k
  └─ Stage 3  CHEAP VLM on contact sheets (25/sheet)           → ~500
  └─ Stage 4  STRONG VLM on full-res chips                     → ~100
  └─ Stage 5  HUMAN reviews a contact sheet                    → the good ones
```

**The imagery fetch (Stage 2) is the real bottleneck, not model cost.** ~2–4 s
per chip serial = 33–67 h for 60k. Parallelise it (20 workers → ~2–3 h), cache
every chip to disk keyed by rounded lat/lon, and never fetch twice.

### Stage 1 — where do candidate locations come from?

Do **not** rely solely on the existing InSAR detector. Generate candidates from
multiple independent priors so you are not blind to static features:

1. **Grid sampling** of a region (e.g. every 500 m) — unbiased, expensive.
2. **DEM anomalies** — local relief/curvature outliers, closed depressions,
   suspiciously circular or linear topography (Copernicus GLO-30).
3. **Spectral oddity** — pixels whose Sentinel-2 spectral signature is an
   outlier vs their neighbourhood (bare-soil index, NDVI anomaly, brightness).
4. **Geometry detectors on imagery** — Hough circles/lines, straight-edge
   density, symmetry. (Existing `field_regularity_score` shows the pattern.)
5. **Change over time** — Sentinel-2 across years, and CORONA-1960s vs today.
   *Things that appeared or vanished are inherently interesting.*
6. **Ground motion** — the existing InSAR sweep. One signal among many.

### Stage 2 — computer-vision scores per chip (cheap, local, no API)

Compute a feature vector per chip. Suggested, extend freely:
- geometric regularity (lines, circles, right angles, symmetry)
- "out-of-placeness": how different is this chip from a random sample of chips
  *from the same region* (embedding distance or simple texture/colour stats)
- edge sharpness / man-made-ness
- existing confound scores: `agriculture_score`, `industrial_pad_score`, slope
- OSM emptiness: is anything mapped here? (unmapped + structured = interesting)

**Key idea: rank by NOVELTY RELATIVE TO SURROUNDINGS, not by absolute score.**
A circle in the desert is interesting; a circle in a centre-pivot field is not.

---

## 3. Hard-won lessons — violating these will waste you days

These are all real failures from this project. Take them seriously.

1. **Physical-plausibility filtering is mandatory.** Our detector's "top 20" was
   uplift at **515 cm/yr with 0.4 cm cumulative motion** — impossible, pure
   unwrapping noise. 3,899 of 4,258 "detections" were junk. *Always* cross-check
   that quantities which must agree, do agree. Add plausibility gates before
   ranking, or your shortlist will be 90 % artifacts.

2. **Synthetic tests are not validation.** An agriculture detector passed 6
   synthetic tests, then scored *barren Mojave desert* as cultivated on real
   imagery. It took **three rounds of real-data testing** to fix. Validate every
   scorer on real labelled chips, not just generated ones.

3. **A counter going up is not proof of work.** A sweep logged
   "granule 344/344" while writing **zero output files** — an auth failure was
   returning instantly. Always assert on *artifacts produced*, not progress logs.

4. **Test your discriminator on the hard negative, not the easy one.** The
   well-pad detector's first version used connected-component rectangularity and
   scored **0.00 on an actual well pad** — equipment and shadows fragment a pad
   into ~83 blobs. Centre-disc-vs-annulus brightness worked (6/6 on real chips).

5. **Clustering/neighbour rules must be computed among SURVIVORS, not among all
   raw detections.** Ours counted neighbours among 4,258 raw hits, so everything
   looked "clustered" and it vetoed the single best candidate in the dataset.

6. **Do not hard-veto on a single weak signal — rank instead.** A binary rule cut
   4,258 candidates to 1, discarding a 97 cm subsidence bowl for having the
   wrong classification label. Reserve hard vetoes for *definitive* explanations
   (visibly on a well pad, visibly cultivated). Everything else adjusts rank.

7. **General VLMs are weak on overhead imagery scale and orientation** (see
   VRSBench / GeoGround literature). Always state in the prompt: the ground
   width in metres, that north is up, and the sensor/resolution.

8. **Force a mundane explanation before an interest score.** Open-ended prompts
   otherwise manufacture mysteries. Our prompt requires "most likely ordinary
   explanation" first, then what does *not* fit.

---

## 4. Deliverables

1. `interesting_intel/` package (new), with:
   - `priors.py` — candidate generation (§2, at least 3 independent priors)
   - `features.py` — pure CV scorers, no network, unit-tested
   - `pipeline.py` — the funnel with disk caching + resumability + parallel fetch
   - `cli.py` — `python -m interesting_intel --bbox ... --out DIR`
2. Tests in `tests/`, following existing style. **Every scorer needs at least one
   real-imagery test**, not only synthetic.
3. `docs/INTERESTING_FINDINGS.md` — ranked output with thumbnails and, for each,
   the mundane explanation + what doesn't fit.
4. Cost + runtime report: actual $ and hours for the region you ran.

## 5. Validation you must do before claiming it works

- **Positive controls:** seed known-interesting sites and confirm they rank
  highly. Suggestions: Nazca lines (−14.739, −75.130), Racetrack Playa
  (36.681, −117.563), a Minuteman silo field (~47.0, −101.5), Sedlec/large
  quarries, Bingham Canyon Mine (40.523, −112.151), Richat Structure
  (21.124, −11.401). **If these do not surface, the system does not work.**
- **Negative control:** a large boring area (e.g. central Kansas cropland). The
  top hits there should be visibly dull — if it "finds" as much there as in a
  rich area, you are ranking noise.
- **Report precision honestly:** of your top 100, how many did a human agree were
  worth the glance? That number is the product.

## 6. Explicit non-goals

- Do not build another subsidence detector.
- Do not claim discoveries. Everything is a *lead* until verified.
- Do not publish precise coordinates of archaeological sites in conflict zones
  (existing project rule: round to ~0.1° publicly, keep precise data local).
- Do not optimise for a benchmark. Optimise for "a human said huh".
