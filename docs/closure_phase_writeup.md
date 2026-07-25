# Closure phase as a ranking channel for archaeological tell detection: a pre-registered, covariate-matched evaluation on free InSAR data

**Draft technical note — v0.1, 2026-07-24. Not peer-reviewed. Numbers are
reproducible from the repository; see Data & Code.**

---

## Abstract

Interferometric SAR *closure phase* — the residual of the phase loop
Φ₁₂ + Φ₂₃ − Φ₁₃ over three acquisitions — is an established observable for
soil-moisture retrieval. We ask whether it also carries information useful for
**archaeological tell detection**. Using free Sentinel-1 data processed through
ASF HyP3 and the Menze & Ur (2012) catalog of 14,324 sites as ground truth, we
measure the rank separability (AUC) of mean |closure| magnitude between known
flat tell sites and controls in the North Mesopotamian steppe. The effect is
small but robust: **AUC ≈ 0.60–0.62**, with the 95% confidence interval
excluding chance in every configuration. It **replicates** across two areas 90 km
apart (n = 141 and n = 733), **survives** matching controls on landscape
covariates (elevation, slope, topographic position), **persists** in the dry
season (arguing against a purely seasonal-moisture confound), and **survives**
Bonferroni correction across nine tested channels. We make no discovery claim
and no claim to competitive detection performance: this is a single, weak
*feature*, reported with its confidence intervals and its limits, that could be
added to an operational multi-feature detector such as Orengo et al. (2020).

---

## 1. Background and scope

Automated detection of archaeological mounds ("tells") from satellite data is an
active field. Menze & Ur (2012, PNAS) mapped ~14,000 settlement sites in NE
Syria by classifying anthrosol spectral signatures in multispectral time series
and measuring mound volume in a DEM. Orengo et al. (2020, PNAS) trained a
random-forest classifier on multitemporal SAR + multispectral composites to
produce a mound-probability field in Cholistan. Both are operational,
**multi-feature** systems and both report site counts and field/legacy
validation rather than a held-out separability metric.

Our contribution is deliberately narrow and different in kind. We evaluate **one
previously-unused single feature** — InSAR closure-phase magnitude — as a
**ranking channel**, and we report it with the statistical machinery (CIs,
covariate-matched controls, pre-registration, multiple-comparison correction)
that the operational-detection literature typically omits. We do **not** claim
to detect new sites, nor to outperform any existing system.

> **On novelty.** Closure phase for soil-moisture retrieval is well established
> (e.g. De Zan et al.; several 2018–2025 papers). We found no prior application
> of closure phase specifically to archaeological site ranking in a
> *non-exhaustive* literature search. That is absence of a found precedent, not
> a guarantee of novelty, and a full review is required before any first-use
> claim in a submitted version.

---

## 2. Data and methods

**SAR.** Sentinel-1 SLC scenes (free), interferograms produced with ASF HyP3
(INSAR_GAMMA). Unwrapped phase read directly from products.

**Closure.** For consecutive acquisition triplets (i, i+1, i+2) we form
C = Φ_{i,i+1} + Φ_{i+1,i+2} − Φ_{i,i+2}, referenced by subtracting the scene
median (removing the orbital/reference-pixel constant), and take the per-pixel
mean of |C| across all triplets. Implementation and unit tests:
`archaeo_intel/closure.py`, `tests/test_archaeo_closure.py`.

**Ground truth.** Menze & Ur (2012) catalog (Harvard Dataverse
doi:10.7910/DVN/7H8K3N). To avoid winning on topography or modern occupation,
sites are filtered to **flat, unoccupied** ones (area > 1 ha, mound height < 2 m
or unrecorded). Controls are drawn ≥ 0.5 km from any catalog site.

**Metric.** Rank AUC (site vs control), reported polarity-agnostically as
separability = max(AUC, 1−AUC), with Hanley–McNeil 95% CIs. Pre-registered pass
bar (set before any closure data existed): separability ≥ 0.60. All
registrations, controls, and verdicts — including two retracted invalid runs —
are timestamped in `docs/RESEARCH_TRACKS.md`.

---

## 3. Results

**Primary and replication.** Two areas, ~90 km apart, no spatial overlap,
identical recipe:

| AOI | n (per arm) | separability | 95% CI | site vs control median (rad) |
|---|---|---|---|---|
| AOI-1 (Khabur) | 141 | 0.619 | [0.554, 0.684] | 0.0595 / 0.0505 |
| AOI-2 (east) | 733 | 0.603 | [0.574, 0.632] | 0.0912 / 0.0688 |

**Covariate-matched controls.** Tells are not randomly sited; random controls
differ from them in landscape position before any buried structure is
considered. Re-running with controls matched to sites on elevation, slope, and
topographic position index (nearest-neighbour in z-space):

| AOI | random controls | matched controls |
|---|---|---|
| AOI-1 | 0.619 [0.554, 0.684] | **0.589 [0.523, 0.655]** |
| AOI-2 | 0.603 [0.574, 0.632] | **0.614 [0.585, 0.642]** |

The positional confound is real (sites do occupy distinct terrain positions) but
does not explain the signal: matched CIs still exclude 0.5. Best single estimate
after matching: **0.614 [0.585, 0.642]** (AOI-2, n = 733).

**Seasonal (mechanism) test.** Repeating AOI-1 in the dry season (Aug–Oct):

| season | separability | site / control median (rad) |
|---|---|---|
| wet (Mar–Jul) | 0.619 | 0.0595 / 0.0505 |
| dry (Aug–Oct) | 0.618 | 0.0282 / 0.0242 |

Separability is unchanged while absolute magnitudes roughly halve. The signal
persists outside the moisture-contrast window — evidence against a purely
seasonal moisture/vegetation confound — while the halved magnitude is consistent
with soil moisture as the *medium*. This does not isolate buried architecture
from a persistent anthrosol/soil-texture difference; it rules out a
season-specific artifact.

**Multiple comparisons.** Nine channels were tested against the same ground
truth. Bonferroni-corrected (null AUC = 0.5, Mann–Whitney null SE = 0.0344 at
n = 141): closure remains significant at p = 2.5×10⁻³; the family-wise
probability that *any* channel reaches 0.60 by chance is 0.016 (expected 0.02
channels vs 4 observed). The cluster of ≥0.60 channels is not a selection
artifact.

**Consistency.** Across two areas, two seasons, and random vs matched controls,
every measurement lands in **0.589–0.619** — a 0.03-wide band. This stability
across independent axes of variation is the strongest single argument that the
effect is real (if small).

---

## 4. Limitations

- **Small effect.** AUC ~0.61 is ranking-grade, not a detector. At n ≈ 141 the
  95% CI is ±0.065; pass/fail language against a 0.60 bar is not statistically
  meaningful and is avoided.
- **Partial independence of the replication.** AOI-2 shares track, frame, season
  and processing chain with AOI-1; it is a spatial, not fully independent,
  replication. A different-track and different-region test remains to be done.
- **Mechanism inferred, not demonstrated.** The seasonal test rules out a
  seasonal confound but cannot separate buried architecture from persistent
  anthrosol soil properties.
- **Arid-only.** Coherence collapses over vegetated/wet terrain; the method is
  restricted to drylands (consistent with independent findings in this project
  that InSAR-archaeology fails on vegetated earthworks).
- **Single analyst / self-administered.** Pre-registrations are git-timestamped
  but not externally registered; independent replication is warranted.

---

## 5. Conclusion

InSAR closure-phase magnitude carries a small, robust, positionally-controlled
signal that ranks known tell sites above matched controls (AUC ≈ 0.61) on free
data, and the signal persists out of season. It is not a detector and does not
compete with operational multi-feature systems; it is a candidate *feature* for
inclusion in one. The result is offered primarily as a methodological
demonstration — pre-registered, interval-reported, covariate-matched,
season-controlled, and multiple-comparison-corrected — of how a weak remote-
sensing channel can be evaluated honestly.

---

## Data & code

- Method + tests: `archaeo_intel/closure.py`, `tests/test_archaeo_closure.py`
- Full pre-registration/results ledger: `docs/RESEARCH_TRACKS.md`
- Adversarial self-review incl. all open weaknesses: `docs/CRITIQUE.md`
- Ground truth: Menze & Ur 2012, Harvard Dataverse doi:10.7910/DVN/7H8K3N

## References

Archaeological detection (prior art):
- Menze, B. H. & Ur, J. A. (2012). Mapping patterns of long-term settlement in
  Northern Mesopotamia at a large scale. *PNAS* 109(14), E778–E787.
- Orengo, H. A., Conesa, F. C., Garcia-Molsosa, A., et al. (2020). Automated
  detection of archaeological mounds using machine-learning classification of
  multisensor and multitemporal satellite data. *PNAS* 117(31), 18240–18250.

Closure phase / soil moisture (the established use of the observable):
- De Zan, F., Zonno, M., & López-Dekker, P. (2015). Phase inconsistencies and
  multiple scattering in SAR interferometry. *IEEE TGRS* 53(12), 6608–6616.
  [foundational: closure phase, soil-moisture-driven phase bias.]
- De Zan, F. et al. (2014). A SAR interferometric model for soil moisture.
  *IEEE TGRS* 52(1), 418–425.
- Zwieback, S. et al. — "Vegetation and soil moisture inversion from SAR
  closure phases" (Remote Sensing of Environment). [exact vol./pp. to confirm
  before submission.]
- Additional recent soil-moisture-from-closure-phase work (2023–2025):
  "Modeling, prediction, and retrieval of surface soil moisture from InSAR
  closure phase"; "Fine-Resolution Measurement of Soil Moisture From Cumulative
  InSAR Closure Phase" (RSE / TechRxiv). [full citations to confirm.]

*Note for submission:* the three "to confirm" entries above must be pinned to
exact volume/page/DOI before this note is circulated; the De Zan 2015 and
Menze & Ur / Orengo citations are verified.
