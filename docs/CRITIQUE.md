# Self-Critique — GeoAnomalyMapper
### Adversarial review of the project's own claims, 2026-07-22

This document exists to attack the project's results as hard as an unfriendly
reviewer would. It is written into the public repo deliberately: a program whose
selling point is validation discipline should publish its own weaknesses, not
only its passes.

Findings are ordered by severity. Each states the issue, why it matters, and
what would fix it.

---

## FIX STATUS (updated 2026-07-22, remediation in progress)

| # | Issue | Status |
|---|---|---|
| 1.1 | No confidence intervals | **FIXED** — `auc_ci()` added to `archaeo_intel/closure.py`; `separability()` now returns se/ci_low/ci_high; ledger retro-annotated with all CIs |
| 1.2 | Replication not fully independent | OPEN — needs a different-track / dry-season run |
| 1.3 | Controls not covariate-matched | **FIXED & TESTED** — pre-registered matched-control test run on **both** AOIs. Signal **SURVIVES**. New headline **0.614 [0.585, 0.642]** (AOI-2 matched, n=733) |
| 1.4 | Moisture seasonality uncontrolled | OPEN — dry-season run designed, not yet run |
| **1.6** | **Novelty claim was wrong** | **RETRACTED & FIXED** — closure-phase soil-moisture retrieval is a mature published field; "discarded as error" framing deleted; novelty narrowed to the archaeology *application*, stated as a non-exhaustive search. **Publication-critical.** |
| 1.5 | No family-wise correction | OPEN |
| 2.1 | Tampa depths contradict FL karst physics | **FIXED** — depths **withdrawn** from the brief with an explanation; observables retained |
| 2.2 | Mogi fits underdetermined at 8–9 px | **FIXED in the brief** (no depth/volume claimed); the detector still reports them internally |
| 2.3 | Ring registration ≈ feature size | OPEN — rings 9/25/28 remain provisional |
| 2.4 | Ring 32 unblinded reading | OPEN |
| 2.5 | "Not in catalog" ≠ unknown | **FIXED** — wording corrected in the write-ups |
| 2.6 | Mojave filter circularity | OPEN |
| 3.1 | Most analyses not reproducible from repo | PARTIAL — agriculture + closure are tested modules; others still gitignored |
| 3.2 | No recall/false-negative testing | **FIXED & MEASURED** — injection testing gives a detection envelope: hard size floor ~90 m across, rate floor ~2–3 cm/yr. Desert null is now bounded; Tampa brief updated. (v1 harness retracted — it measured coverage, not recall) |
| 3.3 | Self-administered validation | OPEN (structural) |
| 3.4 | Language outruns statistics | **FIXED** — CI language adopted; see 1.1 |

---

## TIER 1 — Issues that weaken the headline claims

### 1.1 No confidence intervals were ever computed. The "0.60 bar" is over-precise.

Every AUC in this project has been reported as a bare point estimate. Computing
Hanley–McNeil standard errors for the first time (2026-07-22):

| Channel | AUC | n (per arm) | 95 % CI | > chance? | **significantly > 0.60?** |
|---|---|---|---|---|---|
| Closure AOI-1 | 0.619 | 141 | [0.554, 0.684] | **yes** | **NO** |
| Closure AOI-2 (replication) | 0.603 | 733 | [0.574, 0.632] | **yes** | **NO** |
| Anisotropy original | 0.639 | 141 | [0.575, 0.703] | **yes** | **NO** |
| Anisotropy replication | 0.608 | 150 | [0.544, 0.672] | **yes** | **NO** |
| Thermal | 0.622 | 141 | [0.557, 0.687] | **yes** | **NO** |
| Combined-ML "ceiling" | 0.616 | 141 | [0.551, 0.681] | **yes** | **NO** |

**What this means, precisely:**
- The good news is real: **every channel's CI excludes 0.5**, so these are genuine
  above-chance effects, not noise. The closure replication (n = 733) is the most
  robust single result in the program.
- The bad news is a framing error: **not one of them is statistically
  distinguishable from the 0.60 pass bar.** "Clears the pre-registered 0.60 bar"
  describes a point estimate whose CI comfortably includes 0.55. A run that scored
  0.58 and "failed" would not be statistically different from one that scored 0.62
  and "passed."

**Impact.** This does not invalidate the findings — it invalidates the *precision
of the pass/fail language* used throughout the ledger. The honest statement is
"a real but small effect, ~0.60 ± 0.06," not "clears 0.60."

**Fix.** Report CIs everywhere. Re-express pass criteria as "CI lower bound
> 0.55" or similar, which is an actually-testable claim. Retro-annotate the ledger.

### 1.2 The closure-phase "independent replication" is only partially independent

Run 2 is presented as an independent replication. It is independent in **sites**
(733 different sites, ~90 km away, no spatial overlap) — but it shares:

- the **same satellite track and frame** (50 descending, frame 471),
- the same **orbit geometry and look angle**,
- the same **season** (Mar–Jul 2023) and largely the same dates,
- the same **processing chain** (HyP3 GAMMA, same parameters),
- the same **analyst and code**.

A systematic that is correlated with terrain or land-cover and common to that
track/season would reproduce in both runs. This is a *spatial* replication, not an
independent one.

**Impact.** The replication is meaningfully weaker than the word implies. It rules
out "one-tile fluke"; it does not rule out a track- or season-specific systematic.

**Fix.** Replicate on a **different track** (ascending 43 or 145 both contain the
AOI), a **different season** (a dry-season run is the sharper test — see 1.4), and
ideally a different region entirely.

### 1.3 Controls are not matched on landscape position — a live confound

Sites are catalog tells filtered to "flat, unoccupied" (area > 1 ha, height < 2 m).
Controls are random points ≥ 0.5 km from any catalog site. That controls for
*proximity to known sites* and nothing else.

But tells are **not randomly placed on the landscape**. They sit near water, on
particular soils, on route corridors, on slightly elevated ground, near drainage.
Random steppe controls differ from tells in all of those variables *before any
buried architecture is considered*.

**Impact.** This affects **every archaeology channel in the program**, not just
closure phase. An AUC of ~0.60 is exactly the magnitude one would expect from a
landscape-position confound (soil moisture regime, drainage proximity), so the
measured signal may be substantially "tells sit in different places" rather than
"buried architecture changes the signal."

**Fix.** Covariate-matched controls: match each site to a control on elevation,
slope, distance-to-drainage, and soil/geology unit. If the AUC survives matching,
the architecture interpretation is much stronger. This is the single highest-value
experiment the project could run next, and it is free.

### 1.4 Moisture seasonality is uncontrolled — the mechanism test was never run

Closure phase responds to soil moisture. Both closure runs used the same
**Mar–Jul moisture-contrast season**, chosen deliberately to maximize signal. The
inferred mechanism (buried architecture alters moisture retention) predicts the
signal should **persist, weaken, or invert in a dry season** in a specific way.
That prediction was never tested.

**Impact.** The mechanism remains inferred, as stated — but a cheap decisive test
exists and was skipped.

**Fix.** Run the identical recipe on a dry-season window (Aug–Oct). A signal that
vanishes entirely suggests seasonal-moisture/vegetation confound; one that persists
argues for a structural cause.

### 1.6 THE NOVELTY CLAIM WAS WRONG — retracted 2026-07-22 (publication-critical)

Throughout this project I repeatedly wrote that closure phase is "the signal
every InSAR pipeline discards as calibration error" and that "no prior use of
closure phase for archaeology is known." **I had never actually searched the
literature.** A search finally done on 2026-07-22 shows:

**Closure phase for soil moisture is a mature, active research field.** It is
not discarded and it is not obscure. Representative published work includes
De Zan et al. on whether InSAR coherence and closure phase can estimate soil
moisture changes; "Vegetation and soil moisture inversion from SAR closure
phases"; "Modeling, prediction, and retrieval of surface soil moisture from
InSAR closure phase"; "Fine-Resolution Measurement of Soil Moisture From
Cumulative InSAR Closure Phase"; and IEEE work on estimating soil moisture from
interferometry with closure phases. Triplet-closure stacking is also used to
estimate fading/bias signals, with soil moisture as a by-product.

**SAR for archaeology also already exists** (e.g. "Detection of Archaeological
Residues in Vegetated Areas Using Satellite Synthetic Aperture Radar",
Remote Sensing 2017), and **automated tell detection is an established field
with published benchmarks** — Menze & Ur (PNAS 2012, the very catalog used
here as ground truth) and machine-learning mound classification from
multisensor/multitemporal satellite data (PNAS 2020).

**What must change before any publication or public post:**
1. **Delete the "discarded as error" framing entirely.** It is false and it is
   the single most reviewer-baiting sentence in the project.
2. **Soften the novelty claim** to what is actually supportable: *"we found no
   prior application of closure phase specifically to archaeological site
   detection in a non-exhaustive search."* Two web searches are not a
   literature review.
3. **Benchmark against the existing literature.** An AUC of 0.614 must be
   presented next to what published tell-detection methods already achieve. If
   the PNAS 2020 ML approach substantially outperforms it, then this result is
   *a single weak novel channel*, not a competitive detector — and it must be
   framed that way or it will look naive.

**Honest residual contribution** (still worth something, stated narrowly):
applying closure phase as a *ranking channel for archaeological site detection*,
validated against a 14,324-site catalog on two areas with covariate-matched
controls and reported with confidence intervals. That is a modest, careful,
publishable-as-a-note result — not a discovery of a discarded signal.

### 1.5 Multiple comparisons across channels were never accounted for

The program has tested roughly a dozen channels against the same ground truth:
prominence, BSI, texture, thermal, VV kurtosis, VH/VV intercept, anisotropy,
closure, TDA-H1, hollow-ways, crop-marks, plus ML combinations. Finding two or
three at 0.60–0.64 is **not surprising by chance** if the underlying true effect
for most is ~0.55.

The N = 3 kill switch controls *iterations within a method*. Nothing controls the
*family-wise error rate across methods*.

**Impact.** The "measured free-data ceiling of 0.616" and the cluster of 0.60+
channels are partly a selection effect. The channels that survived are the ones
that happened to land high.

**Fix.** Apply a family-wise correction (Bonferroni/Holm) across the channel family,
or — better, since the channels are correlated — report the *joint* result: "of 12
channels tested, 4 exceeded 0.60, with CIs overlapping heavily," which is a much
more honest summary than a leaderboard.

---

## TIER 2 — Issues that weaken specific candidates

### 2.1 Tampa: the inverted source depths contradict the stated mechanism

The three Tampa candidates are described as "sinkhole-type," but the Mogi
inversions return source depths of **478 m, 663 m, and 586 m**. Florida
cover-collapse sinkholes involve raveling of surficial sediment into the shallow
Ocala/Avon Park limestone — typically **tens of metres**, not 500+.

Either the inversion is poorly constrained (most likely — see 2.2) or the signal is
something other than sinkhole mechanics (deeper aquifer compaction). **The brief
reports these depths without flagging the inconsistency.** That is an internal
contradiction a geologist receiving the brief would catch immediately.

**Fix.** Either flag the depths as unreliable with stated uncertainty, or drop them
from the brief and describe only the observable (localized accelerating bowl of
~90 m at ~−6.8 cm/yr). **The brief should be revised before it is sent.**

### 2.2 Mogi inversions on 8–9 pixels are severely underdetermined

Every headline candidate — Tampa ×3, Mojave — rests on a Mogi point-source fit to
a bowl of **8–9 coherent pixels**. Fitting 3–4 parameters (depth, volume rate,
position) to ~9 spatially-correlated observations is close to unconstrained. The
reported r² (0.78–0.87) is not strong evidence at that n, and
`source_depth_range_m` is **null** in every record — no uncertainty was ever
propagated.

**Fix.** Bootstrap or grid-search the depth/volume posterior and report ranges. If
depth is unconstrained (likely), say so rather than quoting a single number.

### 2.3 Ring verdicts rest on registration accuracy comparable to the feature size

CORONA georeferencing accuracy is ~200 m near the Chuera anchor, degrading to
~300–800 m at 25 km. The ring features are **350–400 m across**. For rings 9, 25,
and 28 the registration residual (~350 m) is **the same size as the feature being
looked for** — meaning "the feature is/is not present in 1967" may be comparing the
wrong patch of ground.

**Impact.** Ring 32's verdict is the most secure (closest to the anchor, and the
1967 village is unmistakable and self-consistent). The weak-positive/inconclusive
verdicts on 9/25/28 carry less information than their labels imply.

**Fix.** Per-target 2-GCP local registration before assigning any verdict; treat
current 9/25/28 labels as provisional.

### 2.4 Ring 32's interpretation is unblinded, single-analyst visual reading

The promotion rests on my description of a 1967 image: "courtyard houses on a
raised sub-circular platform, central mound, pond, radiating hollow ways." That is
one analyst, knowing what he was looking for, describing a grainy panchromatic
frame. Confirmation bias risk is high.

The ring-34 kill is genuine evidence the method *can* discriminate — but a kill is
an easier judgement ("nothing there") than a promotion ("this is an ancient ringed
settlement").

**Fix.** Blind chip test: mix ring-32 with N random 1967 village chips and known
tells, relabel, and score them cold. Better: show the pair to an actual Near
Eastern archaeologist. This is cheap and would substantially firm or deflate the
candidate.

### 2.5 "Not in the Menze–Ur catalog" ≠ "unknown to archaeology"

The novelty claim rests on absence from **one** catalog covering a specific survey
area with specific inclusion criteria. The project's own memory already records
that open-database cross-referencing is nearly meaningless in Syria. Moreover, the
1967 image shows a **living village** on the site — it was certainly known to the
people living there and plausibly to Syrian DGAM records.

**Fix.** Soften to "absent from the Menze–Ur catalog" (accurate) rather than any
implication of being undiscovered. Real novelty assessment needs a literature and
DGAM check, which is expert-gated.

### 2.6 The Mojave "sole survivor" status is partly filter-dependent

The filter thresholds (area < 0.10 km², |regional corr| < 0.30, void_likelihood
≥ 0.9) were chosen while looking at the data, and the agriculture screen that
killed the other isolated survivors was **built and tuned during the same session**.
There is mild circularity: the survivor is the point that survived a screen
developed partly by examining what needed killing.

**Fix.** Freeze the filter, then re-run on a held-out region. Or, better, run the
false-negative test in 3.2.

---

## TIER 3 — Structural and process issues

### 3.1 Most results are not reproducible from the repository

The ledger records outcomes, but the code that produced **most** of them lives in
`data/research/scripts/` (149 files) and the session scratchpad — both **gitignored**.
Only two methods (`deformation_intel/context.py` agriculture screening,
`archaeo_intel/closure.py`) exist as version-controlled, tested modules.

**Impact.** This is the most serious *structural* problem in the project. A
program whose entire value proposition is rigor cannot have the majority of its
analyses sitting outside version control. Nobody — including future-you — can
re-run the anisotropy sweep, the TDA sweep, the Cap-1 analysis, or the desert
sweep from the repo.

**Fix.** Promote the load-bearing analyses to tested modules, as was just done for
closure phase. Priority order: the desert-sweep detector chain, the anisotropy
feature, the TDA ring detector, the Cap-1 coherence rule.

### 3.2 No false-negative / recall testing anywhere

Every validation measures **false positives** (does it flag things that turn out to
be agriculture/mountains/weather?). Nothing measures **sensitivity**: if a real
void or a real tell were present, would the pipeline actually keep it?

The desert funnel (6,379 → 20 → 1) is characterized entirely by what it *rejected*.
Its recall is unknown. It is entirely possible the strict filter discards real
voids.

**Fix.** Injection testing: insert synthetic Mogi bowls of known depth/rate into
real OPERA cubes, run the full pipeline, and measure the recovery fraction as a
function of size and rate. This converts "no new voids found" from an ambiguous
result into a quantified detection limit.

### 3.3 All validation is self-administered

Every pre-registration, control, and verdict was written, executed, and judged by
the same agent. Git commit timestamps are decent evidence that registrations
preceded results — genuinely better than nothing — but there is no external
registry, no second analyst, and no outside review of any candidate.

**Fix.** For anything intended for publication, use a real preregistration
(OSF/AsPredicted) and get one external reader.

### 3.4 The ledger's language outruns its statistics in places

Phrases like "clears the bar," "PASS," "validated," and "the new best single
channel" imply a precision the CIs (1.1) do not support. The project has been
admirably honest about *kills*, and about walking back the anisotropy "record"
claim — but the residual pass/fail vocabulary still overstates.

**Fix.** Adopt effect-size-plus-CI language throughout: "0.62 [0.55–0.68], a small
but above-chance effect."

---

## What this critique does **not** undermine

To keep the review calibrated — these hold up:

- **The kills are the strongest part of the project.** Hunt 11 (N = 3, floor
  quantified), the ICA retirement (refused re-scoring a mis-sited control), the
  crop-mark 3/3, the rain-veto NULL, the anisotropy sweep 12/12 explained, and the
  desert sweep's honest negative — this is real discipline, and rarer than results.
- **Every channel is genuinely above chance.** All CIs exclude 0.5. The effects
  are small, not absent.
- **The closure replication with n = 733 is the most solid number here** — the
  tight CI [0.574, 0.632] is well clear of chance.
- **Ring 34's kill and the desert sweep's self-correction** (Carson Sink is not
  agricultural) show the project correcting itself against its own interests.
- **The agriculture-screen build** caught two of its own bugs through escalating
  real-data validation — that is the process working as intended.
- **The redaction and ethics posture** (conflict-zone coordinates local, regulators
  before publicity) is correct and consistently applied.

---

## Recommended priority of fixes

1. **Covariate-matched controls** (1.3) — free, and it is the experiment most
   likely to change what the archaeology results *mean*
2. **Report CIs everywhere; retro-annotate the ledger** (1.1) — free, one pass
3. **Revise the Tampa brief** to fix the depth inconsistency (2.1) — must happen
   *before* it is sent
4. **Dry-season closure run** (1.4) — cheap, and it is the mechanism test
5. **Injection/recall testing** (3.2) — converts null results into stated limits
6. **Promote load-bearing scripts into tested modules** (3.1) — the structural debt
7. **Blind chip test on ring 32** (2.4) — cheap, firms or deflates the top candidate
8. **Different-track closure replication** (1.2) — makes "replicated" fully earned

---

## PUBLICATION / PUBLIC-POST SAFETY CHECKLIST

Added 2026-07-22 in response to "make sure we will not be fools if we publish."
These are the things that would actually cause embarrassment or harm.

### MUST FIX BEFORE ANY PUBLICATION (blocking)

| # | Risk | Status |
|---|---|---|
| P1 | **False novelty claim** — "closure phase is discarded as error" / "no prior use". Closure-phase soil-moisture retrieval is a mature published field; the first reviewer would cite it immediately | **FIXED** (see 1.6) — framing deleted, claim narrowed, non-exhaustive search stated |
| P2 | **No benchmark against prior art.** AUC 0.614 must be shown against published tell-detection performance (Menze & Ur PNAS 2012; ML mound classification PNAS 2020), or it looks naive | **OPEN — blocking.** Must read those papers and report their numbers alongside ours |
| P3 | **Publishing precise coordinates.** Syrian sites = looting risk. Tampa candidates = private property, and "your land is sinking" is defamatory-adjacent if wrong | Syria redacted ✓. **Tampa coords must NOT be published** — they go to agencies only, never in a public post |
| P4 | **Not reproducible.** Most analyses are gitignored scratchpad scripts (3.1). A public claim nobody can re-run is a liability | **OPEN** — promote the load-bearing scripts first |
| P5 | **Unregistered multiple comparisons** (1.5). ~12 channels tested; a reviewer will ask | **OPEN** — report the family, not a leaderboard |

### MUST NOT SAY (retired claims)

- ❌ "closure phase is a signal everyone throws away" — **false**
- ❌ "no prior use of closure phase for archaeology" — unsupported; say *"we found none in a non-exhaustive search"*
- ❌ "clears the 0.60 bar" / "PASS" — statistically meaningless at n≈141 (1.1)
- ❌ "the new best single channel" / "record" — already walked back once; CIs overlap
- ❌ any legacy mineral-prospectivity number (8.5×, 32.4 %, >7σ) — unverified
- ❌ "discovery", "we found a lost city", "confirmed void" — nothing is ground-verified
- ❌ Mogi depths for Tampa — withdrawn as physically inconsistent (2.1)

### SAFE TO SAY (defensible today)

- ✓ "A pre-registered, covariate-matched analysis finds closure-phase magnitude
  separates known tell sites from matched controls at AUC 0.614 [0.585, 0.642],
  n = 733 — a small but reliably above-chance effect."
- ✓ "Ranking-grade, not a detector. Mechanism inferred, not demonstrated."
- ✓ "A 65-tile desert survey produced no void candidates above a measured
  detection limit of ~90 m diameter and ~2–3 cm/yr."
- ✓ "CORONA 1967 imagery discriminates modern from pre-modern earthworks; it
  killed one ring candidate and promoted another."
- ✓ The kill list — this is the most defensible content in the project.
- ✓ The methods/code, with tests, as an open-source contribution.

### THE HONEST ELEVATOR PITCH (if posting publicly)

> An open, free-data pipeline for ranking archaeological candidates and
> detecting active ground deformation, with unusually explicit validation: every
> method pre-registered, confidence intervals on every score, controls matched on
> landscape covariates, a measured detection limit, and a documented list of the
> methods that **failed**. Headline result is deliberately modest — a small,
> replicated, positionally-controlled effect — and the failures are published
> alongside it.

That framing is defensible, useful, and cannot be embarrassed by a reviewer,
because it claims exactly what was measured and nothing more.
