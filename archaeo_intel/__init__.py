"""Archaeological surface-proxy intelligence from free public satellite data.

Validated-detector philosophy (same as deformation_intel): every analytic must
pass synthetic unit tests, and each channel must reproduce its positive control
(Tell Brak for mound detection; Menze-Ur catalog AUC for population claims)
before being pointed at unknown landscapes.

Channels and their measured honest performance (2026-07, Upper Khabur ground
truth = Menze & Ur 2012 catalog, 14,324 sites):
- DEM prominence + VLM triage: finds LARGE mounds; externally validated 4/4
  confident detections matched catalog sites at 41-373 m.
- Multi-temporal BSI composite: amplifies contrast at big tells (2.3x at Tell
  Brak) but population AUC ~0.55 — not a general small-site detector.
- Landsat thermal composite (via Planetary Computer): best single population
  feature (AUC 0.62) yet still weak; 100 m resolution.
- Combined ML ceiling (5-fold CV, 241 sites vs matched controls): AUC 0.616.
- Hollow-way ridge detection (NDMI composites): size-agnostic route/convergence
  signal; see detect.radial_alignment.
"""

__version__ = "0.1.0"
