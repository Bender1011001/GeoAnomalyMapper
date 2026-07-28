"""interesting_intel — a general "worth a glance" ranker for free satellite
imagery.

Not a phenomenon detector: a FUNNEL that turns large areas of NAIP / Sentinel-2
/ Copernicus-DEM into a short ranked queue of chips a curious human would want
to look at (archaeology, weird geometry, industrial oddities, craters, scars,
anything out of place for its surroundings).

    candidates (priors.py)  ->  CV features (features.py)  ->  cheap VLM on
    contact sheets  ->  strong VLM on full-res chips  ->  human contact sheet

The output is a queue, not a label. Success is measured as "of the top 100,
how many made a human say huh" — see docs/INTERESTING_FINDINGS.md.
"""
from interesting_intel import features, priors  # noqa: F401

__all__ = ["features", "priors"]
