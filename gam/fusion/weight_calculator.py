"""Utility functions for computing fusion weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class WeightResult:
    weights: Dict[str, float]


def resolution_weighting(resolutions: Dict[str, float], temperature: float = 1.0) -> WeightResult:
    if not resolutions:
        raise ValueError("No resolutions provided")
    inv = np.array([1.0 / float(res) for res in resolutions.values()], dtype=float)
    inv = inv ** (1.0 / max(temperature, 1e-6))
    probs = inv / inv.sum()
    weights = {name: float(weight) for name, weight in zip(resolutions.keys(), probs)}
    return WeightResult(weights)


__all__ = ["resolution_weighting", "WeightResult"]
