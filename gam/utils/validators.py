"""Validation helpers for ensuring pipeline consistency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def assert_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> None:
    """Assert that two arrays are within ``tol`` relative difference."""
    if a.shape != b.shape:
        raise AssertionError(f"Array shapes differ: {a.shape} vs {b.shape}")
    denom = np.maximum(np.abs(a), np.abs(b))
    denom[denom == 0] = 1.0
    rel = np.abs(a - b) / denom
    if np.nanmax(rel) > tol:
        raise AssertionError(f"Arrays differ by more than {tol}")


@dataclass
class AreaCheckResult:
    zone: str
    polygon_index: int
    area_before: float
    area_after: float

    @property
    def relative_error(self) -> float:
        return abs(self.area_after - self.area_before) / self.area_before


def ensure_area_preservation(results: Sequence[AreaCheckResult], tolerance: float = 0.01) -> None:
    """Validate that reprojection area checks satisfy ``tolerance``."""
    failures = [r for r in results if r.relative_error > tolerance]
    if failures:
        details = "; ".join(
            f"zone {f.zone} polygon {f.polygon_index}: err={f.relative_error:.4f}"
            for f in failures
        )
        raise AssertionError(f"Area preservation failed: {details}")


__all__ = ["assert_close", "AreaCheckResult", "ensure_area_preservation"]
