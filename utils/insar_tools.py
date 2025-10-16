"""Utility functions for InSAR preprocessing workflows."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def apply_gacos_correction(
    interferogram_path: Path,
    gacos_grid_path: Path,
    output_path: Path,
    executable: str = 'gacos',
    extra_args: Optional[Sequence[str]] = None
) -> Path:
    """Apply GACOS atmospheric correction via the command-line tool."""

    interferogram_path = Path(interferogram_path)
    gacos_grid_path = Path(gacos_grid_path)
    output_path = Path(output_path)

    if shutil.which(executable) is None:
        raise FileNotFoundError(f"GACOS executable not found: {executable}")

    if not interferogram_path.exists():
        raise FileNotFoundError(f"Interferogram not found: {interferogram_path}")

    if not gacos_grid_path.exists():
        raise FileNotFoundError(f"GACOS grid not found: {gacos_grid_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [executable, str(interferogram_path), str(gacos_grid_path), str(output_path)]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)

    logger.debug("Running GACOS command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def apply_coherence_mask(
    interferogram: np.ndarray,
    coherence_map: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Mask low-coherence pixels by setting them to NaN."""

    if interferogram.shape != coherence_map.shape:
        raise ValueError("Interferogram and coherence map must have the same shape")

    masked = interferogram.astype(np.float32, copy=True)
    mask = coherence_map < float(threshold)
    masked[mask] = np.nan
    return masked


def project_los_to_vertical(
    los_displacement: np.ndarray,
    incidence_angle_map: np.ndarray
) -> np.ndarray:
    """Project LOS displacement to the vertical component."""

    if los_displacement.shape != incidence_angle_map.shape:
        raise ValueError("LOS displacement and incidence angle map must have the same shape")

    incidence_rad = np.deg2rad(incidence_angle_map)
    with np.errstate(divide='ignore', invalid='ignore'):
        vertical = los_displacement / np.cos(incidence_rad)
    vertical[np.isnan(los_displacement)] = np.nan
    return vertical.astype(np.float32)

