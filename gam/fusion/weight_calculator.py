"""Physics-aware weighting utilities for raster fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
VACUUM_PERMEABILITY = 4 * math.pi * 1e-7  # H m^-1


@dataclass(frozen=True)
class WeightResult:
    """Encapsulate the weight assigned to each layer."""

    weights: Dict[str, float]


def _gravity_slab_response(config: Mapping[str, float]) -> float:
    """Compute expected mGal response for a Bouguer slab anomaly."""

    try:
        density = float(config["density_contrast_kg_m3"])
        thickness = float(config["target_thickness_m"])
    except KeyError as exc:  # pragma: no cover - validated via tests
        raise KeyError(f"Gravity slab model requires '{exc.args[0]}' parameter") from exc

    depth = float(config.get("target_depth_m", thickness))
    resolution = max(float(config.get("resolution", depth)), 1.0)

    base_response = 2 * math.pi * GRAVITATIONAL_CONSTANT * density * thickness
    decay = math.exp(-2 * math.pi * depth / resolution)
    return base_response * decay * 1e5  # convert m/s^2 to mGal


def _magnetic_dipole_response(config: Mapping[str, float]) -> float:
    """Estimate expected nanoTesla response for a buried dipole."""

    try:
        magnetization = float(config["magnetization_a_m"])
    except KeyError as exc:  # pragma: no cover - validated via tests
        raise KeyError(f"Magnetic dipole model requires '{exc.args[0]}' parameter") from exc

    volume = float(config.get("anomaly_volume_m3", 1.0))
    moment = magnetization * volume

    depth = max(float(config.get("target_depth_m", 1.0)), 1.0)
    inclination = math.radians(float(config.get("inclination_deg", 60.0)))

    field = (VACUUM_PERMEABILITY / (4 * math.pi)) * (2 * moment * math.cos(inclination)) / (depth**3)
    return abs(field) * 1e9  # Tesla -> nanoTesla


def _topography_response(config: Mapping[str, float]) -> float:
    """Estimate metre-scale sensitivity for elevation gradients."""

    try:
        relief = float(config["characteristic_relief_m"])
    except KeyError as exc:  # pragma: no cover - validated via tests
        raise KeyError(f"Topography model requires '{exc.args[0]}' parameter") from exc

    resolution = max(float(config.get("resolution", 30.0)), 1.0)
    slope = math.tan(math.radians(float(config.get("max_slope_deg", 35.0))))
    return relief * slope / resolution


_MODEL_REGISTRY = {
    "gravity_slab": _gravity_slab_response,
    "magnetic_dipole": _magnetic_dipole_response,
    "topography_gradient": _topography_response,
}


def physics_weighting(layer_configs: Dict[str, Mapping[str, float]]) -> WeightResult:
    """Compute fusion weights using physics-grounded signal and noise models.

    Each layer configuration must provide a ``model`` key referencing a
    sensitivity function in :data:`_MODEL_REGISTRY`.  The helper evaluates the
    expected signal amplitude and scales it by the declared noise floor and
    spatial resolution to estimate information content.  Weights are normalised
    to sum to one.
    """

    if not layer_configs:
        raise ValueError("At least one layer configuration is required")

    information = {}
    for name, cfg in layer_configs.items():
        model_name = cfg.get("model")
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported fusion model '{model_name}' for layer '{name}'")
        response = _MODEL_REGISTRY[model_name](cfg)
        noise = max(float(cfg.get("noise_floor", 1.0)), 1e-12)
        resolution = max(float(cfg.get("resolution", 1.0)), 1.0)
        information[name] = (response / noise) ** 2 / resolution

    total = float(sum(information.values()))
    if total <= 0:
        raise ValueError("Computed non-positive information content for all layers")

    weights = {name: value / total for name, value in information.items()}
    return WeightResult(weights)


__all__ = ["WeightResult", "physics_weighting"]
