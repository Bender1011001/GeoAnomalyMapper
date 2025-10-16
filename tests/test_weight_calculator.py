import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gam.fusion.weight_calculator import physics_weighting, WeightResult


def test_physics_weighting_balanced_layers():
    layers = {
        "gravity": {
            "model": "gravity_slab",
            "resolution": 1000,
            "density_contrast_kg_m3": 420,
            "target_thickness_m": 180,
            "target_depth_m": 600,
            "noise_floor": 0.08,
        },
        "magnetics": {
            "model": "magnetic_dipole",
            "resolution": 1000,
            "magnetization_a_m": 9.5,
            "anomaly_volume_m3": 2.2e5,
            "target_depth_m": 600,
            "inclination_deg": 63,
            "noise_floor": 1.2,
        },
    }

    result = physics_weighting(layers)
    assert isinstance(result, WeightResult)
    assert pytest.approx(sum(result.weights.values()), rel=1e-9) == 1.0
    assert pytest.approx(result.weights["gravity"], rel=1e-3) == 0.6089533493
    assert pytest.approx(result.weights["magnetics"], rel=1e-3) == 0.3910466507


def test_physics_weighting_requires_supported_model():
    with pytest.raises(ValueError, match="Unsupported fusion model"):
        physics_weighting({"invalid": {"model": "unknown", "resolution": 1000, "noise_floor": 1.0}})


def test_magnetic_model_requires_magnetization():
    with pytest.raises(KeyError):
        physics_weighting({
            "magnetics": {
                "model": "magnetic_dipole",
                "resolution": 1000,
                "noise_floor": 1.0,
            }
        })
