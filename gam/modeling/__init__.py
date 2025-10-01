"""GAM Modeling/Inversion Module.

This module provides the core functionality for geophysical inversions and
multi-modal fusion in GeoAnomalyMapper. It includes abstract interfaces for
inverters, data structures for results, modality-specific implementations
(gravity, magnetic, seismic, InSAR), joint fusion algorithms, anomaly detection,
mesh generation utilities, and the central ModelingManager for workflow
orchestration.

Key components:
- Inverter ABC: Base for all inversion algorithms
- InversionResults: Standardized output structure
- AnomalyOutput: DataFrame for detected anomalies
- Modality inverters: GravityInverter, MagneticInverter, etc.
- JointInverter: Bayesian/cross-gradient fusion
- AnomalyDetector: Statistical/LOF detection
- MeshGenerator: Adaptive/regular meshes
- ModelingManager: Pipeline coordinator

Usage:
from gam.modeling import ModelingManager, InversionResults, AnomalyOutput
manager = ModelingManager()
results = manager.run_inversion(data, 'gravity')
fused = manager.fuse_models([results])
anomalies = manager.detect_anomalies(fused)

Configuration: modeling section in config.yaml for params like mesh_resolution,
regularization, fusion_scheme.

Dependencies: SimPEG, PyGIMLi, NumPy, SciPy, scikit-learn, Pandas, PyYAML, Pydantic.

Logging: Configured at module level; uses logging.getLogger(__name__).
"""

import logging

from .anomaly_detection import AnomalyDetector
from .base import Inverter
from .data_structures import AnomalyOutput, InversionResults
from .fusion import JointInverter
# Import heavy engines lazily/optionally to avoid hard failures if optional deps are missing
try:
    from gam.engines.gravity_simpeg import GravityInverter  # requires simpeg, discretize
except Exception as _e:
    GravityInverter = None  # type: ignore
    logging.getLogger(__name__).warning(f"GravityInverter unavailable: {type(_e).__name__}: {_e}")
try:
    from gam.engines.insar_mogi_okada import InSARInverter  # optional PyCoulomb at runtime
except Exception as _e:
    InSARInverter = None  # type: ignore
    logging.getLogger(__name__).warning(f"InSARInverter unavailable: {type(_e).__name__}: {_e}")
try:
    from gam.engines.magnetics_simpeg import MagneticInverter  # requires simpeg, discretize
except Exception as _e:
    MagneticInverter = None  # type: ignore
    logging.getLogger(__name__).warning(f"MagneticInverter unavailable: {type(_e).__name__}: {_e}")
from .manager import ModelingManager
from .mesh import MeshGenerator
try:
    from gam.engines.seismic_pygimli import SeismicInverter  # requires pygimli
except Exception as _e:
    SeismicInverter = None  # type: ignore
    logging.getLogger(__name__).warning(f"SeismicInverter unavailable: {type(_e).__name__}: {_e}")

# Set up module logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

__all__ = [
    'AnomalyDetector',
    'AnomalyOutput',
    'GravityInverter',
    'InSARInverter',
    'InversionResults',
    'Inverter',
    'JointInverter',
    'MagneticInverter',
    'MeshGenerator',
    'ModelingManager',
    'SeismicInverter',
]