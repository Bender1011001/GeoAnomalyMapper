"""Main modeling manager for GAM inversion workflow coordination."""

from __future__ import annotations

import logging
import yaml
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from gam.core.exceptions import GAMError
from gam.modeling.anomaly_detection import AnomalyDetector
from gam.modeling.base import Inverter
from gam.modeling.data_structures import AnomalyOutput, InversionResults
from gam.modeling.fusion import JointInverter
from gam.engines.gravity_simpeg import GravityInverter
from gam.engines.insar_mogi_okada import InSARInverter
from gam.engines.magnetics_simpeg import MagneticInverter
from gam.modeling.mesh import MeshGenerator
try:
    from gam.engines.seismic_pygimli import SeismicInverter
except Exception as _e:
    SeismicInverter = None  # type: ignore
    logging.getLogger(__name__).warning(f"SeismicInverter unavailable: {type(_e).__name__}: {_e}")
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


class ModelingConfig(BaseModel):
    """Pydantic model for modeling configuration validation."""
    mesh_resolution: float = Field(0.01, description="Grid resolution in degrees")
    regularization: Dict[str, float] = Field(default_factory=dict, description="Reg params e.g., {'alpha_s': 1e-4}")
    max_iterations: int = Field(20, ge=1, description="Max inversion iterations")
    random_seed: Optional[int] = Field(None, description="For reproducibility")
    fusion_scheme: str = Field("bayesian", description="Fusion method")
    anomaly_threshold: float = Field(2.5, description="Detection threshold")
    modalities: List[str] = Field(default_factory=list, description="Available modalities")

    @validator('regularization')
    def validate_reg(cls, v):
        if not v:
            v = {'alpha_s': 1e-4, 'alpha_x': 1.0, 'alpha_y': 1.0, 'alpha_z': 1.0}
        return v

    class Config:
        extra = 'forbid'


class ModelingManager:
    """
    Central coordinator for GAM modeling/inversion workflows.

    This class orchestrates the full modeling pipeline: selects modality-specific
    inverters, runs inversions with mesh generation, fuses results, detects anomalies,
    and supports iterative refinement. Integrates configuration from YAML, provides
    progress reporting via logging, and handles errors gracefully (e.g., skip failed
    modalities). Designed for extensibility via inverter registry.

    Key features:
    - Config-driven: Loads/validates modeling params from config.yaml
    - Modality selection: Factory for GravityInverter, MagneticInverter, etc.
    - End-to-end: run_inversion -> fuse_models -> detect_anomalies
    - Iterative: Refine with updated priors (e.g., from previous fusion)
    - Progress: Logging at INFO level for steps
    - Error handling: Continues on single failure, logs warnings
    - Reproducibility: Seeds from config

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml (default: 'config.yaml').

    Attributes
    ----------
    config : ModelingConfig
        Validated configuration.
    mesh_gen : MeshGenerator
        Mesh utility.
    inverter_registry : Dict[str, Inverter]
        Mapping modality -> inverter instance.
    joint_fuser : JointInverter
        For multi-modal fusion.
    anomaly_detector : AnomalyDetector
        For post-fusion detection.

    Methods
    -------
    load_config(path: str) -> None
        Load and validate config.
    run_inversion(data: ProcessedGrid, modality: str, **kwargs) -> InversionResults
        Run single modality inversion.
    fuse_models(results: List[InversionResults], **kwargs) -> np.ndarray
        Fuse multiple results.
    detect_anomalies(fused: np.ndarray, **kwargs) -> AnomalyOutput
        Detect anomalies in fused model.
    full_pipeline(data: ProcessedGrid, modalities: List[str], **kwargs) -> Tuple[np.ndarray, AnomalyOutput]
        Complete workflow.
    iterative_refine(data: ProcessedGrid, modalities: List[str], n_iters: int = 2, **kwargs) -> AnomalyOutput
        Iterative refinement loop.

    Notes
    -----
    - **Config**: modeling section in YAML; defaults provided.
    - **Registry**: Lazy instantiation; extend by adding to dict.
    - **Mesh**: Auto-generated per inversion; shared if possible.
    - **Progress**: Logs "Running gravity inversion...", etc.
    - **Errors**: If modality unknown, skip and log WARNING.
    - **Performance**: Sequential by default; parallelize via Dask in future.
    - **Validation**: Ensures data compatibility, config types.
    - **Dependencies**: All submodules; pydantic for config.
    - **Edge Cases**: Empty modalities (empty results), no data (error), single modality (no fuse).

    Examples
    --------
    >>> manager = ModelingManager('config.yaml')
    >>> grav_res = manager.run_inversion(data, 'gravity')
    >>> fused = manager.fuse_models([grav_res, mag_res])
    >>> anomalies = manager.detect_anomalies(fused, threshold=2.0)
    >>> full_anoms = manager.full_pipeline(data, ['gravity', 'magnetic'])
    """

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self.load_config(config_path)
        self.mesh_gen = MeshGenerator(crs='EPSG:4326')
        self.joint_fuser = JointInverter()
        self.anomaly_detector = AnomalyDetector()
        # Build registry only with available engines
        self.inverter_registry = {}
        if GravityInverter is not None:
            self.inverter_registry['gravity'] = GravityInverter()
        if MagneticInverter is not None:
            self.inverter_registry['magnetic'] = MagneticInverter()
        if SeismicInverter is not None:
            self.inverter_registry['seismic'] = SeismicInverter()
        if InSARInverter is not None:
            self.inverter_registry['insar'] = InSARInverter()
        # Add more as implemented

    def load_config(self, path: str) -> ModelingConfig:
        """Load and validate modeling config from YAML."""
        try:
            with open(path, 'r') as f:
                full_config = yaml.safe_load(f)
            modeling_section = full_config.get('modeling', {})
            config = ModelingConfig(**modeling_section)
            logger.info(f"Loaded modeling config from {path}")
            return config
        except Exception as e:
            logger.warning(f"Config load failed ({e}); using defaults")
            return ModelingConfig()

    def run_inversion(self, data: ProcessedGrid, modality: str, **kwargs) -> InversionResults:
        """
        Run inversion for specified modality.

        Selects inverter, generates mesh, runs invert with config/kwargs.

        Parameters
        ----------
        data : ProcessedGrid
            Input processed data.
        modality : str
            'gravity', 'magnetic', etc.
        **kwargs : dict
            Overridden by config; e.g., 'regularization'.

        Returns
        -------
        InversionResults
            Inversion output.

        Raises
        ------
        GAMError
            Unknown modality or failure.
        """
        if modality not in self.inverter_registry:
            raise GAMError(f"Unknown modality: {modality}")

        logger.info(f"Running {modality} inversion...")
        inverter = self.inverter_registry[modality]

        # Mesh (per modality if needed)
        mesh_type = kwargs.get('mesh_type', 'adaptive' if modality in ['gravity', 'magnetic'] else 'seismic')
        mesh = self.mesh_gen.create_mesh(data, type=mesh_type, **self.config.regularization)

        # Params from config + kwargs
        params = self.config.dict()
        params.update(kwargs)
        params['mesh'] = mesh
        params['random_seed'] = self.config.random_seed

        try:
            results = inverter.invert(data, **params)
            logger.info(f"{modality} inversion completed successfully")
            return results
        except Exception as e:
            logger.error(f"{modality} inversion failed: {e}")
            raise GAMError(f"Inversion failed for {modality}: {e}")

    def fuse_models(self, results: List[InversionResults], **kwargs) -> np.ndarray:
        """
        Fuse list of results using JointInverter.

        Parameters
        ----------
        results : List[InversionResults]
            Modality results.
        **kwargs : dict
            Fusion params (scheme, weights).

        Returns
        -------
        np.ndarray
            Fused 3D model.
        """
        if len(results) < 2:
            logger.warning("Single result; returning as-is")
            return results[0].model

        logger.info(f"Fusing {len(results)} models...")
        params = self.config.dict()
        params.update(kwargs)
        fused = self.joint_fuser.fuse(results, **params)
        logger.info("Fusion completed")
        return fused

    def detect_anomalies(self, fused: np.ndarray, **kwargs) -> AnomalyOutput:
        """
        Detect anomalies in fused model.

        Parameters
        ----------
        fused : np.ndarray
            3D fused model.
        **kwargs : dict
            Detection params (threshold, method).

        Returns
        -------
        AnomalyOutput
            Detected anomalies.
        """
        logger.info("Detecting anomalies...")
        params = {'threshold': self.config.anomaly_threshold}
        params.update(kwargs)
        anomalies = self.anomaly_detector.detect(fused, **params)
        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def full_pipeline(self, data: ProcessedGrid, modalities: List[str], **kwargs) -> Tuple[np.ndarray, AnomalyOutput]:
        """
        Complete modeling pipeline.

        Runs inversions, fuses, detects.

        Parameters
        ----------
        data : ProcessedGrid
            Input data.
        modalities : List[str]
            Modalities to process.
        **kwargs : dict
            Passed to sub-methods.

        Returns
        -------
        Tuple[np.ndarray, AnomalyOutput]
            Fused model and anomalies.
        """
        results = []
        for mod in modalities:
            try:
                res = self.run_inversion(data, mod, **kwargs)
                results.append(res)
            except GAMError as e:
                logger.warning(f"Skipping {mod}: {e}")
                continue

        if not results:
            raise GAMError("No successful inversions")

        fused = self.fuse_models(results, **kwargs)
        anomalies = self.detect_anomalies(fused, **kwargs)
        return fused, anomalies

    def iterative_refine(self, data: ProcessedGrid, modalities: List[str], n_iters: int = 2, **kwargs) -> AnomalyOutput:
        """
        Iterative refinement: fuse -> use as prior -> re-invert.

        Parameters
        ----------
        data : ProcessedGrid
            Input.
        modalities : List[str]
            Modalities.
        n_iters : int
            Refinement iterations.
        **kwargs : dict
            Pipeline params.

        Returns
        -------
        AnomalyOutput
            Final anomalies.
        """
        current_prior = None
        for i in range(n_iters):
            logger.info(f"Iteration {i+1}/{n_iters}")
            fused, _ = self.full_pipeline(data, modalities, **kwargs)
            if current_prior is None:
                current_prior = fused
            else:
                current_prior = 0.7 * current_prior + 0.3 * fused  # Update prior
            kwargs['reference_model'] = current_prior  # For next inversion

        _, final_anoms = self.full_pipeline(data, modalities, **kwargs)
        return final_anoms