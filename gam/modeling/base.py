"""Abstract base classes for the GAM modeling/inversion module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from gam.preprocessing.data_structures import ProcessedGrid


class Inverter(ABC):
    """
    Abstract base class for geophysical inversion algorithms in GAM.

    This interface defines the contract for all modality-specific inverters.
    Implementations must provide concrete inversion logic using appropriate
    geophysical modeling frameworks (e.g., SimPEG, PyGIMLi). The class supports
    both individual modality inversions and joint fusion of multiple models.

    All implementations must handle numerical stability, convergence checking,
    and return standardized InversionResults. The fuse method enables structural
    coupling across modalities using cross-gradient constraints or Bayesian
    weighting.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    invert(data: ProcessedGrid, **kwargs) -> InversionResults
        Perform inversion for the specific modality.
    fuse(models: List[InversionResults], **kwargs) -> np.ndarray
        Fuse multiple inversion results into a joint model.

    Notes
    -----
    - **kwargs** in methods allow flexible parameter passing (e.g., regularization
      strength, mesh resolution, random seed for reproducibility).
    - Implementations should use logging for progress and errors.
    - Ensure thread-safety for parallel execution via Dask.
    - Handle edge cases: sparse data, non-convergent inversions (raise
      InversionConvergenceError), invalid inputs (raise PreprocessingError).
    - Reproducibility: Set random seeds via kwargs['random_seed'] if provided.
    - Performance: Optimize for large grids using vectorized operations and
      sparse matrices where applicable.
    - Integration: Compatible with ModelingManager for orchestration.

    Examples
    --------
    >>> inverter = GravityInverter()
    >>> results = inverter.invert(processed_grid, regularization=1e-6)
    >>> fused = inverter.fuse([results_grav, results_mag], weight_scheme='bayesian')
    """

    @abstractmethod
    def invert(self, data: ProcessedGrid, **kwargs) -> "InversionResults":
        """
        Perform modality-specific inversion on processed grid data.

        This method sets up the forward modeling problem, optimizes the
        inverse problem (e.g., least-squares with regularization), and
        returns the estimated subsurface model with uncertainties.

        Parameters
        ----------
        data : ProcessedGrid
            Input gridded data from preprocessing module, containing
            observed geophysical measurements (e.g., gravity anomalies).
        **kwargs : dict, optional
            Inversion parameters, including:
            - 'mesh': Custom mesh (if not auto-generated)
            - 'regularization': L1/L2 weights (default: {'alpha_s': 1e-4})
            - 'max_iterations': Maximum solver iterations (default: 20)
            - 'random_seed': For reproducible results (default: None)
            - Modality-specific params (e.g., 'inclination' for magnetics)

        Returns
        -------
        InversionResults
            Dict-like structure with:
            - 'model': np.ndarray, 3D subsurface model (e.g., density contrasts)
            - 'uncertainty': np.ndarray, 3D uncertainty estimates (standard deviation)
            - 'metadata': dict, including convergence info, iterations, residuals

        Raises
        ------
        PreprocessingError
            If input data is invalid or incompatible with modality.
        InversionConvergenceError
            If the inversion solver does not converge within max_iterations.
        ValueError
            If required kwargs are missing or invalid.

        Notes
        -----
        - The model and uncertainty arrays share the same 3D shape (lat, lon, depth).
        - Units: Model in physical units (e.g., kg/m³ for density); uncertainty in same units.
        - Convergence: Check residuals < tolerance (e.g., 1e-3); log warnings if marginal.
        - Memory: For large grids, use sparse representations if supported by framework.
        - Validation: Ensure output arrays are finite and non-NaN where possible.

        Examples
        --------
        >>> results = inverter.invert(data, max_iterations=50, random_seed=42)
        >>> print(results.model.shape)  # e.g., (100, 100, 50)
        """
        pass

    @abstractmethod
    def fuse(self, models: List["InversionResults"], **kwargs) -> np.ndarray:
        """
        Fuse multiple InversionResults into a joint subsurface model.

        This method combines models from different modalities using
        structural constraints (e.g., cross-gradients) or statistical
        methods (e.g., Bayesian averaging). The output is a single 3D
        array representing the fused property (e.g., combined anomaly strength).

        Parameters
        ----------
        models : List[InversionResults]
            List of inversion results from different modalities (e.g., gravity, magnetics).
            All models must share compatible spatial dimensions.
        **kwargs : dict, optional
            Fusion parameters, including:
            - 'method': Fusion algorithm ('cross_gradient', 'bayesian', 'weighted_avg')
            - 'weights': Modality weights (default: equal weighting)
            - 'joint_weight': Structural coupling strength (default: 1.0)
            - 'normalize': Whether to scale models (default: True)
            - 'random_seed': For stochastic fusion methods (default: None)

        Returns
        -------
        np.ndarray
            3D fused model array (lat, lon, depth) with combined information.
            Values represent integrated anomaly indicators (e.g., probability or strength).

        Raises
        ------
        ValueError
            If models have incompatible shapes or missing required fields.
        NotImplementedError
            If fusion method is unsupported by the inverter.

        Notes
        -----
        - Output shape matches the common grid of input models (broadcast if needed).
        - Normalization: Scale models to [0,1] or unit variance before fusion.
        - Cross-gradient: Enforces structural similarity while preserving amplitude differences.
        - Bayesian: Incorporates uncertainties as inverse variances in weighting.
        - Performance: Use iterative reweighted least squares for efficiency.
        - Validation: Check for numerical stability (e.g., avoid division by zero).

        Examples
        --------
        >>> fused = inverter.fuse([grav_results, mag_results], method='bayesian', weights=[0.6, 0.4])
        >>> print(fused.shape)  # Matches input shapes
        """
        pass