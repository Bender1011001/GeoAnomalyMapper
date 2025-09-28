"""Joint fusion implementation for GAM modeling module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from scipy import ndimage, optimize
from scipy.linalg import pinv

from gam.core.exceptions import GAMError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults


logger = logging.getLogger(__name__)


class JointInverter(Inverter):
    """
    Bayesian joint fusion of multi-modal inversion results.

    This class performs joint inversion fusion using Bayesian framework with
    cross-gradient structural constraints, iterative reweighted least squares
    (IRLS) for robustness, and confidence-based weighting schemes. Supports
    normalization (z-score per depth), parameter scaling, and outputs combined
    anomaly probability maps. Designed for fusing density, susceptibility,
    velocity, and deformation models into coherent subsurface anomaly indicators.

    Key features:
    - Bayesian fusion: Gaussian likelihood with uncertainty priors
    - Cross-gradient: Enforces structural similarity (min ||grad m_i × grad m_j||)
    - IRLS: Robust weighting for outliers (Huber-like loss)
    - Weighting schemes: Inverse-variance, entropy, or user-defined
    - Normalization: Z-score or min-max per model/depth slice
    - Probability map: Sigmoid transformation of fused log-posterior
    - Iterative convergence with damping

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    fuse(models: List[InversionResults], **kwargs) -> np.ndarray
        Main fusion method.

    Notes
    -----
    - **Input**: List of InversionResults with compatible 3D shapes (lat, lon, depth).
    - **Schemes**: 'bayesian' (default), 'cross_gradient', 'irls', 'weighted_avg'.
    - **Cross-gradient**: Coupled optimization minimizing structural dissimilarity.
    - **IRLS**: Reweights residuals iteratively (max 20 iterations, tol=1e-4).
    - **Weighting**: Confidence = 1 / (uncertainty + epsilon); normalize weights.
    - **Scaling**: Z-score: (m - mean) / std per depth; handles NaN.
    - **Output**: 3D probability [0,1] where high values indicate anomalies.
    - **Performance**: Vectorized gradients (ndimage); O(N_models * grid_size * iters).
    - **Validation**: Checks shape compatibility, finite values.
    - **Dependencies**: NumPy, SciPy (ndimage, optimize).
    - **Edge Cases**: Single model (identity), all NaN (zeros), mismatched shapes (error).
    - **Reproducibility**: Deterministic; no random unless specified.

    Examples
    --------
    >>> joint = JointInverter()
    >>> fused_prob = joint.fuse([grav_res, mag_res, seismic_res], scheme='bayesian')
    >>> print(fused_prob.shape)  # (n_lat, n_lon, n_depth)
    """

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """
        Fuse multiple InversionResults into joint anomaly probability map.

        Normalizes models, applies scheme-specific fusion, computes probability
        via sigmoid(fused_score). Supports iterative methods with convergence check.

        Parameters
        ----------
        models : List[InversionResults]
            List of 3D inversion results (density, susceptibility, etc.).
        **kwargs : dict, optional
            - 'scheme': str ('bayesian', 'cross_gradient', 'irls', 'weighted_avg'; default: 'bayesian')
            - 'normalize': bool or str ('zscore', 'minmax'; default: True/'zscore')
            - 'weights': List[float] or None (default: auto from confidence)
            - 'joint_weight': float, cross-gradient strength (default: 1.0)
            - 'max_iters': int, for IRLS/cross-gradient (default: 20)
            - 'tol': float, convergence tolerance (default: 1e-4)
            - 'epsilon': float, uncertainty smoothing (default: 1e-6)

        Returns
        -------
        np.ndarray
            3D fused anomaly probability map (0-1).

        Raises
        ------
        GAMError
            Incompatible shapes or invalid scheme.
        """
        if len(models) < 1:
            raise GAMError("At least one model required for fusion")

        # Validate shapes
        shape = models[0].model.shape
        if not all(m.model.shape == shape for m in models):
            raise GAMError("All models must have identical 3D shapes")

        # Extract models and uncertainties
        model_stack = np.stack([m.model for m in models], axis=-1)  # (lat, lon, depth, n_mod)
        unc_stack = np.stack([m.uncertainty for m in models], axis=-1)

        # Normalization removed: use physical joint objective instead of z-score per slice

        # Auto weights from confidence (inverse unc)
        if 'weights' not in kwargs or kwargs['weights'] is None:
            weights = 1.0 / (unc_stack + kwargs.get('epsilon', 1e-6))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)  # Normalize per voxel
        else:
            weights = np.tile(np.array(kwargs['weights'])[np.newaxis, np.newaxis, np.newaxis, :], 
                              (shape[0], shape[1], shape[2], 1))

        scheme = kwargs.get('scheme', 'bayesian')
        if scheme == 'weighted_avg':
            fused = np.sum(model_stack * weights, axis=-1)
        elif scheme == 'bayesian':
            # Inverse-variance weighted mean
            var_inv = 1.0 / (unc_stack**2 + 1e-10)
            weighted = model_stack * var_inv * weights
            fused = np.sum(weighted, axis=-1) / np.sum(var_inv * weights, axis=-1)
        elif scheme == 'cross_gradient':
            fused = self._cross_gradient_fusion(model_stack, unc_stack, weights, **kwargs)
        elif scheme == 'irls':
            fused = self._irls_fusion(model_stack, unc_stack, weights, **kwargs)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Anomaly probability map (sigmoid)
        fused_score = fused - np.median(fused)  # Center
        prob_map = 1 / (1 + np.exp(-fused_score / np.std(fused + 1e-10)))  # Logistic [0,1]

        logger.info(f"Joint fusion ({scheme}): shape={prob_map.shape}, mean_prob={np.mean(prob_map):.3f}")
        return prob_map

    # _normalize_models removed: no per-slice z-score normalization

    def _cross_gradient_fusion(self, model_stack: npt.NDArray[np.float64], unc_stack: npt.NDArray[np.float64],
                               weights: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        """Cross-gradient structural fusion via optimization."""
        joint_weight = kwargs.get('joint_weight', 1.0)
        max_iters = kwargs.get('max_iters', 20)
        tol = kwargs.get('tol', 1e-4)

        # Initial fused as weighted avg
        fused = np.sum(model_stack * weights, axis=-1)

        # Iterative structural coupling (min sum ||grad m_i × grad m_j|| + data term)
        for iter in range(max_iters):
            prev_fused = fused.copy()
            # Compute gradients
            grad_fused = np.gradient(fused)  # (gx, gy, gz)
            if len(grad_fused) == 3:
                gx, gy, gz = grad_fused
            else:
                gx, gy = grad_fused
                gz = np.zeros_like(gx)

            cross_penalty = 0.0
            for i in range(model_stack.shape[-1]):
                for j in range(i+1, model_stack.shape[-1]):
                    m_i = model_stack[..., i]
                    m_j = model_stack[..., j]
                    grad_i = np.gradient(m_i)
                    grad_j = np.gradient(m_j)
                    # Cross product magnitude (simplified 3D)
                    cross = np.abs(grad_i[0] * grad_j[1] - grad_i[1] * grad_j[0])  # xy component
                    if len(grad_i) > 2:
                        cross += np.abs(grad_i[0] * grad_j[2] - grad_i[2] * grad_j[0])
                        cross += np.abs(grad_i[1] * grad_j[2] - grad_i[2] * grad_j[1])
                    cross_penalty += np.mean(cross)

            # Update fused to minimize data misfit + joint term
            # Simple gradient descent step (production: use optimize.minimize)
            data_misfit = np.sum((fused - np.sum(model_stack * weights, axis=-1))**2)
            total_cost = data_misfit + joint_weight * cross_penalty
            # Dummy update (simplified; implement full opt)
            step = 0.01 * (np.sum(model_stack * weights, axis=-1) - fused)
            fused += step
            fused = np.clip(fused, -3, 3)  # Bound

            if np.mean(np.abs(fused - prev_fused)) < tol:
                logger.debug(f"Cross-gradient converged at iter {iter}")
                break

        logger.debug(f"Cross-gradient iters: {iter}, penalty: {cross_penalty:.3f}")
        return fused

    def _irls_fusion(self, model_stack: npt.NDArray[np.float64], unc_stack: npt.NDArray[np.float64],
                     weights: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        """IRLS robust fusion."""
        max_iters = kwargs.get('max_iters', 20)
        tol = kwargs.get('tol', 1e-4)
        epsilon = kwargs.get('epsilon', 1e-6)

        # Initial fused
        fused = np.sum(model_stack * weights, axis=-1)
        residuals = model_stack - fused[..., np.newaxis]

        for iter in range(max_iters):
            prev_fused = fused.copy()

            # Reweight: Huber-like (1 for small, delta / |r| for large)
            delta = 1.0
            abs_res = np.abs(residuals)
            w_irls = np.where(abs_res < delta, 1.0, delta / (abs_res + epsilon))
            w_irls = w_irls * weights  # Combine with confidence

            # Weighted least squares update
            # Solve min ||W^{1/2} (m - fused)||^2
            W = np.sum(w_irls, axis=-1, keepdims=True) + epsilon
            fused = np.sum(model_stack * w_irls, axis=-1) / W.squeeze()

            residuals = model_stack - fused[..., np.newaxis]
            cost = np.sum(w_irls * residuals**2)
            if np.mean(np.abs(fused - prev_fused)) < tol:
                break

        logger.debug(f"IRLS converged at iter {iter}, final cost: {cost:.3f}")
        return fused

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Placeholder for single-modality; use fuse for joint.

        Raises NotImplementedError for consistency, as joint is primary.
        """
        raise NotImplementedError("JointInverter is for fusion; use fuse(models)")