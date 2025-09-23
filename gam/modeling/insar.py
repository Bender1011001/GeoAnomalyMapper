"""InSAR inversion implementation for GAM using elastic half-space models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.mesh import MeshGenerator  # Assumed available
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


def mogi_forward(params: np.ndarray, obs_los: np.ndarray, inc: float, head: float, 
                 poisson: float = 0.25) -> np.ndarray:
    """
    Mogi point source forward model for LOS displacement.

    Parameters
    ----------
    params : np.ndarray
        [x, y, depth, dv] - location (m), depth (m), volume change (m³)
    obs_los : np.ndarray
        Observation points [N, 3] (x, y, los_unit_vector)
    inc : float
        Incidence angle (rad)
    head : float
        Heading angle (rad)
    poisson : float
        Poisson ratio

    Returns
    -------
    np.ndarray
        Predicted LOS displacements (m)
    """
    x0, y0, d, dv = params
    r = np.sqrt((obs_los[:, 0] - x0)**2 + (obs_los[:, 1] - y0)**2 + d**2)
    disp_u = dv * (x0 - obs_los[:, 0]) * d / (4 * np.pi * r**3)
    disp_v = dv * (y0 - obs_los[:, 1]) * d / (4 * np.pi * r**3)
    disp_w = dv * r**2 - 3 * d**2 / (4 * np.pi * r**3)
    
    # LOS unit vector
    los = np.array([np.sin(inc) * np.cos(head), np.sin(inc) * np.sin(head), np.cos(inc)])
    los_disp = disp_u * los[0] + disp_v * los[1] + disp_w * los[2]
    return los_disp


def okada_forward(params: np.ndarray, obs_pos: np.ndarray, inc: float, head: float, 
                  poisson: float = 0.25) -> np.ndarray:
    """
    Okada rectangular dislocation forward model for LOS displacement.

    Simplified Okada (1985) for strike-slip; extend for dip-slip/tensile.

    Parameters
    ----------
    params : np.ndarray
        [x0, y0, depth, L, W, strike, dip, slip] - center x,y (m), depth (m),
        length L (m), width W (m), strike/dip (deg), slip (m)
    obs_pos : np.ndarray
        [N, 2] observation x,y
    inc, head : float
        Radar geometry (rad)
    poisson : float

    Returns
    -------
    np.ndarray
        LOS displacements (m)
    """
    # Simplified implementation; full Okada requires integration over fault
    # For demo, approximate as point source equivalent
    # In production, use full numerical integration or Pyrocko
    x0, y0, d, L, W, strike, dip, slip = params
    # Approximate volume change dv = L * W * slip
    dv = L * W * slip
    # Use Mogi as proxy for point-like
    obs_3d = np.column_stack([obs_pos, np.zeros(len(obs_pos))])
    return mogi_forward(np.array([x0, y0, d, dv]), obs_3d, inc, head, poisson)


class InSARInverter(Inverter):
    """
    InSAR deformation source inversion using Mogi/Okada elastic models.

    This class inverts line-of-sight (LOS) displacements for point (Mogi) or
    rectangular (Okada) sources in elastic half-space. Supports grid search
    initialization followed by least-squares refinement. Handles atmospheric
    corrections via linear ramp fit, noise estimation from data std. Converts
    source params to volume change (m³) and pressure change (MPa) estimates
    assuming crustal modulus.

    Key features:
    - Point source (Mogi): Spherical inflation/deflation
    - Rectangular dislocation (Okada): Fault slip or sill
    - Non-linear least-squares inversion (scipy.optimize)
    - Atmospheric phase screen correction (planar fit)
    - Uncertainty via Jacobian (Hessian approx)
    - Units: Input LOS in mm, output volume in m³, pressure in MPa

    Parameters
    ----------
    poisson : float, optional
        Poisson ratio (default: 0.25).
    modulus : float, optional
        Young's modulus (GPa, default: 80 for crust).

    Attributes
    ----------
    poisson : float
        Elastic parameter.
    modulus : float
        For pressure calculation.

    Methods
    -------
    Inherited from Inverter.

    Notes
    -----
    - **Model**: Analytical elastic half-space; no topography.
    - **Inversion**: Grid search for init, then Levenberg-Marquardt.
    - **Atm Correction**: Subtracts best-fit plane (ramp + offset).
    - **Uncertainty**: From covariance = (J^T J)^{-1} * sigma^2
    - **Performance**: O(N_obs * N_grid) for forward; suitable for regional.
    - **Limitations**: Assumes isotropic; no viscoelastic; Okada simplified.
    - **Dependencies**: scipy.optimize, numpy.
    - **Reproducibility**: Fixed seed for random init if needed.
    - **Error Handling**: Convergence if cost < 1.5 * initial.

    Examples
    --------
    >>> inverter = InSARInverter(poisson=0.25, modulus=80e9)
    >>> results = inverter.invert(data, source_type='mogi', inc=0.3)
    >>> volume = results.metadata['volume_change']  # m³
    """

    def __init__(self, poisson: float = 0.25, modulus: float = 80e9):
        self.poisson = poisson
        self.modulus = modulus  # Pa

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Invert LOS displacements for deformation source.

        Applies atm correction, sets up forward model (Mogi/Okada), performs
        grid search + least-squares, estimates volume/pressure, uncertainty.
        Interpolates source to 3D grid (delta at source location).

        Parameters
        ----------
        data : ProcessedGrid
            LOS displacements ('data' in mm; coords lat/lon; metadata: inc, head).
        **kwargs : dict, optional
            - 'source_type': 'mogi' or 'okada' (default: 'mogi')
            - 'inc': float, incidence rad (default: from metadata)
            - 'head': float, heading rad
            - 'grid_bounds': Dict for search grid (default: data extent ±50km)
            - 'max_iterations': int (default: 100)
            - 'random_seed': int
            - 'atm_correction': bool (default: True)

        Returns
        -------
        InversionResults
            'model' 3D volume change proxy, metadata with source params.

        Raises
        ------
        GAMError
            Invalid data or params.
        InversionConvergenceError
            Non-convergence.
        """
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)

        # Extract data
        if 'data' not in data.ds:
            raise GAMError("Missing LOS 'data'")
        los_disp = data.ds['data'].values.flatten() / 1000.0  # mm to m
        mask = ~np.isnan(los_disp)
        if not np.any(mask):
            raise GAMError("No valid LOS data")
        los_disp = los_disp[mask]
        obs_xy = np.column_stack([data.ds['lon'][mask].ravel() * 111000,  # deg to m approx
                                  data.ds['lat'][mask].ravel() * 111000 * np.cos(np.deg2rad(data.ds['lon'].mean()))])
        
        # Radar geometry
        inc = kwargs.get('inc', data.ds.attrs.get('incidence', 0.3))  # rad
        head = kwargs.get('head', data.ds.attrs.get('heading', 0.0))  # rad
        obs_los = np.column_stack([obs_xy, np.zeros(len(obs_xy))])  # z=0

        # Atmospheric correction (planar ramp)
        if kwargs.get('atm_correction', True):
            los_disp = self._correct_atmosphere(obs_xy, los_disp)

        # Noise estimation
        noise_std = np.std(los_disp) * 0.1  # 10% relative

        # Source type
        source_type = kwargs.get('source_type', 'mogi')
        if source_type == 'mogi':
            forward_func = lambda p, obs: mogi_forward(p, obs, inc, head, self.poisson)
            n_params = 4  # x,y,depth,dv
            param_names = ['x', 'y', 'depth', 'dv']
        elif source_type == 'okada':
            forward_func = lambda p, obs: okada_forward(p, obs[:, :2], inc, head, self.poisson)
            n_params = 8  # x0,y0,depth,L,W,strike,dip,slip
            param_names = ['x0', 'y0', 'depth', 'L', 'W', 'strike', 'dip', 'slip']
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # Grid search for initial guess
        bounds = kwargs.get('grid_bounds', {'x': [obs_xy[:,0].min()-50000, obs_xy[:,0].max()+50000],
                                            'y': [obs_xy[:,1].min()-50000, obs_xy[:,1].max()+50000],
                                            'depth': [0, 10000]})
        init_guess = self._grid_search(forward_func, los_disp, obs_los, bounds, n_grid=10)

        # Bounds for lsq
        if source_type == 'mogi':
            bounds_lsq = ([bounds['x'][0], bounds['y'][0], 0, -1e6],  # dv can be negative (deflation)
                          [bounds['x'][1], bounds['y'][1], bounds['depth'][1], 1e6])
        else:
            bounds_lsq = ([bounds['x'][0], bounds['y'][0], 0, 1000, 1000, 0, 0, -10],
                          [bounds['x'][1], bounds['y'][1], bounds['depth'][1], 50000, 50000, 360, 90, 10])

        # Least squares inversion
        res = least_squares(
            lambda p: (forward_func(p, obs_los) - los_disp) / noise_std,
            init_guess,
            bounds=bounds_lsq,
            max_nfev=kwargs.get('max_iterations', 100),
            method='trf'
        )

        if res.cost > 1.5 * len(los_disp):  # Normalized chi2 /2 >1.5
            raise InversionConvergenceError(f"Cost={res.cost:.2f} > threshold")

        params = res.x
        pred_disp = forward_func(params, obs_los)

        # Uncertainty (Jacobian approx)
        J = self._compute_jacobian(forward_func, params, obs_los)
        cov = np.linalg.pinv(J.T @ J) * noise_std**2
        unc_params = np.sqrt(np.diag(cov))

        # Volume/pressure
        if source_type == 'mogi':
            dv = params[3]
            pressure = (dv * self.modulus * (1 - self.poisson)) / ((1 + self.poisson) * (1 - 2 * self.poisson)) / 1e6  # MPa, approx
        else:
            dv = params[3] * params[4] * params[7]  # L*W*slip
            pressure = 0.0  # Simplified

        # 3D model (delta at source)
        model_3d = np.zeros((len(data.ds['lat']), len(data.ds['lon']), len(data.ds['depth'])))
        # Find grid index for source
        src_x_idx = np.argmin(np.abs(data.ds['lon'].values * 111000 - params[0]))
        src_y_idx = np.argmin(np.abs(data.ds['lat'].values * 111000 * np.cos(np.deg2rad(data.ds['lon'].mean())) - params[1]))
        src_z_idx = np.argmin(np.abs(data.ds['depth'].values - params[2]))
        model_3d[src_y_idx, src_x_idx, src_z_idx] = dv / np.prod([data.ds['lat'].diff().mean(), 
                                                                  data.ds['lon'].diff().mean() * 111000, 
                                                                  data.ds['depth'].diff().mean()])  # Density
        unc_3d = np.zeros_like(model_3d)
        unc_3d[src_y_idx, src_x_idx, src_z_idx] = unc_params[3] / np.prod([data.ds['lat'].diff().mean(), 
                                                                           data.ds['lon'].diff().mean() * 111000, 
                                                                           data.ds['depth'].diff().mean()])

        metadata = {
            'converged': res.cost <= 1.5 * len(los_disp),
            'iterations': res.nfev,
            'residuals': np.sum(res.fun**2),
            'units': 'm³',
            'algorithm': f'insar_{source_type}',
            'parameters': dict(zip(param_names, params)),
            'uncertainty_params': dict(zip(param_names, unc_params)),
            'volume_change': dv,
            'pressure_change': pressure,
            'source_type': source_type,
            'incidence': inc,
            'heading': head,
        }

        results = InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)
        logger.info(f"InSAR inversion ({source_type}): volume={dv:.2e} m³, converged={metadata['converged']}")
        return results

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """
        Fuse InSAR volume changes with other models.

        Weighted sum of volume proxies.

        Parameters
        ----------
        models : List[InversionResults]
            Including InSAR.
        **kwargs : dict
            'weights': List[float]

        Returns
        -------
        np.ndarray
            Fused 3D volume density.
        """
        weights = kwargs.get('weights', np.ones(len(models)) / len(models))
        fused = sum(w * m.model for w, m in zip(weights, models))
        return fused

    def _correct_atmosphere(self, obs_xy: npt.NDArray[np.float64], los_disp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Remove linear atmospheric ramp via least squares."""
        A = np.column_stack([np.ones(len(obs_xy)), obs_xy[:, 0], obs_xy[:, 1]])  # offset, x, y
        coeffs, _, _, _ = np.linalg.lstsq(A, los_disp, rcond=None)
        ramp = A @ coeffs
        corrected = los_disp - ramp
        logger.debug(f"Atm correction: coeffs={coeffs}")
        return corrected

    def _grid_search(self, forward_func, los_disp: npt.NDArray[np.float64], obs_los: npt.NDArray[np.float64],
                     bounds: Dict[str, List[float]], n_grid: int = 10) -> np.ndarray:
        """Coarse grid search for initial params."""
        x_grid = np.linspace(bounds['x'][0], bounds['x'][1], n_grid)
        y_grid = np.linspace(bounds['y'][0], bounds['y'][1], n_grid)
        depth_grid = np.linspace(bounds['depth'][0], bounds['depth'][1], 5)
        
        best_cost = np.inf
        best_params = None
        for x in x_grid:
            for y in y_grid:
                for d in depth_grid:
                    # Rough dv estimate (scale to data range)
                    pred = forward_func(np.array([x, y, d, np.std(los_disp)]), obs_los)
                    cost = np.sum((pred - los_disp)**2)
                    if cost < best_cost:
                        best_cost = cost
                        best_params = np.array([x, y, d, np.std(los_disp)])
        
        if best_params is None:
            # Fallback to center
            best_params = np.array([np.mean(bounds['x']), np.mean(bounds['y']), bounds['depth'][0], np.std(los_disp)])
        
        logger.info(f"Grid search best cost: {best_cost:.2f}")
        return best_params

    def _compute_jacobian(self, forward_func, params: np.ndarray, obs_los: npt.NDArray[np.float64], 
                          eps: float = 1e-6) -> npt.NDArray[np.float64]:
        """Numerical Jacobian."""
        n_params = len(params)
        J = np.zeros((len(obs_los), n_params))
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            pred_plus = forward_func(params_plus, obs_los)
            J[:, i] = (pred_plus - forward_func(params - eps * np.eye(n_params)[i], obs_los)) / (2 * eps)
        return J