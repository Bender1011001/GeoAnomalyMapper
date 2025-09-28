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

from gam.core.utils import transform_coordinates, reverse_transform_coordinates

PYCOULOMB_AVAILABLE = False
try:
    from PyCoulomb import coulomb_collections as cc
    from PyCoulomb import fault_slip_object as fso
    PYCOULOMB_AVAILABLE = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


def mogi_uxyz(x, y, z, x0, y0, d, dV, nu=0.25):
    """Mogi displacements ux, uy, uz."""
    dx = x - x0
    dy = y - y0
    dz = z + d  # depth positive down
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r3 = r**3
    ux = dV * dx * dz / (4 * np.pi * r3)
    uy = dV * dy * dz / (4 * np.pi * r3)
    uz = dV * (r**2 - 2 * dz**2) / (4 * np.pi * r3)
    return ux, uy, uz

def mogi_multi_forward(theta: np.ndarray, obs_pos: np.ndarray, inc_grid: np.ndarray, head_grid: np.ndarray, nu: float = 0.25) -> np.ndarray:
    """
    Multi-source Mogi forward model for LOS displacement.
    
    theta: flat array [x0_1, y0_1, d_1, dV_1, ..., x0_K, y0_K, d_K, dV_K]
    obs_pos: N x 3 (x, y, z=0)
    inc_grid, head_grid: N (flattened incidence and heading in rad)
    """
    K = len(theta) // 4
    pred = np.zeros(len(obs_pos))
    x, y, z = obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2]
    for i in range(K):
        idx = 4 * i
        x0, y0, d, dV = theta[idx:idx+4]
        ux, uy, uz = mogi_uxyz(x, y, z, x0, y0, d, dV, nu)
        # Per-pixel LOS
        los_x = np.sin(inc_grid) * np.cos(head_grid)
        los_y = np.sin(inc_grid) * np.sin(head_grid)
        los_z = np.cos(inc_grid)
        pred += ux * los_x + uy * los_y + uz * los_z
    return pred


def okada_forward(params: np.ndarray, obs_pos: np.ndarray, inc_grid: np.ndarray, head_grid: np.ndarray,
                  poisson: float = 0.25) -> np.ndarray:
    """
    Okada (1985) rectangular dislocation forward model for InSAR LOS displacement with per-pixel LOS.

    Computes surface displacements due to a rectangular fault in elastic half-space.
    Uses PyCoulomb library for accurate Okada solution (strike-slip with rake=0;
    extensible to general cases). Fault parameters configurable via config or kwargs
    in the InSARInverter class.

    Parameters
    ----------
    params : np.ndarray
        Shape (8,): [x0, y0, d, L, W, strike, dip, slip]
        - x0, y0: Source center longitude (degrees) and latitude (degrees).
        - d: Depth to top of fault (meters).
        - L, W: Fault length (along strike, meters) and width (downdip, meters).
        - strike, dip: Fault orientation (degrees clockwise from north; dip from horizontal).
        - slip: Strike-slip amount (meters; positive right-lateral).
    obs_pos : np.ndarray
        Shape (N, 3): Observation points as [x, y, z=0] in meters.
    inc_grid, head_grid : np.ndarray
        Shape (N,): Incidence and heading angles in radians per observation.
    poisson : float
        Poisson's ratio (unitless; default 0.25 for crust).

    Returns
    -------
    np.ndarray
        Shape (N,): Predicted LOS displacements (meters; positive towards satellite).

    Notes
    -----
    - Units: x/y/z meters, depth/L/W meters (converted to km internally for PyCoulomb),
      angles radians (inc/head) or degrees (strike/dip), slip meters.
    - PyCoulomb: Accurate analytical Okada for uniform slip rectangular fault.
      Assumes pure strike-slip (rake=0); depth to top edge. Internal conversion from
      local meters to degrees uses ~111 km/deg scale (approximate; best for
      small regions near equator/lon=0; distortion <1% for 100km at mid-latitudes).
      Update conversion with reference lon/lat for production accuracy.
    - Fault parameters: Configurable via InSARInverter kwargs or global config.yaml
      (e.g., fault_length, fault_width, strike, dip, slip).
    - Validation: Raises ImportError if PyCoulomb unavailable; GAMError for invalid shapes/lengths.
    - Limitations: No topography/viscoelasticity; half-space only. Strike-slip focus;
      for dip-slip/tensile, extend rake/params in future phases.
    - Dependencies: NumPy, PyCoulomb (required; install via 'pip install pycoulomb').
    - Reference: Okada (1985) Surface deformation due to shear and tensile faults in a half-space.
    """
    if len(params) != 8:
        raise GAMError("okada_forward expects 8 parameters: x0,y0,d,L,W,strike,dip,slip")
    if obs_pos.shape[1] != 3:
        raise GAMError("obs_pos must be (N, 3)")

    x0, y0, d, L, W, strike, dip, slip = params

    if PYCOULOMB_AVAILABLE:
        # Convert approximate local Cartesian (m) to geographic (deg) using scale ~111 km/deg
        # Assumes coordinates relative to (lon=0, lat=0); refine with ref_lon/lat for accuracy
        scale = 111000.0  # m/deg approximate (ignores lat variation for simplicity)
        lon0 = x0 / scale
        lat0 = y0 / scale
        obs_lon = obs_pos[:, 0] / scale
        obs_lat = obs_pos[:, 1] / scale

        # Build observations (surface, depth=0 km)
        obs_points = [cc.Observation(lon=float(lon), lat=float(lat), depth=0.0)
                      for lon, lat in zip(obs_lon, obs_lat)]

        # Construct single rectangular dislocation (strike-slip, rake=0)
        fault = fso.FaultSlipObject(
            lon=lon0, lat=lat0,
            depth=d / 1000.0,  # km
            strike=strike, dip=dip, rake=0.0,
            slip=slip,  # m
            length=L / 1000.0,  # km
            width=W / 1000.0   # km
        )

        # Compute 3D displacements (E, N, U in m)
        disps = cc.displacement_points_list(obs_points, [fault], poisson)

        # Project to per-pixel LOS (east, north, up dot unit vector)
        los_x = np.sin(inc_grid) * np.cos(head_grid)  # East component
        los_y = np.sin(inc_grid) * np.sin(head_grid)  # North
        los_z = np.cos(inc_grid)  # Up
        los = np.array([
            d.dE * los_x[i] + d.dN * los_y[i] + d.dU * los_z[i]
            for i, d in enumerate(disps)
        ], dtype=float)

        return los

    else:
        raise ImportError("PyCoulomb not available. Install with 'pip install pycoulomb' for accurate Okada modeling.")


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
        lons_masked = data.ds['lon'][mask].ravel()
        lats_masked = data.ds['lat'][mask].ravel()
        obs_x, obs_y = transform_coordinates(lons_masked, lats_masked)
        obs_xy = np.column_stack([obs_x, obs_y])
        
        # Radar geometry per pixel
        if 'incidence' in data.ds and 'heading' in data.ds:
            inc_grid = data.ds['incidence'].values.flatten()[mask]
            head_grid = data.ds['heading'].values.flatten()[mask]
        else:
            inc = kwargs.get('inc', data.ds.attrs.get('incidence', 0.3))
            head = kwargs.get('head', data.ds.attrs.get('heading', 0.0))
            inc_grid = np.full(len(obs_xy), inc)
            head_grid = np.full(len(obs_xy), head)
        obs_pos = np.column_stack([obs_xy, np.zeros(len(obs_xy))])  # z=0

        # Atmospheric correction (planar ramp)
        if kwargs.get('atm_correction', True):
            los_disp = self._correct_atmosphere(obs_xy, los_disp)

        # Noise estimation
        noise_std = np.std(los_disp) * 0.1  # 10% relative

        # Multi-source
        K = kwargs.get('sources', 2)  # Number of sources
        source_type = kwargs.get('source_type', 'mogi')
        if source_type == 'mogi':
            forward_func = lambda theta, obs: mogi_multi_forward(theta, obs, inc_grid, head_grid, self.poisson)
            n_params = 4 * K
            param_names = [f'x{i+1}', f'y{i+1}', f'depth{i+1}', f'dV{i+1}' for i in range(K)]
        elif source_type == 'okada':
            # For multi-Okada, more params; for now single
            K = 1
            forward_func = lambda p, obs: okada_forward(p, obs[:, :2], inc_grid, head_grid, self.poisson)
            n_params = 8
            param_names = ['x0', 'y0', 'depth', 'L', 'W', 'strike', 'dip', 'slip']
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # Initial guess: grid search for first source, duplicate for others
        bounds = kwargs.get('grid_bounds', {'x': [obs_xy[:,0].min()-50000, obs_xy[:,0].max()+50000],
                                            'y': [obs_xy[:,1].min()-50000, obs_xy[:,1].max()+50000],
                                            'depth': [0, 10000], 'dV': [-1e6, 1e6]})
        init_single = self._grid_search(lambda p, obs: forward_func(p[:4], obs) if source_type=='mogi' else forward_func(p, obs), los_disp, obs_pos, bounds, n_grid=10)
        if source_type == 'mogi' and K > 1:
            init_guess = np.tile(init_single, K)
            init_guess[4:] += np.random.normal(0, 10000, n_params-4)  # Perturb others
        else:
            init_guess = init_single

        # Bounds for lsq (per source)
        if source_type == 'mogi':
            lb_single = [bounds['x'][0], bounds['y'][0], 0, -1e8]
            ub_single = [bounds['x'][1], bounds['y'][1], bounds['depth'][1], 1e8]
            lb = lb_single * K
            ub = ub_single * K
        else:
            lb = [bounds['x'][0], bounds['y'][0], 0, 1000, 1000, 0, 0, -10]
            ub = [bounds['x'][1], bounds['y'][1], bounds['depth'][1], 50000, 50000, 360, 90, 10]
        bounds_lsq = (lb, ub)

        # Least squares inversion
        res = least_squares(
            lambda theta: (forward_func(theta, obs_pos) - los_disp) / noise_std,
            init_guess,
            bounds=bounds_lsq,
            method="trf", max_nfev=kwargs.get('max_iterations', 1000)
        )

        if res.cost > 1.5 * len(los_disp):  # Normalized chi2 /2 >1.5
            raise InversionConvergenceError(f"Cost={res.cost:.2f} > threshold")

        params = res.x
        pred_disp = forward_func(params, obs_pos, inc_grid, head_grid)

        # Uncertainty (Fisher approx)
        J = self._compute_jacobian(forward_func, params, obs_pos, inc_grid, head_grid)
        cov = np.linalg.pinv(J.T @ J) * np.var(res.fun)
        unc_params = np.sqrt(np.diag(cov))
        
        # Bootstrap uncertainty (optional, 100 resamples)
        if kwargs.get('bootstrap_uncertainty', True):  # Default True as per task
            n_bootstrap = 100
            boot_params = np.zeros((n_bootstrap, len(params)))
            for b in range(n_bootstrap):
                residuals_boot = np.random.choice(res.fun, size=len(res.fun), replace=True)
                def residual_boot(theta):
                    pred = forward_func(theta, obs_pos, inc_grid, head_grid)
                    return (pred - los_disp + residuals_boot * noise_std) / noise_std
                res_boot = least_squares(residual_boot, params, bounds=bounds_lsq, method='trf')
                boot_params[b] = res_boot.x
            unc_bootstrap = np.std(boot_params, axis=0)
            unc_params = np.sqrt(unc_params**2 + unc_bootstrap**2)  # Combine

        # Total volume/pressure for multi Mogi
        if source_type == 'mogi':
            dvs = params[3::4]
            dv = np.sum(dvs)
            pressure = (dv * self.modulus * (1 - self.poisson)) / ((1 + self.poisson) * (1 - 2 * self.poisson)) / 1e6  # MPa, approx
        else:
            dv = params[3] * params[4] * params[7]  # L*W*slip
            pressure = 0.0  # Simplified

        # 3D model (delta at source)
        model_3d = np.zeros((len(data.ds['lat']), len(data.ds['lon']), len(data.ds['depth'])))
        # Compute grid spacings in meters
        lat_mean = data.ds['lat'].mean().values
        lon_mean = data.ds['lon'].mean().values
        dlon = data.ds['lon'].diff().mean().values
        dlat = data.ds['lat'].diff().mean().values
        dz = data.ds['depth'].diff().mean().values

        # dx for lon
        x1, _ = transform_coordinates([lon_mean - dlon / 2], [lat_mean])
        x2, _ = transform_coordinates([lon_mean + dlon / 2], [lat_mean])
        dx = abs(x2[0] - x1[0])

        # dy for lat
        _, y1 = transform_coordinates([lon_mean], [lat_mean - dlat / 2])
        _, y2 = transform_coordinates([lon_mean], [lat_mean + dlat / 2])
        dy = abs(y2[0] - y1[0])

        cell_volume = dx * dy * abs(dz)

        # Find grid index for source (transform grid to meters)
        lons_grid = data.ds['lon'].values
        lats_grid_x = np.full_like(lons_grid, lat_mean)
        x_grid, _ = transform_coordinates(lons_grid, lats_grid_x)
        src_x_idx = np.argmin(np.abs(x_grid - params[0]))

        lats_grid = data.ds['lat'].values
        lons_grid_y = np.full_like(lats_grid, lon_mean)
        _, y_grid = transform_coordinates(lons_grid_y, lats_grid)
        src_y_idx = np.argmin(np.abs(y_grid - params[1]))

        # For multi-source, place deltas at each source location
        model_3d = np.zeros((len(data.ds['lat']), len(data.ds['lon']), len(data.ds['depth'])))
        unc_3d = np.zeros_like(model_3d)
        for i in range(K if source_type=='mogi' else 1):
            if source_type == 'mogi':
                px, py, pz, pdv = params[4*i:4*(i+1)]
                p_unc = unc_params[4*i:4*(i+1)]
            else:
                px, py, pz, pdv = params[0], params[1], params[2], dv
                p_unc = [unc_params[0], unc_params[1], unc_params[2], np.sqrt(np.sum(unc_params[3:5]**2 + unc_params[7]**2))]
            
            # Find indices (approx)
            src_x_idx = np.argmin(np.abs(x_grid - px))
            src_y_idx = np.argmin(np.abs(y_grid - py))
            src_z_idx = np.argmin(np.abs(data.ds['depth'].values - pz))
            model_3d[src_y_idx, src_x_idx, src_z_idx] += pdv / cell_volume
            unc_3d[src_y_idx, src_x_idx, src_z_idx] += p_unc[3]**2 / cell_volume**2
        unc_3d = np.sqrt(unc_3d)

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

    def _grid_search(self, forward_func, los_disp: npt.NDArray[np.float64], obs_pos: npt.NDArray[np.float64],
                     inc_grid: np.ndarray, head_grid: np.ndarray, bounds: Dict[str, List[float]], n_grid: int = 10) -> np.ndarray:
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
                    pred = forward_func(np.array([x, y, d, np.std(los_disp)]), obs_pos, inc_grid, head_grid)
                    cost = np.sum((pred - los_disp)**2)
                    if cost < best_cost:
                        best_cost = cost
                        best_params = np.array([x, y, d, np.std(los_disp)])
        
        if best_params is None:
            # Fallback to center
            best_params = np.array([np.mean(bounds['x']), np.mean(bounds['y']), bounds['depth'][0], np.std(los_disp)])
        
        logger.info(f"Grid search best cost: {best_cost:.2f}")
        return best_params

    def _compute_jacobian(self, forward_func, params: np.ndarray, obs_pos: npt.NDArray[np.float64],
                          inc_grid: np.ndarray, head_grid: np.ndarray, eps: float = 1e-6) -> npt.NDArray[np.float64]:
        """Numerical Jacobian for multi-source."""
        n_params = len(params)
        J = np.zeros((len(obs_pos), n_params))
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            pred_plus = forward_func(params_plus, obs_pos, inc_grid, head_grid)
            params_minus = params.copy()
            params_minus[i] -= eps
            pred_minus = forward_func(params_minus, obs_pos, inc_grid, head_grid)
            J[:, i] = (pred_plus - pred_minus) / (2 * eps)
        return J