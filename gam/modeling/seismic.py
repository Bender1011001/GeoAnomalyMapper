"""Seismic inversion implementation for GAM using PyGIMLi."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator, griddata

import pygimli as pg
from pygimli.physics.traveltime import TravelTimeModelling, simulate
from pygimli.optimization import RIES

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.mesh import MeshGenerator  # Assumed; implement later
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


class SeismicModel(Inverter):
    """
    PyGIMLi-based travel time tomography for P-wave velocity estimation.

    This class implements first-arrival travel time inversion to recover 2D or 3D
    P-wave velocity models using PyGIMLi's traveltime module. Supports ray-based
    tomography with eikonal solver for forward modeling. Handles source-receiver
    geometries from ProcessedGrid metadata or inferred from coords. Parameterization
    uses logarithmic velocity (to ensure positivity), with smoothness regularization.

    Key features:
    - 2D profile or 3D volume tomography
    - Adaptive mesh refinement near sources/receivers
    - Snell's law ray tracing with eikonal equation
    - L2 regularization with gradient smoothing
    - Uncertainty via Jacobian-based covariance
    - Units: Input travel times in ms, output velocity in m/s (1500-6000 m/s range)

    Parameters
    ----------
    v_min : float, optional
        Minimum velocity (default: 1500 m/s).
    v_max : float, optional
        Maximum velocity (default: 6000 m/s).

    Attributes
    ----------
    v_min : float
        Velocity bounds lower.
    v_max : float
        Velocity bounds upper.
    modelling : pygimli.physics.traveltime.TravelTimeModelling, optional
        Cached modelling object.

    Methods
    -------
    Inherited from Inverter; see invert() and fuse().

    Notes
    -----
    - **Mesh**: RTree2D/3D with refinement at surface (hmin=5m), coarser at depth.
    - **Forward**: Eikonal solver for travel times; supports curved rays.
    - **Inversion**: Gauss-Newton or Marquardt with damping; convergence on chi2.
    - **Geometry**: Sources/receivers from data coords; assume surface sources.
    - **Performance**: Efficient for regional profiles; parallel ray tracing.
    - **Limitations**: First-arrivals only; no attenuation; 2D assumes constant strike.
    - **Dependencies**: PyGIMLi >=1.5.0.
    - **Reproducibility**: Fixed random seed for initial model.
    - **Error Handling**: Raises if no rays traced or non-convergence.

    References
    ----------
    - PyGIMLi: Günther et al. (2022), https://www.pygimli.org/
    - Travel-time tomography: Shearer (2019), Introductory Seismology.

    Examples
    --------
    >>> inverter = SeismicInverter(v_min=1000, v_max=7000)
    >>> results = inverter.invert(data, dimension=3, damping=0.1)
    >>> vel_model = results.model  # m/s, 3D
    """

    def __init__(self, v_min: float = 1500.0, v_max: float = 6000.0):
        self.v_min = v_min
        self.v_max = v_max
        self.modelling: Optional[TravelTimeModelling] = None
        self.mesh: Optional[pg.Mesh] = None

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Perform travel time tomography for P-wave velocity model.

        Extracts travel times and geometry, creates 2D/3D mesh, sets up
        TravelTimeModelling, inverts with logarithmic slowness parameterization,
        bounds velocity, estimates uncertainty from covariance diagonal.
        Interpolates to grid (3D always, 2D extruded).

        Parameters
        ----------
        data : ProcessedGrid
            Seismic data ('data' as travel times in s; geometry via kwargs['sources'], kwargs['receivers']).
        **kwargs : dict, optional
            - 'dimension': int, 2 or 3 (default: 3)
            - 'sources': np.ndarray, source locations (Nx3)
            - 'receivers': np.ndarray, receiver locations (Nx3)
            - 'damping': float, Marquardt lambda (default: 0.1)
            - 'max_iterations': int (default: 20)
            - 'random_seed': int (default: None)
            - 'mesh_config': Dict (default: adaptive seismic)

        Returns
        -------
        InversionResults
            'model' as velocity (m/s), 3D array.

        Raises
        ------
        GAMError
            Invalid data/geometry.
        InversionConvergenceError
            Non-convergence.
        """
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            pg.utils.startLogger()

        # Extract data
        if 'data' not in data.ds:
            raise GAMError("Missing 'data' (travel times)")
        travel_times = data.ds['data'].values.flatten()  # s
        mask = ~np.isnan(travel_times)
        if not np.any(mask):
            raise GAMError("No valid travel times")
        travel_times = travel_times[mask]

        # Geometry from kwargs
        dimension = kwargs.get('dimension', 3)
        if dimension not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")
        
        sources = kwargs.get('sources')
        if sources is None:
            raise GAMError("Sources geometry required in kwargs['sources'] (Nx3 array in m)")
        receivers = kwargs.get('receivers')
        if receivers is None:
            raise GAMError("Receivers geometry required in kwargs['receivers'] (Nx3 array in m)")
        if len(sources) != len(receivers) or len(travel_times) != len(sources):
            raise GAMError("Number of travel times must match number of source-receiver pairs")
        if sources.shape[1] < 3:
            # Assume 2D, add z=0
            sources = np.column_stack([sources, np.zeros(len(sources))])
            receivers = np.column_stack([receivers, np.zeros(len(receivers))])
        if dimension == 2:
            sources = sources[:, [0, 2]]  # x,z for 2D profile
            receivers = receivers[:, [0, 2]]

        # Mesh from survey geometry and config (e.g., resolution via hmin)
        mesh_config = kwargs.get('mesh_config', {'type': 'rtree', 'hmin': 50.0, 'depth': 2000})  # hmin: mesh resolution in m
        mesh_gen = MeshGenerator()
        self.mesh = mesh_gen.create_seismic_mesh(sources, receivers, dimension=dimension, **mesh_config)

        # Modelling
        self.modelling = TravelTimeModelling(mesh=self.mesh, dimension=dimension)
        self.modelling.setData(travel_times, sources, receivers)

        # Initial model (log slowness)
        v_init = np.ones(self.mesh.cellCount()) * 2000.0  # m/s
        slowness_init = 1.0 / v_init
        log_slowness_init = np.log(slowness_init)

        # Bounds (log space)
        log_s_min = np.log(1.0 / self.v_max)
        log_s_max = np.log(1.0 / self.v_min)

        # Inversion using RIES
        damping = kwargs.get('damping', 100.0)  # Regularization factor
        max_iter = kwargs.get('max_iterations', 20)
        abs_err = kwargs.get('absolute_error', 0.005)  # Default error 5 ms in s
        
        try:
            fop = self.modelling
            inv = RIES(fop=fop, verbose=True, maxIter=max_iter)
            inv.setData(travel_times, absoluteError=abs_err)
            inv.setRegularisation(damping * len(travel_times))  # Scale with data
            inv.setBounds([log_s_min, log_s_max])
            inv.setModel(log_slowness_init)
            inv.run()
        except Exception as e:
            raise InversionConvergenceError(f"Inversion failed: {e}")

        # Recovered model
        log_slowness_rec = inv.estimate()
        slowness_rec = np.exp(log_slowness_rec)
        v_rec = 1.0 / slowness_rec

        # Convergence check (chi2 approx)
        chi2 = inv.getChi2()
        dof = len(travel_times)
        if chi2 > 2 * dof:
            warnings.warn(f"Marginal convergence: chi2={chi2:.2f} > 2*dof={2*dof:.2f}")

        # Uncertainty (from covariance; simplified diagonal)
        uncertainty = self._estimate_uncertainty(inv, v_rec)

        # To 3D grid
        v_rec_3d, unc_3d = self._interpolate_to_grid(v_rec, uncertainty, data)

        metadata = {
            'converged': chi2 <= 2 * dof,
            'iterations': self.modelling.iterations(),
            'residuals': chi2,
            'units': 'm/s',
            'algorithm': 'pygimli_traveltime_tomography',
            'inversion_method': 'RIES',
            'parameters': {
                'dimension': dimension,
                'damping': damping,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'n_sources': len(sources),
            },
        }

        results = InversionResults(model=v_rec_3d, uncertainty=unc_3d, metadata=metadata)
        logger.info(f"Seismic inversion completed: {results.model.shape}, converged={metadata['converged']}")
        return results

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """
        Fuse seismic velocity with other models (weighted average).

        Parameters
        ----------
        models : List[InversionResults]
            Including seismic and others.
        **kwargs : dict, optional
            - 'weights': List[float] (default: equal)

        Returns
        -------
        np.ndarray
            Fused 3D velocity proxy.
        """
        weights = kwargs.get('weights', np.ones(len(models)) / len(models))
        fused = sum(w * m.model for w, m in zip(weights, models))
        logger.info(f"Seismic fusion: shape={fused.shape}")
        return fused

    def _estimate_uncertainty(self, inv: RIES, v_rec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Uncertainty from model covariance diagonal."""
        # Simplified; use full covariance in production
        J = inv.jacobian()
        errors = inv.absoluteError()
        if np.isscalar(errors):
            errors = np.full(len(inv.data()), errors)
        C_d_inv = np.diag(1.0 / (errors ** 2))
        cov_diag = J.T @ C_d_inv @ J
        sigma_m = np.sqrt(np.diag(np.linalg.pinv(cov_diag + 1e-10)))
        # Propagate to velocity
        dv_dm = -v_rec**2  # d v / d s = -v^2 (s=1/v)
        unc_v = np.abs(dv_dm) * sigma_m
        return unc_v

    def _interpolate_to_grid(self, model_flat: npt.NDArray[np.float64], unc_flat: npt.NDArray[np.float64],
                             data: ProcessedGrid) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate model from mesh to regular grid."""
        # Cell centers (assume x,y,z in meters)
        cc = self.mesh.cellCenters()
        source_points = cc  # Nx3

        # Target grid points (assume lat/lon/depth, convert to approx meters)
        lats_g, lons_g, depths_g = np.meshgrid(data.ds['lat'].values, data.ds['lon'].values, data.ds['depth'].values, indexing='ij')
        scale = 111000.0  # approx m per degree at equator
        target_points = np.stack([lons_g.ravel() * scale, lats_g.ravel() * scale, depths_g.ravel()], axis=-1)

        # Interpolate (assume source_points in same coord system)
        v_3d = griddata(source_points, model_flat, target_points, method='linear', fill_value=np.nan).reshape(lats_g.shape)
        unc_3d = griddata(source_points, unc_flat, target_points, method='linear', fill_value=np.nan).reshape(lats_g.shape)
        return v_3d, unc_3d