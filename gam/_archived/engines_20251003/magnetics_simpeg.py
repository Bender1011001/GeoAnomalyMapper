"""Magnetic inversion implementation for GAM using SimPEG."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator, griddata

import simpeg
from simpeg import (
    data as spdata,
    data_misfit,
    directives,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
    simulation,
    utils,
)
from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg import inverse_problem
from simpeg.potential_fields import magnetics
# depth_weighting location varies across SimPEG versions; provide robust fallback
try:
    # SimPEG <= 0.19 style
    from simpeg.potential_fields.utils import depth_weighting  # type: ignore
except Exception:
    try:
        # Some versions expose it under simpeg.utils
        from simpeg.utils import depth_weighting  # type: ignore
    except Exception:
        # Minimal fallback: uniform weights for active cells
        def depth_weighting(mesh, indActive=None, exponent: float = 2.0):
            n = int(np.sum(indActive)) if indActive is not None else (
                int(getattr(mesh, "n_cells", 0)) or int(getattr(mesh, "nC", 0)) or 0
            )
            return np.ones(n, dtype=float)

from gam.core.utils import transform_coordinates

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.mesh import MeshGenerator  # Assumed available; implement in mesh.py
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


class MagneticInverter(Inverter):
    """
    SimPEG-based magnetic inversion for magnetic susceptibility estimation.

    This class implements total magnetic intensity (TMI) inversion to recover
    3D susceptibility models using SimPEG's potential_fields.magnetics module.
    Supports adaptive TreeMesh, linear inversion for compact sources, and smooth
    inversions with depth weighting. Parameterization directly maps to susceptibility
    (SI units). Handles regional field parameters (inclination, declination) and
    assumes induced magnetization (no remanence).

    Key features:
    - Adaptive meshing refined near data locations
    - TMI forward modeling with spherical Earth approximation
    - Compact (L1) or smooth (L2) regularization options
    - Depth weighting to counter skin-depth decay
    - Uncertainty from model covariance diagonal
    - Units: Input TMI in nT, output susceptibility in SI (dimensionless)

    Parameters
    ----------
    mu_0 : float, optional
        Vacuum permeability (default: 4*pi*1e-7).
    background_sus : float, optional
        Background susceptibility (default: 0.0 SI).

    Attributes
    ----------
    mu_0 : float
        Magnetic constant.
    mesh : simpeg.mesh.TreeMesh, optional
        Cached mesh.
    survey : simpeg.survey.Survey, optional
        Magnetic survey setup.

    Methods
    -------
    Inherited from Inverter; see invert() and fuse() for details.

    Notes
    -----
    - **Mesh**: TreeMesh with surface refinement (hmin=5m), deeper coarsening.
    - **Forward**: Magnetic dipole sources on active cells; TMI projection.
    - **Inversion**: Projected Gauss-Newton; supports L1 for sparsity.
    - **Regional Field**: B0 vector from inclination/declination (degrees).
    - **Performance**: Efficient for airborne/ground surveys; vectorized.
    - **Limitations**: Assumes induced field; extend for remanent via MultiMagnetization.
    - **Dependencies**: SimPEG >=0.19.0.
    - **Reproducibility**: random_seed controls optimization.
    - **Error Handling**: Convergence checked against chi-squared.

    Examples
    --------
    >>> inverter = MagneticInverter(background_sus=0.01)
    >>> results = inverter.invert(data, inclination=60, declination=-10)
    >>> sus_model = results.model  # SI units, 3D array
    """

    def __init__(self, mu_0: float = 4 * np.pi * 1e-7, background_sus: float = 0.0):
        self.mu_0 = mu_0
        self.background_sus = background_sus
        self.mesh: Optional[TreeMesh] = None
        self.survey: Optional[simulation.Simulation3DLinear] = None
        self.inv: Optional[inversion.BaseInversion] = None

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Perform magnetic inversion to estimate 3D susceptibility.

        Generates mesh from data, sets up TMI survey with regional field from
        inclination/declination, uses IdentityMap for susceptibility, applies
        depth weighting, runs inversion with smoothing or compact regularization.
        Interpolates recovered model to input grid.

        Parameters
        ----------
        data : ProcessedGrid
            Magnetic data ('data' in nT, coords lat/lon/depth).
        **kwargs : dict, optional
            - 'inclination': float, degrees (default: 60)
            - 'declination': float, degrees (default: -10)
            - 'mesh_config': Dict for mesh (default: adaptive magnetic)
            - 'regularization': Dict {'type': 'smooth' or 'compact', 'alpha': 1e-3}
            - 'reference_model': np.ndarray (default: background_sus)
            - 'max_iterations': int (default: 15)
            - 'random_seed': int (default: None)
            - 'depth_weighting': bool (default: True)

        Returns
        -------
        InversionResults
            'model' as susceptibility (SI), 'uncertainty' from cov.

        Raises
        ------
        GAMError
            Invalid input or parameters.
        InversionConvergenceError
            Non-convergence.
        """
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            simpeg.utils.rand.seed(random_seed)

        # Validate and extract data
        if 'data' not in data.ds:
            raise GAMError("ProcessedGrid missing 'data' variable")
        observed = data.ds['data'].values.flatten()  # nT
        mask = ~np.isnan(observed)
        if not np.any(mask):
            raise GAMError("No valid data points")
        observed = observed[mask]

        # Regional field
        inclination = np.deg2rad(kwargs.get('inclination', 60.0))
        declination = np.deg2rad(kwargs.get('declination', -10.0))
        b0 = np.array([np.cos(inclination) * np.cos(declination),
                       np.cos(inclination) * np.sin(declination),
                       np.sin(inclination)])  # Unit vector

        # Mesh generation
        mesh_config = kwargs.get('mesh_config', {'type': 'tree', 'hmin': 5.0, 'hmax': 500.0})
        mesh_gen = MeshGenerator()
        self.mesh = mesh_gen.create_mesh(data, **mesh_config)
        active_cells = self._get_active_cells(data)

        # Survey setup
        locations = self._get_locations(data)[mask]  # Filter valid
        receiver_list = magnetics.receivers.Point(locations, components=("tmi",))
        src = magnetics.sources.SourceField(
           receiver_list=[receiver_list],
           parameters=magnetics.EarthField(B=np.linalg.norm(b0), inclination=np.rad2deg(inclination), declination=np.rad2deg(declination))
        )
        self.survey = magnetics.survey.Survey(source_field=src)

        # Simulation
        chi_map = maps.IdentityMap(nP=int(active_cells.sum()))
        sim = magnetics.simulation.Simulation3DIntegral(
            survey=self.survey, mesh=self.mesh, chiMap=chi_map, indActive=active_cells,
            store_sensitivities="forward_only",
        )

        # Build Data object with standard deviations for each observation.
        # SimPEG's least-squares formulation requires a per-datum standard deviation.
        # We assume a relative error (default 5%) of the absolute anomaly plus a noise floor (1 nT).
        rel_err = float(kwargs.get("relative_error", 0.05))
        floor_nt = float(kwargs.get("noise_floor_nt", 1.0))
        # Standard deviation in nT
        std = rel_err * np.abs(observed) + floor_nt
        data_obj = spdata.Data(dobs=observed)
        data_obj.standard_deviation = std
        dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)

        # Regularization with depth weighting.  Use the potential_fields.utils version.
        w_depth = depth_weighting(self.mesh, indActive=active_cells, exponent=2.0)
        reg_type = kwargs.get('reg_type', kwargs.get('regularization', {}).get('type', 'l2'))
        mapping = maps.IdentityMap(nP=int(active_cells.sum()))
        if reg_type == 'l2':
            reg = regularization.Simple(
                mesh=self.mesh, indActive=active_cells, mapping=mapping, cell_weights=w_depth,
            )
        else:  # TV
            reg = regularization.Sparse(
                mesh=self.mesh, indActive=active_cells, mapping=mapping, cell_weights=w_depth, norms=[0,1,1,1]
            )
        reg_params = kwargs.get('regularization', {'alpha_s': 1e-4, 'alpha_x': 1, 'alpha_y': 1, 'alpha_z': 1})
        reg.alpha_s = reg_params['alpha_s']
        reg.alpha_x = reg_params['alpha_x']
        reg.alpha_y = reg_params['alpha_y']
        reg.alpha_z = reg_params['alpha_z']

        # Reference model
        m0 = kwargs.get('reference_model', np.full(self.mesh.nC, self.background_sus))
        reg.mref = m0

        # Additional weighting using approximate sensitivity.  Normalize so that weights sum to unity.
        if kwargs.get("depth_weighting", True):
            try:
                wr = sim.getJtJdiag(m0) ** 0.5
                if np.linalg.norm(wr) > 0:
                    wr = wr / np.linalg.norm(wr)
                reg.cell_weights = wr
            except Exception:
                # Fall back to depth weighting only
                reg.cell_weights = w_depth

        # Optimization
        opt = optimization.InexactGaussNewton(maxIter=kwargs.get('max_iterations', 15))

        # Inversion setup
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
        beta0_ratio = float(kwargs.get("beta0_ratio", 1.0))
        # Target misfit: for L2 data misfit, chi^2 target ~ 0.5 * N (number of observations)
        N = len(observed)
        target_misfit = float(kwargs.get("target_misfit", 0.5 * N))
        directives_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
            directives.UpdateSensitivityWeights(),
            directives.BetaSchedule(coolingFactor=float(kwargs.get("beta_cooling", 2.0))),
            directives.TargetMisfit(target=target_misfit),
            directives.SaveOutputEveryIteration(save_txt=False),
        ]
        if reg_type != "l2":
            # Total variation / sparse. Use IRLS to approximate L1.
            directives_list.append(
                directives.Update_IRLS(f_min_change=1e-2, maxIRLSIter=10, beta_tol=1e-1, coolEps=1.5)
            )
        self.inv = inversion.BaseInversion(invProb, directiveList=directives_list)

        # Directives included in inversion setup

        # Run
        m_rec = self.inv.run(np.zeros(int(active_cells.sum())))

        # Convergence check
        phi_d = self.inv.dmis.phi_d
        chi2 = len(observed)
        if phi_d > 1.5 * chi2:
            raise InversionConvergenceError(f"phi_d={phi_d:.2f} > 1.5*chi2={1.5*chi2:.2f}")

        # Uncertainty: diagonal posterior variance approximation
        uncertainty = self._estimate_uncertainty_diag(sim, data_obj, reg, m_rec)

        # Interpolate to grid
        model_3d, unc_3d = self._interpolate_to_grid(m_rec, uncertainty, data, mask)

        metadata = {
            'converged': phi_d <= 1.5 * chi2,
            'iterations': self.inv.nIterations,
            'residuals': phi_d,
            'units': 'SI',
            'algorithm': 'simpeg_magnetics_tmi',
            'parameters': {
                'inclination': np.rad2deg(inclination),
                'declination': np.rad2deg(declination),
                'regularization': kwargs.get('regularization', {}),
                'background_sus': self.background_sus,
                'depth_weighting': kwargs.get('depth_weighting', True),
            },
        }

        results = InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)
        logger.info(f"Magnetic inversion completed: {results.model.shape}, converged={metadata['converged']}")
        return results

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """
        Fuse magnetic results with other modalities using inverse-variance weighting.

        Simple statistical fusion; advanced structural in JointInverter.

        Parameters
        ----------
        models : List[InversionResults]
            Including magnetic and others.
        **kwargs : dict, optional
            - 'weights': List[float] (default: equal)
            - 'method': 'inv_var' or 'weighted_avg'

        Returns
        -------
        np.ndarray
            Fused 3D susceptibility proxy.
        """
        method = kwargs.get('method', 'inv_var')
        weights = kwargs.get('weights', np.ones(len(models)) / len(models))

        fused = np.zeros_like(models[0].model)
        if method == 'weighted_avg':
            for i, m in enumerate(models):
                fused += weights[i] * m.model
        elif method == 'inv_var':
            for i, m in enumerate(models):
                w_i = 1.0 / (m.uncertainty**2 + 1e-10)
                fused += weights[i] * m.model * w_i
            fused /= np.sum([weights[i] * (1.0 / (m.uncertainty**2 + 1e-10)) for i, m in enumerate(models)], axis=0) + 1e-10
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Magnetic fusion: method={method}, shape={fused.shape}")
        return fused

    def _get_active_cells(self, data: ProcessedGrid) -> npt.NDArray[np.bool_]:
        """Define active cells below topography using active_from_xyz."""
        topo = data.ds.attrs.get('topography')
        if topo is None:
            # Default: below z=0
            actv = self.mesh.gridCC[:, 2] <= 0
            logger.info("No topography; active cells below z=0")
            return actv

        # Create surface points: lat, lon, elev
        lats, lons = np.meshgrid(data.ds['lat'].values, data.ds['lon'].values, indexing='ij')
        elevs = topo.ravel()  # assume topo is (n_lat, n_lon)
        # Transform to xyz
        x_surf, y_surf = transform_coordinates(lons.ravel(), lats.ravel())
        z_surf = elevs  # assume elev in meters
        topo_points = np.c_[x_surf, y_surf, z_surf]

        # Active below topo
        actv = active_from_xyz(self.mesh, topo_points, "top")
        logger.info(f"Active cells from topo: {actv.sum()} / {self.mesh.nC}")
        return actv

    def _get_locations(self, data: ProcessedGrid) -> npt.NDArray[np.float64]:
        """Observation locations."""
        lons, lats = np.meshgrid(data.ds['lon'].values, data.ds['lat'].values)
        lons_flat, lats_flat = lons.ravel(), lats.ravel()
        x, y = transform_coordinates(lons_flat, lats_flat)
        locations = np.column_stack([x, y, np.zeros(len(lons_flat))])
        return locations

    def _estimate_uncertainty_diag(
        self,
        sim: magnetics.simulation.Simulation3DIntegral,
        data_obj: spdata.Data,
        reg: regularization.BaseRegularization,
        m_rec: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Approximate posterior standard deviation from diagonal of (Jᵀ Wdᵀ Wd J + β R)⁻¹.
        Uses Simulation3DIntegral.getJtJdiag if available; falls back to 10% |m|.
        """
        try:
            # Weighting by inverse standard deviation
            Wd = 1.0 / np.maximum(data_obj.standard_deviation, 1e-12)
            jtjd = sim.getJtJdiag(Wd=Wd, m=m_rec)
            beta = float(self.inv.invProb.beta) if hasattr(self.inv, "invProb") else 1.0
            # Regularization diagonal (alpha_s term)
            reg_diag = getattr(reg, "alpha_s", 1.0) * np.ones_like(jtjd)
            diag_A = jtjd + beta * reg_diag
            var = 1.0 / np.maximum(diag_A, 1e-12)
            std = np.sqrt(var)
        except Exception:
            # Fallback: 10% of recovered susceptibility magnitude
            std = 0.10 * np.abs(m_rec) + 1e-12
        # Full array on mesh
        full_unc = np.zeros(self.mesh.nC)
        full_unc[reg.indActive] = std
        return full_unc

    def _interpolate_to_grid(self, model_flat: npt.NDArray[np.float64], unc_flat: npt.NDArray[np.float64],
                             data: ProcessedGrid, mask: Optional[npt.NDArray[np.bool_]] = None) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate model from active cells to data grid using griddata."""
        actv = self._get_active_cells(data)
        cc = self.mesh.gridCC[actv]
        model_active = model_flat
        unc_active = unc_flat[actv]
        # Data points in xyz
        lats, lons = np.meshgrid(data.ds['lat'].values, data.ds['lon'].values, indexing='ij')
        depths = data.ds['depth'].values
        lat_flat, lon_flat = lats.ravel(), lons.ravel()
        x_data, y_data = transform_coordinates(lon_flat, lat_flat)
        # For each depth level
        model_3d = np.zeros((len(data.ds['lat']), len(data.ds['lon']), len(depths)))
        unc_3d = np.zeros_like(model_3d)
        for i_d, d in enumerate(depths):
            z_data = np.full(len(lat_flat), -d)  # assume depth positive down
            points_data = np.column_stack([x_data, y_data, z_data])
            # Interpolate
            model_3d[:, :, i_d] = griddata(cc, model_active, points_data, method='linear', fill_value=0.0).reshape(len(data.ds['lat']), len(data.ds['lon']))
            unc_3d[:, :, i_d] = griddata(cc, unc_active, points_data, method='linear', fill_value=0.0).reshape(len(data.ds['lat']), len(data.ds['lon']))
        return model_3d, unc_3d