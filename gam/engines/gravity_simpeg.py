"""Gravity inversion implementation for GAM using SimPEG."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio
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
from simpeg.potential_fields import gravity
from simpeg.potential_fields.utils import depth_weighting

from gam.core.utils import transform_coordinates, reverse_transform_coordinates
from gam.core.config import get_config

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.mesh import MeshGenerator  # Will implement later; assume available
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


class GravityInverter(Inverter):
    """
    SimPEG-based gravity inversion for density contrast estimation.

    This class implements gravity anomaly inversion to recover 3D density models
    using SimPEG's potential_fields.gravity module. Supports both linear (compact)
    and smooth inversions with adaptive TreeMesh for efficient resolution based
    on data density. Handles terrain corrections if topography data available in
    ProcessedGrid metadata. Parameterization uses integral chargeable volumes
    mapped to density (kg/m³). Includes regularization with smoothing and reference
    model constraints.

    Key features:
    - Adaptive meshing with refinement near surface/data locations
    - Forward modeling with terrain effects (if provided)
    - Tikhonov regularization (smallness + smoothness)
    - Convergence monitoring with beta cooling and targeting
    - Uncertainty estimation via approximate diagonal Hessian
    - Units: Input gravity in mGal, output density in kg/m³ (2000 base density assumed)

    Parameters
    ----------
    base_density : float, optional
        Reference crustal density (default: 2000 kg/m³).
    g : float, optional
        Gravitational constant scaling (default: 6.67e-11, but SimPEG normalizes).

    Attributes
    ----------
    base_density : float
        Reference density for contrasts.
    mesh : simpeg.mesh.TreeMesh, optional
        Generated mesh (cached for reuse).

    Methods
    -------
    Inherited from Inverter; see invert() and fuse() for specifics.

    Notes
    -----
    - **Mesh**: TreeMesh with hmin=10m near surface, coarsening to 1km at depth.
    - **Forward**: Point mass sources on active cells; terrain via SimPEG sources.
    - **Inversion**: Gauss-Newton with L2 misfit; directives for beta schedule.
    - **Uncertainty**: From model covariance (diagonal approximation).
    - **Performance**: Vectorized forward; suitable for regional scales (100x100x50 cells).
    - **Limitations**: Assumes vertical component (gz); extend for full tensor.
    - **Dependencies**: SimPEG >=0.19.0; requires scipy, numpy.
    - **Reproducibility**: Set random_seed in kwargs for directive randomization.
    - **Error Handling**: Raises InversionConvergenceError if phi_d > chi2 target.

    Examples
    --------
    >>> inverter = GravityInverter(base_density=2200)
    >>> results = inverter.invert(processed_grid, regularization={'alpha_s': 1e-3})
    >>> density_model = results.model  # Shape: (n_lat, n_lon, n_depth)
    """

    def __init__(self, base_density: float = 2000.0, g: float = 6.67e-11):
        self.base_density = base_density
        self.g = g
        self.mesh: Optional[TreeMesh] = None
        self.survey: Optional[simulation.Gravity] = None
        self.inv: Optional[inversion.BaseInversion] = None

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Perform gravity inversion to estimate 3D density contrasts.

        Sets up TreeMesh from data extent, creates survey from observation locations,
        parameterizes with IdentityMap or LogMap for positivity, runs inversion with
        smoothing regularization and reference model (zero by default). Supports
        terrain corrections if 'topography' in data attrs.

        Parameters
        ----------
        data : ProcessedGrid
            Gravity anomaly data (variable 'data' in mGal, coords lat/lon/depth).
        **kwargs : dict, optional
            - 'mesh_config': Dict for MeshGenerator (default: adaptive)
            - 'regularization': Dict {'alpha_s': 1e-4, 'alpha_x': 1, 'alpha_y': 1, 'alpha_z': 1}
            - 'reference_model': np.ndarray or None (default: zeros)
            - 'max_iterations': int (default: 20)
            - 'random_seed': int (default: None)
            - 'terrain_correction': bool (default: True if topo available)
            - 'component': str ('gz' default for vertical)

        Returns
        -------
        InversionResults
            With 'model' as density contrasts (kg/m³), 'uncertainty' from cov.

        Raises
        ------
        GAMError
            Invalid data or setup.
        InversionConvergenceError
            Non-convergence.
        """
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            simpeg.utils.rand.seed(random_seed)

        # Validate input and mask NaNs
        if "data" not in data.ds:
            raise GAMError("ProcessedGrid missing 'data' variable")
        obs_mgal_flat = data.ds["data"].values.reshape(-1)
        valid = ~np.isnan(obs_mgal_flat)
        if not np.any(valid):
            raise GAMError("No valid gravity observations (all NaN).")

        # Observation locations at surface (z=0)
        all_xyz = self._get_locations(data)
        xyz = all_xyz[valid, :]
        obs_mgal = obs_mgal_flat[valid].copy()

        # Create survey with selected component
        component = kwargs.get("component", "gz")
        rx = gravity.receivers.Point(xyz, components=(component,))
        src = gravity.sources.SourceField(receiver_list=[rx])
        self.survey = gravity.survey.Survey(source_field=src)

        # Terrain correction option
        config = get_config()
        terrain_correction = kwargs.get(
            "terrain_correction",
            bool(config.get("modeling", {}).get("gravity", {}).get("dem_path"))
            or "topography" in data.ds.attrs,
        )
        if terrain_correction:
            tc = self._compute_terrain_correction(data)
            if tc.shape[0] == self.survey.nD:
                obs_mgal = obs_mgal - tc
                logger.info(
                    f"Applied terrain correction: range [{tc.min():.2f}, {tc.max():.2f}] mGal"
                )
            else:
                logger.warning("Terrain correction size mismatch; skipping correction.")

        # Mesh and active cells
        mesh_cfg = kwargs.get("mesh_config", {"type": "tree", "hmin": 10.0, "hmax": 1000.0})
        mesh_gen = MeshGenerator()
        self.mesh = mesh_gen.create_mesh(data, **mesh_cfg)
        actv = self._get_active_cells(data)
        n_actv = int(actv.sum())
        if n_actv == 0:
            raise GAMError("No active cells (check mesh/topography).")

        # Simulation for density contrast
        rho_map = maps.IdentityMap(nP=n_actv)
        sim = gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=rho_map,
            indActive=actv,
            store_sensitivities="forward_only",
        )

        # Data object (SI units: m/s^2)
        dobs_si = obs_mgal * 1e-5
        data_obj = spdata.Data(survey=self.survey, dobs=dobs_si)
        # Define standard deviation as relative error plus noise floor
        rel_err = float(kwargs.get("relative_error", 0.05))
        floor_mgal = float(kwargs.get("noise_floor_mgal", 0.1))
        std_si = rel_err * np.abs(dobs_si) + (floor_mgal * 1e-5)
        data_obj.standard_deviation = std_si
        dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)

        # Regularization with depth weighting
        wr = depth_weighting(self.mesh, indActive=actv, exponent=2.0)
        reg_type = kwargs.get("reg_type", "l2").lower()
        if reg_type == "tv":
            reg = regularization.Sparse(
                mesh=self.mesh,
                indActive=actv,
                mapping=rho_map,
                cell_weights=wr,
                norms=[0, 1, 1, 1],
            )
        else:
            reg = regularization.Simple(
                mesh=self.mesh, indActive=actv, mapping=rho_map, cell_weights=wr
            )
        reg_params = kwargs.get(
            "regularization",
            {"alpha_s": 1e-4, "alpha_x": 1.0, "alpha_y": 1.0, "alpha_z": 1.0},
        )
        reg.alpha_s = float(reg_params.get("alpha_s", 1e-4))
        reg.alpha_x = float(reg_params.get("alpha_x", 1.0))
        reg.alpha_y = float(reg_params.get("alpha_y", 1.0))
        reg.alpha_z = float(reg_params.get("alpha_z", 1.0))

        # Reference model on active cells
        mref_full = kwargs.get("reference_model", None)
        if mref_full is None:
            mref = np.zeros(n_actv)
        else:
            if mref_full.shape[0] != self.mesh.nC:
                raise GAMError("reference_model must be full-length (mesh.nC).")
            mref = mref_full[actv]
        reg.mref = mref

        # Optimization
        opt = optimization.InexactGaussNewton(maxIter=int(kwargs.get("max_iterations", 20)))

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
        beta0_ratio = float(kwargs.get("beta0_ratio", 1.0))
        N = self.survey.nD
        target = float(kwargs.get("target_misfit", 0.5 * N))
        directives_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
            directives.UpdateSensitivityWeights(),
            directives.BetaSchedule(coolingFactor=float(kwargs.get("beta_cooling", 2.0))),
            directives.SaveOutputEveryIteration(save_txt=False),
            directives.TargetMisfit(target=target),
        ]
        if reg_type == "tv":
            directives_list.append(
                directives.Update_IRLS(
                    f_min_change=1e-2, maxIRLSIter=10, beta_tol=1e-1, coolEps=1.5
                )
            )

        self.inv = inversion.BaseInversion(invProb, directiveList=directives_list)

        # Run inversion starting from zero model
        m0 = np.zeros(n_actv)
        m_actv = self.inv.run(m0)

        # Convergence check using phi_d
        phi_d = float(dmis.phi_d)
        if not np.isfinite(phi_d) or phi_d > 1.2 * target:
            raise InversionConvergenceError(
                f"Did not hit target misfit: phi_d={phi_d:.3f} vs target={target:.3f}"
            )

        # Uncertainty approximation on active cells
        unc_actv = self._estimate_uncertainty_diag(sim, data_obj, reg, m_actv)

        # Interpolate to (lat,lon,depth) grid
        model_3d, unc_3d = self._interpolate_to_grid(m_actv, unc_actv, data, actv)

        metadata = {
            "converged": phi_d <= 1.2 * target,
            "phi_d": phi_d,
            "target": target,
            "units": "kg/m^3 (density contrast)",
            "algorithm": "SimPEG gravity (Simulation3DIntegral, TreeMesh)",
            "parameters": {
                "regularization": reg_params,
                "reg_type": reg_type,
                "relative_error": rel_err,
                "noise_floor_mgal": floor_mgal,
                "beta_cooling": float(kwargs.get("beta_cooling", 2.0)),
                "max_iterations": int(kwargs.get("max_iterations", 20)),
                "component": component,
                "terrain_correction": bool(terrain_correction),
                "n_obs": int(N),
                "n_active_cells": int(n_actv),
            },
        }

        logger.info(
            f"Gravity inversion OK: N={N}, phi_d={phi_d:.3f}, active={n_actv}, shape={model_3d.shape}"
        )
        return InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """
        Fuse gravity results with other modalities (basic weighted average).

        For joint inversion, uses simple inverse-variance weighting. Advanced
        cross-gradient in JointInverter.

        Parameters
        ----------
        models : List[InversionResults]
            List including this gravity result and others.
        **kwargs : dict, optional
            - 'weights': List[float] (default: equal)
            - 'method': 'weighted_avg' or 'inv_var'

        Returns
        -------
        np.ndarray
            Fused 3D array (anomaly strength proxy).
        """
        method = kwargs.get('method', 'weighted_avg')
        weights = kwargs.get('weights', np.ones(len(models)) / len(models))

        # Assume all models same shape; take gravity as base
        fused = np.zeros_like(models[0].model)
        if method == 'weighted_avg':
            for i, m in enumerate(models):
                fused += weights[i] * m.model
        elif method == 'inv_var':
            for i, m in enumerate(models):
                w_i = 1.0 / (m.uncertainty**2 + 1e-10)
                fused += weights[i] * m.model * w_i
            fused /= np.sum([weights[i] * w_i for i, m in enumerate(models)], axis=0) + 1e-10
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        logger.info(f"Gravity fusion completed: method={method}, shape={fused.shape}")
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
        """Extract observation locations from grid centers."""
        lons, lats = np.meshgrid(data.ds['lon'].values, data.ds['lat'].values)
        lons_flat, lats_flat = lons.ravel(), lats.ravel()
        x, y = transform_coordinates(lons_flat, lats_flat)
        locations = np.column_stack([x, y, np.zeros(len(lons_flat))])  # z=0
        return locations

    def _compute_terrain_correction(self, data: ProcessedGrid) -> npt.NDArray[np.float64]:
        """
        Prism-based terrain correction using SimPEG.

        Ingests a Digital Elevation Model (DEM) for the region via rasterio from
        config['modeling']['gravity']['dem_path'] or uses existing data.attrs['topography'].
        Uses prism-based calculation to compute the gravitational effect of topography
        (cells above z=0 datum, below surface) at observation locations. Returns values
        to subtract from observed gravity data for residual anomaly inversion on flat datum.

        Leverages SimPEG's Simulation3DIntegral for efficient prism integration.
        Assumes DEM in GeoTIFF format with CRS matching data (EPSG:4326 lat/lon).
        Interpolates DEM to mesh cell centers via coordinate transform.

        References:
        - SimPEG documentation: potential_fields.gravity.Simulation3DIntegral
        - Prism gravity formula: Nagy et al. (2000), "The gravitational potential and its derivatives..."
        - Terrain correction in geophysics: standard Bouguer correction extension.

        Returns
        -------
        np.ndarray
            Terrain correction array (mGal, length matches observations).
        """
        config = get_config()
        dem_path = config.get('modeling', {}).get('gravity', {}).get('dem_path')

        topo = data.ds.attrs.get('topography')
        if topo is None and dem_path:
            try:
                with rasterio.open(dem_path) as src:
                    elev = src.read(1)
                    # Assume 2D array matching data grid shape (n_lat, n_lon)
                    if elev.ndim == 2 and elev.shape == (len(data.ds['lat']), len(data.ds['lon'])):
                        topo = elev
                    else:
                        logger.warning(f"DEM shape {elev.shape} does not match data grid {(len(data.ds['lat']), len(data.ds['lon']))}; skipping load.")
                        topo = None
            except ImportError:
                logger.warning("rasterio not installed; cannot load DEM. Install with 'pip install rasterio'.")
                topo = None
            except Exception as e:
                logger.warning(f"Failed to load DEM from {dem_path}: {e}")
                topo = None

        if topo is None:
            logger.warning("No valid topography data; returning zero correction.")
            n_obs = len(self.survey.receiver_list[0].locations)
            return np.zeros(n_obs)

        # Determine UTM zone from data mean longitude
        mean_lon = data.ds['lon'].mean()
        utm_zone = int((mean_lon + 180) / 6) + 31
        if mean_lon < 0:
            utm_zone -= 1

        # Transform mesh cell centers (x, y) back to (lon, lat)
        cell_xy = self.mesh.gridCC[:, :2]  # (nC, 2) x, y in UTM
        lons_cell, lats_cell = reverse_transform_coordinates(cell_xy[:, 0], cell_xy[:, 1], source_crs=f"EPSG:326{utm_zone}")

        # Create interpolator for topo on lat/lon grid (ij indexing: lat rows, lon cols)
        interp_topo = RegularGridInterpolator(
            (data.ds['lat'].values, data.ds['lon'].values),
            topo,
            method='linear',
            bounds_error=False,
            fill_value=0.0  # Default to sea level
        )

        # Interpolate heights to cell centers
        h = interp_topo((lats_cell, lons_cell))

        # Handle any NaN (extrapolate to 0)
        h = np.nan_to_num(h, nan=0.0)

        # Identify terrain cells: below topo surface AND above datum (z > 0)
        try:
            from discretize.utils import surface2ind_topo
            ind_below_topo = surface2ind_topo(self.mesh, h=h, actv=None)
        except ImportError:
            logger.warning("discretize.utils.surface2ind_topo not available; skipping.")
            n_obs = len(self.survey.receiver_list[0].locations)
            return np.zeros(n_obs)

        ind_terrain = ind_below_topo & (self.mesh.gridCC[:, 2] > 0)
        n_terrain = np.sum(ind_terrain)

        if n_terrain == 0:
            logger.info("No terrain cells above datum; zero correction.")
            n_obs = len(self.survey.receiver_list[0].locations)
            return np.zeros(n_obs)

        # Terrain simulation: prisms with base density
        sim_terrain = gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=maps.IdentityMap(self.mesh),
            indActive=ind_terrain,
        )

        # Uniform density model for terrain prisms
        m_terrain = np.full(n_terrain, self.base_density)

        # Compute vertical gravity effect (gz)
        correction = sim_terrain.dpred(m_terrain)

        # Scale to mGal (SimPEG outputs in m/s²)
        correction *= 1e5

        logger.info(f"Terrain correction computed: {n_terrain} prisms, range [{correction.min():.2f}, {correction.max():.2f}] mGal")
        return correction

    def _estimate_uncertainty(self, m_rec: npt.NDArray[np.float64], inv: inversion.BaseInversion) -> npt.NDArray[np.float64]:
        """Approximate uncertainty from inversion statistics."""
        # Diagonal of model covariance (J^T J + reg)^{-1} * sigma_d^2
        sigma_d = 1.0  # Data std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = inv.reg.hessian(m_rec, req_grad=False)
            unc = np.sqrt(np.diag(np.linalg.pinv(H)) * sigma_d**2)
        unc = unc[inv.reg.indActive]
        # Pad to full mesh
        full_unc = np.zeros(self.mesh.nC)
        full_unc[inv.reg.indActive] = unc
        return full_unc

    def _estimate_uncertainty_diag(
        self,
        sim: gravity.simulation.Simulation3DIntegral,
        d: spdata.Data,
        reg: regularization.BaseRegularization,
        m_actv: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Diagonal posterior variance approximation: diag((Jᵀ Wdᵀ Wd J + β R)⁻¹).
        Uses getJtJdiag if available; else returns conservative 10% of |m|.
        """
        try:
            Wd = 1.0 / np.maximum(d.standard_deviation, 1e-12)
            jtjd = sim.getJtJdiag(Wd=Wd, m=m_actv)
            beta = float(self.inv.invProb.beta) if hasattr(self.inv, "invProb") else 1.0
            reg_diag = getattr(reg, "alpha_s", 1.0) * np.ones_like(jtjd)
            diag_A = jtjd + beta * reg_diag
            var = 1.0 / np.maximum(diag_A, 1e-12)
            return np.sqrt(var)
        except Exception as e:
            logger.warning(f"Uncertainty diag fallback (reason: {e})")
            return 0.10 * np.abs(m_actv) + 1e-9

    def _interpolate_to_grid(
        self,
        m_actv: npt.NDArray[np.float64],
        u_actv: npt.NDArray[np.float64],
        grid: ProcessedGrid,
        actv: npt.NDArray[np.bool_],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate active-cell model to (lat, lon, depth)."""
        assert self.mesh is not None
        cc = self.mesh.gridCC[actv]  # (n_actv, 3)
        lat = grid.ds["lat"].values
        lon = grid.ds["lon"].values
        depth = grid.ds["depth"].values
        LON, LAT = np.meshgrid(lon, lat, indexing="xy")
        x_data, y_data = transform_coordinates(LON.ravel(), LAT.ravel())
        model_3d = np.zeros((len(lat), len(lon), len(depth)), dtype=float)
        unc_3d = np.zeros_like(model_3d)
        for k, d in enumerate(depth):
            z_data = -float(d) * np.ones_like(x_data)
            points = np.column_stack([x_data, y_data, z_data])
            m_slice = griddata(cc, m_actv, points, method="linear", fill_value=0.0)
            u_slice = griddata(cc, u_actv, points, method="linear", fill_value=0.0)
            model_3d[:, :, k] = m_slice.reshape(len(lat), len(lon))
            unc_3d[:, :, k] = u_slice.reshape(len(lat), len(lon))
        return model_3d, unc_3d