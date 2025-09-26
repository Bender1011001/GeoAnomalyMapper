"""Gravity inversion implementation for GAM using SimPEG."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import rasterio
from scipy.interpolate import RegularGridInterpolator

import simpeg
from simpeg import (
    data,
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
from simpeg.potential_fields import gravity

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

        # Validate input
        if 'data' not in data.ds:
            raise GAMError("ProcessedGrid missing 'data' variable")
        observed = data.ds['data'].values.flatten()  # mGal
        if np.any(np.isnan(observed)):
            warnings.warn("NaN values in observed data; masking")
            mask = ~np.isnan(observed)
            observed = observed[mask]

        config = get_config()

        # Generate mesh
        mesh_config = kwargs.get('mesh_config', {'type': 'tree', 'hmin': 10.0, 'hmax': 1000.0})
        mesh_gen = MeshGenerator()
        self.mesh = mesh_gen.create_mesh(data, **mesh_config)
        active_cells = self._get_active_cells(data)

        # Create survey and problem
        locations = self._get_locations(data)
        receiver_list = [gravity.receivers.Point(locations, components=kwargs.get('component', 'gz'))]
        source_field = gravity.sources.UniformBackgroundField(receiver_list=receiver_list)
        self.survey = simpeg.survey.Survey(source_field, receiver_list)

        terrain_correction = kwargs.get('terrain_correction', bool(config.get('modeling', {}).get('gravity', {}).get('dem_path')) or 'topography' in data.ds.attrs)
        if terrain_correction:
            correction = self._compute_terrain_correction(data)
            observed -= correction
            logger.info("Terrain correction subtracted from observed gravity data.")

        simulation_obj = gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=maps.IdentityMap(self.mesh),
            indActive=active_cells,
        )

        observed_si = observed * 1e-5  # Convert mGal to m/s² for SimPEG consistency

        # Observed data now terrain-corrected if applicable; forward uses flat datum

        # Data misfit
        dmisfit = data_misfit.L2DataMisfit(simulation=simulation_obj, data=d.Data(dobs=observed_si))
        # Relative error 5% or 0.1 mGal equivalent
        dmisfit.relative_error = np.ones(len(observed_si)) * 0.05
        dmisfit.absolute_error = np.ones(len(observed_si)) * (0.1 * 1e-5)

        # Regularization
        reg_params = kwargs.get('regularization', {'alpha_s': 1e-4, 'alpha_x': 1, 'alpha_y': 1, 'alpha_z': 1})
        reg_type = kwargs.get('regularization_type', config.get('modeling', {}).get('gravity', {}).get('regularization', 'l2'))
        mapping = maps.IdentityMap(self.mesh)
        if reg_type == 'l2':
            reg = regularization.WeightedLeastSquares(
                mesh=self.mesh,
                indActive=active_cells,
                mapping=mapping,
            )
        else:
            reg = regularization.Sparse(
                mesh=self.mesh,
                indActive=active_cells,
                mapping=mapping,
            )
        reg.alpha_s = reg_params['alpha_s']
        reg.alpha_x = reg_params['alpha_x']
        reg.alpha_y = reg_params['alpha_y']
        reg.alpha_z = reg_params['alpha_z']

        # Reference model
        m0 = kwargs.get('reference_model', np.zeros(self.mesh.nC))
        reg.mref = m0

        # Optimization
        opt = optimization.ProjectedGNCG(
            maxIter=kwargs.get('max_iterations', 20),
            lower=-np.inf,
            upper=np.inf,
            maxIterLS=20,
            maxIterCG=10,
            tolCG=1e-3,
        )

        # Inversion
        self.inv = inversion.BaseInversion(
            dmisfit, reg, opt, beta0=1e3, betaest=inversion.directives.BetaEstimateBySingularValue()
        )

        # Directives
        beta_max = 20
        self.inv.directiveList = [
            directives.BetaSchedule(coolEps_q=True, coolEps_beta=1.0, beta0_ratio=1.0),
            directives.BetaMaxIterative(beta0=beta_max, maxIterLS=20, maxIter=40),
            directives.TargetMisfit(),
            directives.UpdateSensitivityWeights(),
            directives.UpdatePreconditioner(),
        ]
        if reg_type == 'l1':
            self.inv.directiveList.append(
                directives.Update_IRLS(maxIRLSIter=10, beta_tol=1e-1)
            )

        # Run inversion
        m_rec = self.inv.run(m0)

        # Check convergence
        phi_d = self.inv.dmis.host
        chi2 = len(observed)
        if phi_d > 1.5 * chi2:  # Rough target
            raise InversionConvergenceError(f"phi_d={phi_d:.2f} > 1.5*chi2={1.5*chi2:.2f}")

        # Uncertainty (diagonal approx)
        uncertainty = self._estimate_uncertainty(m_rec, self.inv)

        # Convert to density (contrast * base)
        density_model = m_rec * self.base_density  # Assuming m_rec is contrast

        # Interpolate to data grid (3D)
        model_3d, unc_3d = self._interpolate_to_grid(density_model, uncertainty, data)

        metadata = {
            'converged': phi_d <= 1.5 * chi2,
            'iterations': self.inv.nIterations,
            'residuals': phi_d,
            'units': 'kg/m³',
            'algorithm': 'simpeg_gravity_tree',
            'parameters': {
                'base_density': self.base_density,
                'regularization': reg_params,
                'max_iterations': kwargs.get('max_iterations', 20),
                'terrain_correction': terrain_correction,
                'regularization_type': reg_type,
            },
        }

        results = InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)
        logger.info(f"Gravity inversion completed: {results.model.shape}, converged={metadata['converged']}")
        return results

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
        """Define active cells based on data extent and padding."""
        # Pad 10% for boundary effects
        pad = 0.1
        lons_min, lons_max = data.ds['lon'].min() - pad, data.ds['lon'].max() + pad
        lats_min, lats_max = data.ds['lat'].min() - pad, data.ds['lat'].max() + pad
        # Approximate center for transformation
        lon_center = (lons_min + lons_max) / 2
        lat_center = (lats_min + lats_max) / 2
        xmin, _ = transform_coordinates([lons_min], [lat_center])
        xmax, _ = transform_coordinates([lons_max], [lat_center])
        _, ymin = transform_coordinates([lon_center], [lats_min])
        _, ymax = transform_coordinates([lon_center], [lats_max])
        zmin, zmax = 0, 5000  # Default depth range in meters

        active = self.mesh.gridCC[:, 0] > xmin[0]
        active &= self.mesh.gridCC[:, 0] < xmax[0]
        active &= self.mesh.gridCC[:, 1] > ymin[0]
        active &= self.mesh.gridCC[:, 1] < ymax[0]
        active &= self.mesh.gridCC[:, 2] > -zmax
        active &= self.mesh.gridCC[:, 2] < -zmin
        return active

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

    def _interpolate_to_grid(self, model_flat: npt.NDArray[np.float64], unc_flat: npt.NDArray[np.float64],
                             data: ProcessedGrid) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate flat model to 3D grid."""
        # Get cell centers
        cc = self.mesh.gridCC
        # Create interpolator (3D)
        interp_model = RegularGridInterpolator((data.ds['lat'].values, data.ds['lon'].values, data.ds['depth'].values),
                                               model_flat.reshape(self.mesh.shape_cells), method='linear', bounds_error=False)
        interp_unc = RegularGridInterpolator((data.ds['lat'].values, data.ds['lon'].values, data.ds['depth'].values),
                                             unc_flat.reshape(self.mesh.shape_cells), method='linear', bounds_error=False)
        points = np.array(np.meshgrid(data.ds['lat'], data.ds['lon'], data.ds['depth'], indexing='ij')).T.reshape(-1, 3)
        model_3d = interp_model(points).reshape(len(data.ds['lat']), len(data.ds['lon']), len(data.ds['depth']))
        unc_3d = interp_unc(points).reshape(len(data.ds['lat']), len(data.ds['lon']), len(data.ds['depth']))
        return model_3d, unc_3d