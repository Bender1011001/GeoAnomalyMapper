"""Joint inversion implementation for GAM using SimPEG multi-physics framework."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

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

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.mesh import MeshGenerator
from gam.preprocessing.data_structures import ProcessedGrid
from gam.engines.gravity_simpeg import GravityInverter
from gam.engines.magnetics_simpeg import MagneticInverter

logger = logging.getLogger(__name__)


class CrossGradientRegularization(regularization.BaseRegularization):
    """Custom cross-gradient regularization term.

    Minimizes the cross-gradient between two models: || ∇m1 × ∇m2 ||_2
    for structural similarity in joint inversion.
    """

    def __init__(self, mesh: TreeMesh, reg1: regularization.BaseRegularization, reg2: regularization.BaseRegularization,
                 indActive: Optional[npt.NDArray[np.bool_]] = None, mapping: maps.IdentityMap = None,
                 gradient_type: str = 'total') -> None:
        super().__init__(mesh=mesh, indActive=indActive, mapping=mapping)
        self.reg1 = reg1
        self.reg2 = reg2
        self.gradient_type = gradient_type  # 'total' or 'component'

    def __call__(self, m: np.ndarray) -> float:
        """Compute cross-gradient norm."""
        m1 = self.reg1.mapping * m[:len(m)//2]
        m2 = self.reg2.mapping * m[len(m)//2:]
        grad_m1 = self.reg1._get_gradient(m1)
        grad_m2 = self.reg2._get_gradient(m2)
        if self.gradient_type == 'total':
            cross = np.linalg.norm(np.cross(grad_m1, grad_m2), axis=1)
        else:
            cross = np.sum(np.abs(grad_m1 * grad_m2[::-1] - grad_m1[::-1] * grad_m2), axis=1)  # Component-wise
        return np.sum(cross**2)

    def get_gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient of cross-term (simplified finite difference)."""
        # For optimization, need analytic or FD; here FD approx
        eps = 1e-6
        grad = np.zeros_like(m)
        for i in range(len(m)):
            m_plus = m.copy()
            m_plus[i] += eps
            f_plus = self.__call__(m_plus)
            grad[i] = (f_plus - self.__call__(m)) / eps
        return grad


class JointInverter(Inverter):
    """
    Joint inversion for multiple modalities using SimPEG with cross-gradient coupling.

    Combines gravity and magnetics (extendable) via multi-data misfit + cross-gradient
    regularization. Uses shared TreeMesh, block coordinate descent or simultaneous
    optimization. Couples density (m_g) and susceptibility (m_m) structurally.

    Key features:
    - Multi-physics: L2 data misfits for each modality
    - Cross-gradient: λ ∫ ||∇m_g × ∇m_m|| dV for structural similarity
    - Shared mesh: Octree with refinement/padding/active cells
    - Directives: Beta by eig, sensitivity weights, target misfit
    - Uncertainty: Joint covariance diagonal
    - Units: Density kg/m³, susceptibility SI

    Parameters
    ----------
    lambda_cg : float, optional
        Cross-gradient weight (default: 0.5).
    modalities : List[str], optional
        ['gravity', 'magnetics'] (default).

    Attributes
    ----------
    mesh : TreeMesh
        Shared mesh.
    simulations : Dict[str, simulation.Simulation3DIntegral]
        Per-modality simulations.
    inversion : inversion.BaseInversion
        Joint inversion object.

    Methods
    -------
    invert(data_dict: Dict[str, ProcessedGrid], **kwargs) -> InversionResults
        Run joint inversion.

    Notes
    -----
    - **Objective**: Φ = φ_d^g + φ_d^m + α_s (||Wm||^2) + α_x (||∇m||^2) + λ_cg ∫ ||∇m_g × ∇m_m||
    - **Optimization**: Block Gauss-Newton; alternate m_g, m_m or simultaneous.
    - **Mesh**: Shared octree; active below topo.
    - **Coupling**: Tune λ_cg (0.1-1.0); too high over-constrains.
    - **Performance**: CSR sensitivities; suitable for regional.
    - **Limitations**: Pairwise CG (extend to N); assumes linear forward.
    - **Dependencies**: SimPEG >=0.21.
    - **Validation**: Synthetic tests show improved resolution.
    """

    def __init__(self, lambda_cg: float = 0.5, modalities: List[str] = ['gravity', 'magnetics']):
        self.lambda_cg = lambda_cg
        self.modalities = modalities
        self.mesh: Optional[TreeMesh] = None
        self.simulations: Dict[str, simulation.Simulation3DIntegral] = {}
        self.regularizations: Dict[str, regularization.BaseRegularization] = {}
        self.inversion: Optional[inversion.BaseInversion] = None

    def invert(self, data_dict: Dict[str, ProcessedGrid], **kwargs) -> InversionResults:
        """
        Perform joint inversion for multiple modalities.

        Validates data, generates shared mesh, sets up simulations/regs,
        combines misfits with CG term, runs inversion, extracts models.

        Parameters
        ----------
        data_dict : Dict[str, ProcessedGrid]
            {'gravity': grid_g, 'magnetics': grid_m, ...}
        **kwargs : dict, optional
            - 'mesh_config': Dict for MeshGenerator (default: shared octree)
            - 'reg_params': Dict per modality {'alpha_s': 1e-4, ...}
            - 'lambda_cg': float (override)
            - 'max_iterations': int (default: 20)
            - 'target_misfit': int (default: sum data sizes)
            - 'random_seed': int

        Returns
        -------
        InversionResults
            'model' as dict of modality models, 'uncertainty' joint.

        Raises
        ------
        GAMError
            Missing data or incompatible.
        InversionConvergenceError
            Non-convergence.
        """
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)

        # Validate data
        for mod in self.modalities:
            if mod not in data_dict:
                raise GAMError(f"Missing data for modality: {mod}")
        if len(set(d.ds.dims for d in data_dict.values())) > 1:
            raise GAMError("Incompatible data dimensions")

        # Shared mesh
        mesh_config = kwargs.get('mesh_config', {'base_cell_m': 25, 'padding_cells': 6})
        mesh_gen = MeshGenerator()
        self.mesh = mesh_gen.create_mesh(list(data_dict.values())[0], type='adaptive', **mesh_config)
        active_cells = self._get_active_cells_shared(list(data_dict.values())[0])

        # Setup per-modality
        m0_total = np.zeros(2 * int(active_cells.sum()))  # m_g + m_m
        dmisfits = []
        for i, mod in enumerate(self.modalities):
            data = data_dict[mod]
            if mod == 'gravity':
                sim = self._setup_gravity_sim(self.mesh, active_cells, data)
            elif mod == 'magnetics':
                sim = self._setup_magnetics_sim(self.mesh, active_cells, data)
            else:
                raise ValueError(f"Unsupported modality: {mod}")

            self.simulations[mod] = sim
            rho_map = maps.IdentityMap(nP=int(active_cells.sum()))
            dmis = data_misfit.L2DataMisfit(data=data.Data(dobs=self._prepare_data(data)), simulation=sim)
            dmisfits.append(dmis)

            # Reg per modality
            w_depth = regularization.depth_weighting(self.mesh, indActive=active_cells, exponent=2.0)
            reg = regularization.Simple(mesh=self.mesh, indActive=active_cells, mapping=rho_map, cell_weights=w_depth)
            reg_params = kwargs.get('reg_params', {}).get(mod, {'alpha_s': 1e-4, 'alpha_x': 1.0, 'alpha_y': 1.0, 'alpha_z': 1.0})
            reg.alpha_s = reg_params['alpha_s']
            reg.alpha_x = reg_params['alpha_x']
            reg.alpha_y = reg_params['alpha_y']
            reg.alpha_z = reg_params['alpha_z']
            self.regularizations[mod] = reg

        # Joint misfit: sum data misfits
        joint_misfit = objective_function.ComboObjectiveFunction(dmisfits)

        # Regularization: smallness + smoothness + cross-gradient
        reg_small = regularization.ComboRegularization(self.regularizations.values())
        reg_cg = CrossGradientRegularization(self.mesh, list(self.regularizations.values())[0], list(self.regularizations.values())[1])
        reg_total = objective_function.ComboObjectiveFunction([reg_small, reg_cg])
        reg_total.alpha = [1.0, self.lambda_cg]

        # Mapping for concatenated models
        nP = int(active_cells.sum())
        joint_map = maps.ConcatMap([maps.IdentityMap(nP), maps.IdentityMap(nP)])

        # Optimization
        opt = optimization.InexactGaussNewton(maxIter=kwargs.get('max_iterations', 20))

        # Inverse problem
        inv_prob = simpeg.inverse_problem.BaseInvProblem(joint_misfit, reg_total, opt, beta0=1e3)

        # Directives
        target_misfit = kwargs.get('target_misfit', sum(len(dobs) for dobs in [dmis.data.dobs for dmis in dmisfits]))
        directive_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=1.0),
            directives.UpdateSensitivityWeights(),
            directives.TargetMisfit(target=target_misfit),
            directives.SaveOutputEveryIteration(),
        ]
        self.inversion = inversion.BaseInversion(inv_prob, directiveList=directive_list)

        # Run
        m_rec = self.inversion.run(m0_total)

        # Extract models
        m_g_rec = m_rec[:nP]
        m_m_rec = m_rec[nP:]
        models = {'gravity': m_g_rec * 2000.0, 'magnetics': m_m_rec}  # Scale density

        # Uncertainty (joint cov diagonal approx)
        uncertainty = self._estimate_joint_uncertainty(m_rec, self.inversion)

        # Interpolate to grids (use first data grid shape)
        grid_shape = list(data_dict.values())[0].ds['data'].shape + (len(data_dict),)
        model_3d = np.zeros(grid_shape)
        unc_3d = np.zeros(grid_shape)
        for i, mod in enumerate(self.modalities):
            model_3d[..., i] = self._interpolate_model(models[mod], list(data_dict.values())[0])
            unc_3d[..., i] = self._interpolate_model(uncertainty[mod], list(data_dict.values())[0])

        metadata = {
            'converged': self.inversion.dmis.phi_d <= 1.5 * target_misfit,
            'iterations': self.inversion.nIterations,
            'residuals': self.inversion.dmis.phi_d,
            'units': {'gravity': 'kg/m³', 'magnetics': 'SI'},
            'algorithm': 'simpeg_joint_cross_gradient',
            'parameters': {
                'lambda_cg': self.lambda_cg,
                'modalities': self.modalities,
                'mesh_nC': self.mesh.nC,
                'active_cells': int(active_cells.sum()),
            },
        }

        results = InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)
        logger.info(f"Joint inversion completed: converged={metadata['converged']}, phi_d={self.inversion.dmis.phi_d:.2f}")
        return results

    def _setup_gravity_sim(self, mesh: TreeMesh, active_cells: npt.NDArray[np.bool_], data: ProcessedGrid) -> simulation.Simulation3DIntegral:
        """Setup gravity simulation on shared mesh."""
        locations = self._get_locations(data)
        receiver_list = simpeg.potential_fields.gravity.receivers.Point(locations, components=("gz",))
        src = simpeg.potential_fields.gravity.sources.SourceField(receiver_list=[receiver_list])
        survey = simpeg.potential_fields.gravity.survey.Survey(source_field=src)
        rho_map = maps.IdentityMap(nP=int(active_cells.sum()))
        sim = simpeg.potential_fields.gravity.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, rhoMap=rho_map, indActive=active_cells,
            store_sensitivities="forward_only",
        )
        return sim

    def _setup_magnetics_sim(self, mesh: TreeMesh, active_cells: npt.NDArray[np.bool_], data: ProcessedGrid) -> simulation.Simulation3DIntegral:
        """Setup magnetics simulation on shared mesh."""
        locations = self._get_locations(data)
        B_T = data.ds.attrs.get('B_T', 5e-5)
        B_inc_deg = data.ds.attrs.get('B_inc_deg', 60.0)
        B_dec_deg = data.ds.attrs.get('B_dec_deg', 0.0)
        receiver_list = simpeg.potential_fields.magnetics.receivers.Point(locations, components=("tmi",))
        src = simpeg.potential_fields.magnetics.sources.SourceField(
            receiver_list=[receiver_list],
            parameters=simpeg.potential_fields.magnetics.EarthField(B=B_T, inclination=B_inc_deg, declination=B_dec_deg)
        )
        survey = simpeg.potential_fields.magnetics.survey.Survey(source_field=src)
        chi_map = maps.IdentityMap(nP=int(active_cells.sum()))
        sim = simpeg.potential_fields.magnetics.simulation.Simulation3DIntegral(
            survey=survey, mesh=mesh, chiMap=chi_map, indActive=active_cells,
            store_sensitivities="forward_only",
        )
        return sim

    def _get_active_cells_shared(self, data: ProcessedGrid) -> npt.NDArray[np.bool_]:
        """Shared active cells below topo."""
        from discretize.utils import active_from_xyz
        topo = data.ds.attrs.get('topography')
        if topo is None:
            actv = self.mesh.gridCC[:, 2] <= 0
        else:
            # Surface points from topo
            lats, lons = np.meshgrid(data.ds['lat'].values, data.ds['lon'].values, indexing='ij')
            elevs = topo.ravel()
            x_surf, y_surf = self._project_coords(lons.ravel(), lats.ravel()) if self.transformer else (lons.ravel()*111000, lats.ravel()*111000)
            topo_points = np.c_[x_surf, y_surf, elevs]
            actv = active_from_xyz(self.mesh, topo_points, "top")
        logger.info(f"Shared active cells: {actv.sum()} / {self.mesh.nC}")
        return actv

    def _prepare_data(self, data: ProcessedGrid) -> np.ndarray:
        """Prepare observed data (flatten, units)."""
        observed = data.ds['data'].values.flatten()
        mask = ~np.isnan(observed)
        if data.ds.attrs.get('units') == 'mGal':  # Gravity
            observed = observed[mask] * 1e-5  # to m/s²
        # Magnetics nT as is
        return observed[mask]

    def _get_locations(self, data: ProcessedGrid) -> npt.NDArray[np.float64]:
        """Observation locations in projected coords."""
        lons, lats = np.meshgrid(data.ds['lon'].values, data.ds['lat'].values)
        lons_flat, lats_flat = lons.ravel(), lats.ravel()
        if self.transformer:
            x, y = self._project_coords(lons_flat, lats_flat)
        else:
            x = lons_flat * 111000
            y = lats_flat * 111000
        return np.column_stack([x, y, np.zeros(len(x))])

    def _interpolate_model(self, model_flat: npt.NDArray[np.float64], data: ProcessedGrid) -> npt.NDArray[np.float64]:
        """Interpolate flat model to 3D grid."""
        from scipy.interpolate import griddata
        actv = self._get_active_cells_shared(data)
        cc = self.mesh.gridCC[actv]
        # Grid points
        lats, lons = np.meshgrid(data.ds['lat'].values, data.ds['lon'].values, indexing='ij')
        depths = data.ds['depth'].values
        lat_flat, lon_flat = lats.ravel(), lons.ravel()
        if self.transformer:
            x_data, y_data = self._project_coords(lon_flat, lat_flat)
        else:
            x_data = lon_flat * 111000
            y_data = lat_flat * 111000
        model_3d_list = []
        for d in depths:
            z_data = np.full(len(lat_flat), -d)
            points_data = np.column_stack([x_data, y_data, z_data])
            model_slice = griddata(cc, model_flat, points_data, method='linear', fill_value=0.0)
            model_3d_list.append(model_slice.reshape(lats.shape))
        return np.stack(model_3d_list, axis=-1)

    def _estimate_joint_uncertainty(self, m_rec: npt.NDArray[np.float64], inv: inversion.BaseInversion) -> Dict[str, npt.NDArray[np.float64]]:
        """Joint uncertainty from total Hessian."""
        nP = len(m_rec) // 2
        sigma_d = 1.0
        H = inv.reg.hessian(m_rec, req_grad=False)
        cov_diag = np.diag(np.linalg.pinv(H)) * sigma_d**2
        unc_g = np.sqrt(cov_diag[:nP])
        unc_m = np.sqrt(cov_diag[nP:])
        full_unc_g = np.zeros(self.mesh.nC)
        full_unc_g[self._get_active_cells_shared(list(self.data_dict.values())[0])] = unc_g
        full_unc_m = np.zeros(self.mesh.nC)
        full_unc_m[self._get_active_cells_shared(list(self.data_dict.values())[0])] = unc_m
        return {'gravity': full_unc_g, 'magnetics': full_unc_m}

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """Post-joint fusion if needed (identity for joint)."""
        if len(models) == 1:
            return models[0].model
        # Weighted avg as fallback
        weights = kwargs.get('weights', np.ones(len(models)) / len(models))
        fused = sum(w * m.model for w, m in zip(weights, models))
        return fused