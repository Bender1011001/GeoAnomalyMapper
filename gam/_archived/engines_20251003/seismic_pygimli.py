"""Seismic inversion implementation for GAM using PyGIMLi."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata

import pygimli as pg
from pygimli.physics import traveltime as tt  # canonical alias

from gam.core.exceptions import GAMError, InversionConvergenceError
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.preprocessing.data_structures import ProcessedGrid

logger = logging.getLogger(__name__)


class SeismicInverter(Inverter):
    """
    PyGIMLi-based first-arrival traveltime tomography for P-wave velocity.

    - Input travel times must be in seconds.
    - Returns 3D velocity [m/s] interpolated onto the ProcessedGrid (lat, lon, depth).
    - Internally uses TravelTimeManager with Dijkstra/eikonal solver.

    Parameters
    ----------
    v_min : float
        Lower clip for velocity [m/s].
    v_max : float
        Upper clip for velocity [m/s].
    """

    def __init__(self, v_min: float = 1500.0, v_max: float = 6000.0):
        self.v_min = v_min
        self.v_max = v_max
        self.mesh: Optional[pg.Mesh] = None

    def invert(self, data: ProcessedGrid, **kwargs) -> InversionResults:
        """
        Invert first-arrival traveltimes for velocity.

        Parameters (kwargs)
        -------------------
        sources : (N,2|3) array (meters)  required
        receivers : (N,2|3) array (meters) required
        times_ms : bool  default False; if True, inputs are milliseconds and will be converted to seconds
        t_error_abs : float  absolute pick error in seconds (default 0.005 = 5 ms)
        t_error_rel : float  relative error fraction (default 0.0)
        secNodes : int  default 2  (# of secondary nodes for forward mesh)
        paraDX : float  default 20.0
        paraMaxCellSize : float  default 50.0
        maxIter : int  default 10
        lam : float  default 20.0  (damping)
        resolution_test : bool  default False
        verbose : bool  default False

        Returns
        -------
        InversionResults
            model: (lat, lon, depth) velocity [m/s]
            uncertainty: same shape (rough posterior predictive)
            metadata: dict
        """
        # ---- Extract observed times ----
        if "data" not in data.ds:
            raise GAMError("ProcessedGrid missing ds['data'] with traveltimes")

        tt_obs = data.ds["data"].values.reshape(-1)  # user supplies per pair
        mask = ~np.isnan(tt_obs)
        if not np.any(mask):
            raise GAMError("No valid traveltimes in ProcessedGrid")

        # Optional ms->s
        if kwargs.get("times_ms", False):
            tt_obs = tt_obs * 1e-3

        # ---- Geometry ----
        sources = kwargs.get("sources", None)
        receivers = kwargs.get("receivers", None)
        if sources is None or receivers is None:
            raise GAMError("kwargs['sources'] and kwargs['receivers'] are required")
        if len(sources) != len(receivers) or len(tt_obs[mask]) != len(sources):
            raise GAMError("Length mismatch: times, sources, receivers")

        S = np.asarray(sources, dtype=float)
        G = np.asarray(receivers, dtype=float)
        if S.shape[1] not in (2, 3) or G.shape[1] not in (2, 3):
            raise GAMError("sources/receivers must have shape (N,2) or (N,3)")

        # ---- Build DataContainerTT with unique sensors ----
        # dedupe sensors with tolerance (1e-6 m)
        all_pts = np.vstack([S, G])
        key = np.round(all_pts / 1e-6).astype(np.int64)
        _, unique_idx, inv_idx = np.unique(key, axis=0, return_index=True, return_inverse=True)
        sensors = all_pts[unique_idx]  # (M,dim)

        s_idx = inv_idx[: len(S)]               # indices into 'sensors'
        g_idx = inv_idx[len(S):]                # indices into 'sensors'

        data_tt = tt.DataContainerTT()
        data_tt.registerSensorIndex("s")
        data_tt.registerSensorIndex("g")
        data_tt.setSensors(sensors)             # (M,2|3)
        data_tt.resize(len(s_idx))
        data_tt.set("s", s_idx.tolist())
        data_tt.set("g", g_idx.tolist())
        data_tt.set("t", tt_obs[mask].tolist())

        # errors: abs + rel
        t_abs = float(kwargs.get("t_error_abs", 0.005))  # 5 ms default
        t_rel = float(kwargs.get("t_error_rel", 0.0))
        err_vec = np.abs(tt_obs[mask]) * t_rel + t_abs
        data_tt.set("err", err_vec.tolist())
        data_tt.removeInvalid()

        # ---- Invert with TravelTimeManager ----
        para_dx = float(kwargs.get("paraDX", 20.0))
        para_max = float(kwargs.get("paraMaxCellSize", 50.0))
        sec_nodes = int(kwargs.get("secNodes", 2))
        max_iter = int(kwargs.get("maxIter", 10))
        lam = float(kwargs.get("lam", 20.0))
        verbose = bool(kwargs.get("verbose", False))

        mgr = tt.TravelTimeManager(data_tt)
        vest = mgr.invert(
            secNodes=sec_nodes,
            paraDX=para_dx,
            paraMaxCellSize=para_max,
            maxIter=max_iter,
            lam=lam,
            verbose=verbose,
        )  # returns velocity [m/s] on inversion mesh  :contentReference[oaicite:2]{index=2}

        # Guard rails
        vest = np.clip(np.asarray(vest, dtype=float), self.v_min, self.v_max)
        self.mesh = mgr.mesh  # inversion mesh created by manager  :contentReference[oaicite:3]{index=3}

        # χ² from the manager
        try:
            chi2 = float(mgr.inv.inv.chi2())
        except Exception:
            chi2 = np.nan

        # ---- Optional resolution tests ----
        res_meta: Dict[str, Any] = {"checkerboard_mean": None, "spike_bias": None}
        if kwargs.get("resolution_test", False):
            # Checkerboard slowness model
            slowness_cb = 1.0 / self._checkerboard_velocity(self.mesh, v_bg=2000.0, v_high=4000.0, cell_size=100.0)
            syn_cb = tt.simulate(mesh=self.mesh, scheme=data_tt, slowness=slowness_cb, returnArray=False)
            mgr_cb = tt.TravelTimeManager(syn_cb)
            vrec_cb = mgr_cb.invert(secNodes=sec_nodes, paraDX=para_dx, paraMaxCellSize=para_max,
                                    maxIter=max_iter, lam=lam, verbose=False)
            rec_map = np.abs(np.asarray(vrec_cb) - 2000.0) / 2000.0
            res_meta["checkerboard_mean"] = float(np.nanmean(rec_map))

            # Spike anomaly
            spike_center = np.array([np.mean(pg.x(self.mesh.cellCenters())),
                                     np.mean(pg.y(self.mesh.cellCenters())),
                                     np.mean(pg.z(self.mesh.cellCenters()))])
            slowness_spike = 1.0 / self._spike_velocity(self.mesh, v_bg=2000.0, spike_loc=spike_center, v_spike=5000.0)
            syn_sp = tt.simulate(mesh=self.mesh, scheme=data_tt, slowness=slowness_spike, returnArray=False)
            mgr_sp = tt.TravelTimeManager(syn_sp)
            vrec_sp = np.asarray(mgr_sp.invert(secNodes=sec_nodes, paraDX=para_dx, paraMaxCellSize=para_max,
                                               maxIter=max_iter, lam=lam, verbose=False))
            res_meta["spike_bias"] = float(np.nanmean((vrec_sp - 2000.0) / 2000.0))

        # ---- Uncertainty (cheap posterior predictive) ----
        unc_flat = self._posterior_predictive(vest, chi2, dof=max(1, data_tt.size() - self.mesh.cellCount()))

        # ---- Interpolate onto ProcessedGrid (lat, lon, depth) ----
        model_3d, unc_3d = self._interpolate_to_grid(vest, unc_flat, data)

        metadata = {
            "converged": (not np.isnan(chi2)) and (chi2 <= 1.2),
            "chi2": chi2,
            "units": "m/s",
            "algorithm": "pygimli.TravelTimeManager (Dijkstra)",
            "parameters": {
                "secNodes": sec_nodes,
                "paraDX": para_dx,
                "paraMaxCellSize": para_max,
                "maxIter": max_iter,
                "lam": lam,
                "n_pairs": int(data_tt.size()),
                "n_sensors": int(data_tt.sensorCount()),
            },
            "resolution": res_meta,
        }

        logger.info(
            f"Seismic inversion: model cells={self.mesh.cellCount()}, "
            f"pairs={data_tt.size()}, chi2={chi2:.3f if not np.isnan(chi2) else np.nan}"
        )
        return InversionResults(model=model_3d, uncertainty=unc_3d, metadata=metadata)

    def fuse(self, models: List[InversionResults], **kwargs) -> npt.NDArray[np.float64]:
        """Weighted average fusion of 3D velocity proxies."""
        w = np.asarray(kwargs.get("weights", np.ones(len(models)) / len(models)))
        w = w / w.sum()
        return sum(wi * mi.model for wi, mi in zip(w, models))

    # ----------------- helpers -----------------

    def _posterior_predictive(self, velocity: npt.NDArray[np.float64],
                              chi2: float, dof: int) -> npt.NDArray[np.float64]:
        """Heuristic σ_v ~ 10% * sqrt(chi2/dof)."""
        scale = np.sqrt(max(chi2, 1e-12) / max(dof, 1.0)) if np.isfinite(chi2) else 1.0
        return np.asarray(velocity) * 0.10 * scale

    def _checkerboard_velocity(self, mesh: pg.Mesh, v_bg: float, v_high: float, cell_size: float) -> np.ndarray:
        cc = mesh.cellCenters()
        XYZ = np.c_[pg.x(cc), pg.y(cc), pg.z(cc)]
        vel = np.full(mesh.cellCount(), v_bg, dtype=float)
        # alt blocks in X/Y; Z ignored for pattern simplicity
        ix = np.floor_divide(XYZ[:, 0], cell_size).astype(int)
        iy = np.floor_divide(XYZ[:, 1], cell_size).astype(int)
        vel[(ix + iy) % 2 == 0] = v_high
        return vel

    def _spike_velocity(self, mesh: pg.Mesh, v_bg: float, spike_loc: npt.NDArray[np.float64],
                        v_spike: float, radius: float = 50.0) -> np.ndarray:
        cc = mesh.cellCenters()
        XYZ = np.c_[pg.x(cc), pg.y(cc), pg.z(cc)]
        vel = np.full(mesh.cellCount(), v_bg, dtype=float)
        d = np.linalg.norm(XYZ - spike_loc.reshape(1, 3), axis=1)
        vel[d < radius] = v_spike
        return vel

    def _interpolate_to_grid(
        self,
        model_flat: npt.NDArray[np.float64],
        unc_flat: npt.NDArray[np.float64],
        grid: ProcessedGrid,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate cell-centered (mesh) values to regular (lat,lon,depth) grid."""
        assert self.mesh is not None, "mesh missing"
        cc = self.mesh.cellCenters()
        pts = np.c_[pg.x(cc), pg.y(cc), pg.z(cc)]  # (Ncells, 3)

        lat = grid.ds["lat"].values
        lon = grid.ds["lon"].values
        depth = grid.ds["depth"].values  # positive down

        # Make target XYZ in meters (simple lat/lon->m scale; assume small area)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        scale = 111_000.0
        X = (LON.ravel()) * scale
        Y = (LAT.ravel()) * scale

        model_3d = np.zeros((len(lat), len(lon), len(depth)), dtype=float)
        unc_3d = np.zeros_like(model_3d)

        for k, d in enumerate(depth):
            Z = np.full_like(X, -float(d))  # z up positive in PyGIMLi; use negative depth
            Q = np.column_stack([X, Y, Z])
            m = griddata(pts, model_flat, Q, method="linear", fill_value=float(self.v_min))
            u = griddata(pts, unc_flat, Q, method="linear", fill_value=0.0)
            model_3d[:, :, k] = m.reshape(LAT.shape)
            unc_3d[:, :, k] = u.reshape(LAT.shape)

        return model_3d, unc_3d
