"""OPERA DISP-S1 access and displacement-cube assembly.

OPERA L3 DISP-S1 provides InSAR time-series displacement over North America as
a stack of NetCDF granules per frame. Each granule holds a `displacement` grid
(LOS, meters) for one secondary epoch relative to a reference, plus quality
layers (`recommended_mask`, `temporal_coherence`). This module discovers a
frame's granules via asf_search, downloads an AOI-subset for the requested
window, and assembles a coherence-masked (T, H, W) cube on a common grid with a
decimal-year time axis — the input to `deformation_intel.timeseries`.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GRANULE_RE = re.compile(
    r"OPERA_L3_DISP-S1_IW_(?P<frame>F\d+)_VV_"
    r"(?P<ref>\d{8})T\d{6}Z_(?P<sec>\d{8})T\d{6}Z"
)


def _coerce_bytes(value) -> int:
    """ASF may report `bytes` as an int or a dict of per-asset sizes."""
    if isinstance(value, dict):
        nums = [v for v in value.values() if isinstance(v, (int, float))]
        return int(max(nums)) if nums else 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def decimal_year(date_str: str) -> float:
    """YYYYMMDD -> decimal year."""
    d = datetime.strptime(date_str, "%Y%m%d")
    year_start = datetime(d.year, 1, 1)
    next_year = datetime(d.year + 1, 1, 1)
    return d.year + (d - year_start).total_seconds() / (next_year - year_start).total_seconds()


@dataclass
class DispGranule:
    frame: str
    reference_date: str
    secondary_date: str
    url: str
    size_bytes: int = 0

    @property
    def t(self) -> float:
        return decimal_year(self.secondary_date)


def search_disp_frames(lat: float, lon: float, *, max_results: int = 3000) -> Dict[str, List[DispGranule]]:
    """Return OPERA DISP-S1 granules grouped by frame stack ID over a point."""
    import asf_search as asf

    results = asf.search(
        dataset=asf.DATASET.OPERA_S1,
        processingLevel="DISP-S1",
        intersectsWith=f"POINT({lon} {lat})",
        maxResults=max_results,
    )
    frames: Dict[str, List[DispGranule]] = {}
    for r in results:
        p = r.properties
        stack_id = p.get("OperaDispStackID")
        name = p.get("sceneName") or p.get("fileID") or ""
        m = GRANULE_RE.search(name)
        url = p.get("url") or ""
        if not (stack_id and m and url):
            continue
        frames.setdefault(stack_id, []).append(
            DispGranule(
                frame=m.group("frame"),
                reference_date=m.group("ref"),
                secondary_date=m.group("sec"),
                url=url,
                size_bytes=_coerce_bytes(p.get("bytes")),
            )
        )
    for gid in frames:
        frames[gid].sort(key=lambda g: g.secondary_date)
    return frames


def pick_primary_frame(frames: Dict[str, List[DispGranule]]) -> str:
    """Choose the frame stack with the most epochs (best temporal sampling)."""
    if not frames:
        raise ValueError("No OPERA DISP-S1 frames found for this location")
    return max(frames, key=lambda k: len(frames[k]))


def download_granules(
    granules: Sequence[DispGranule],
    out_dir: Path,
    *,
    session=None,
    max_workers: int = 6,
) -> List[Path]:
    """Download granule NetCDFs with an authenticated ASF session (cached)."""
    import asf_search as asf

    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [g.url for g in granules]
    existing = {p.name for p in out_dir.glob("*.nc")}
    todo = [u for u in urls if Path(u).name not in existing]
    if todo:
        logger.info("Downloading %d OPERA granules (%d cached) -> %s", len(todo), len(urls) - len(todo), out_dir)
        asf.download_urls(urls=todo, path=str(out_dir), session=session, processes=max_workers)
    paths = [out_dir / Path(u).name for u in urls]
    return [p for p in paths if p.exists()]


def _read_layer(ds, candidates: Sequence[str]):
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name]
    return None


def _granule_nc_url(g) -> Optional[str]:
    """Return the .nc data URL for an earthaccess granule, preferring the cloud
    (S3/CloudFront) copy but accepting the on-prem datapool HTTPS copy."""
    try:
        links = g.data_links(access="external") + g.data_links()
    except Exception:
        links = []
    nc = [u for u in dict.fromkeys(links) if str(u).endswith(".nc")]
    if not nc:
        return None
    # Prefer cloud host (authenticates cleanly), fall back to datapool.
    cloud = [u for u in nc if "earthdatacloud" in u or "cloudfront" in u or u.startswith("s3")]
    return (cloud or nc)[0]


def _open_authenticated(url: str, fs):
    """Open a remote OPERA .nc via an authenticated fsspec session."""
    import xarray as xr

    fobj = fs.open(url)
    return xr.open_dataset(fobj, engine="h5netcdf")


def _read_one_window(open_ds, lon, lat, half, coherence_threshold):
    """Read the masked AOI window from an already-opened granule dataset factory.

    open_ds() must yield a fresh xarray Dataset each call so a retry gets a clean
    HTTP range session (a 503'd/401'd fsspec handle stays broken).
    Returns (disp, x_sub, y_sub, crs_wkt) or None if the AOI is off-grid.
    """
    import numpy as np
    from pyproj import Transformer

    ds = open_ds()
    try:
        crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
        tr = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        x = ds["x"].values
        y = ds["y"].values
        xs = np.where((x >= cx - half) & (x <= cx + half))[0]
        ys = np.where((y >= cy - half) & (y <= cy + half))[0]
        if xs.size == 0 or ys.size == 0:
            return None
        ysl = slice(int(ys.min()), int(ys.max()) + 1)
        xsl = slice(int(xs.min()), int(xs.max()) + 1)
        disp = ds["displacement"].isel(y=ysl, x=xsl).values.astype(np.float32)
        coh = ds["temporal_coherence"].isel(y=ysl, x=xsl).values
        mask = np.isfinite(disp) & np.isfinite(coh) & (coh >= coherence_threshold)
        if "recommended_mask" in ds.variables:
            mask &= (ds["recommended_mask"].isel(y=ysl, x=xsl).values > 0)
        if "water_mask" in ds.variables:
            mask &= (ds["water_mask"].isel(y=ysl, x=xsl).values > 0)
        disp = np.where(mask, disp, np.nan)
        return disp, x[xsl], y[ysl], crs_wkt
    finally:
        try:
            ds.close()
        except Exception:
            pass


def build_aoi_cube(
    lat: float,
    lon: float,
    *,
    half_width_km: float = 12.0,
    coherence_threshold: float = 0.6,
    max_epochs: Optional[int] = None,
    progress: bool = True,
    cache_dir: Optional[Path] = None,
    max_retries: int = 4,
    retry_backoff_s: float = 3.0,
    allow_download: bool = True,
    asf_session=None,
) -> dict:
    """Stream an OPERA DISP-S1 AOI displacement cube via lazy windowed reads.

    Downloads only the AOI window from each granule (not the ~360 MB full
    frames), coherence-masks, then stitches the OPERA reference eras into one
    continuous cumulative series. Requires earthaccess + h5netcdf + pyproj and
    EARTHDATA_USERNAME/PASSWORD in the environment.

    Returns dict: cube (T,H,W) meters LOS continuous cumulative, t (decimal yr),
    sec_dates, ref_dates, x/y (UTM meters), crs_wkt, frame.
    """
    import earthaccess
    import numpy as np

    earthaccess.login(strategy="environment")
    # One authenticated fsspec HTTPS session works across BOTH the cloud archive
    # and the on-prem ASF datapool (earthaccess.open alone 401s on datapool).
    try:
        fs = earthaccess.get_fsspec_https_session()
    except Exception as exc:
        logger.warning("authenticated fsspec session unavailable (%s); "
                       "falling back to earthaccess.open", type(exc).__name__)
        fs = None

    # ASFSession authenticates to the ASF datapool (URS redirect) for the
    # download fallback, which earthaccess.download does not handle.
    if asf_session is None:
        try:
            import asf_search as asf
            import os as _os
            user = _os.environ.get("EARTHDATA_USERNAME")
            pwd = _os.environ.get("EARTHDATA_PASSWORD")
            if user and pwd:
                asf_session = asf.ASFSession().auth_with_creds(user, pwd)
        except Exception as exc:
            logger.warning("ASF session unavailable for download fallback: %s", type(exc).__name__)
            asf_session = None
    results = earthaccess.search_data(
        short_name="OPERA_L3_DISP-S1_V1",
        point=(lon, lat),
        count=4000,
    )
    # Group by frame (from producer granule id) and keep the largest stack.
    def _frame_of(g):
        name = g["umm"]["GranuleUR"] if "umm" in g else str(g)
        m = GRANULE_RE.search(name)
        return m.group("frame") if m else "unknown"

    by_frame: Dict[str, list] = {}
    for g in results:
        by_frame.setdefault(_frame_of(g), []).append(g)
    frame = max(by_frame, key=lambda k: len(by_frame[k]))
    granules = by_frame[frame]

    def _dates(g):
        name = g["umm"]["GranuleUR"]
        m = GRANULE_RE.search(name)
        return (m.group("ref"), m.group("sec")) if m else (None, None)

    granules = [g for g in granules if _dates(g)[1]]
    granules.sort(key=lambda g: _dates(g)[1])
    if max_epochs and len(granules) > max_epochs:
        # even temporal subsample
        step = len(granules) / max_epochs
        granules = [granules[int(i * step)] for i in range(max_epochs)]
    logger.info("OPERA frame %s: %d epochs to stream", frame, len(granules))
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    half = half_width_km * 1000.0
    cube: list = []
    sec_dates: List[str] = []
    ref_dates: List[str] = []
    xcoord = ycoord = None
    crs_wkt = None
    ref_shape = None
    skipped = 0

    for i, g in enumerate(granules):
        rd, sd = _dates(g)
        cache_path = (cache_dir / f"{frame}_{rd}_{sd}.npz") if cache_dir is not None else None
        result = None

        if cache_path is not None and cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=False)
                result = (z["disp"], z["x"], z["y"], str(z["crs"]))
            except Exception:
                result = None

        if result is None:
            nc_url = _granule_nc_url(g) if fs is not None else None

            def _lazy_factory():
                if fs is not None and nc_url is not None:
                    return _open_authenticated(nc_url, fs)
                import xarray as xr
                return xr.open_dataset(earthaccess.open([g])[0], engine="h5netcdf")

            # 1) Fast path: lazy cloud/S3 windowed read, with retry on transient
            #    errors (a 503'd/401'd fsspec handle stays broken -> re-open).
            for attempt in range(1, max_retries + 1):
                try:
                    result = _read_one_window(_lazy_factory, lon, lat, half, coherence_threshold)
                    break
                except Exception:
                    if attempt >= max_retries:
                        result = None
                        break
                    time.sleep(retry_backoff_s * attempt)

            # 2) Fallback for datapool-only frames (no cloud-lazy access):
            #    authenticated full download via asf_search (ASFSession handles
            #    the ASF datapool URS redirect that earthaccess.download does
            #    not) -> extract AOI window -> delete. Bounded disk (one .nc at
            #    a time); the cached window makes it one-time.
            if result is None and allow_download and asf_session is not None:
                dl_url = _granule_nc_url(g) or nc_url
                if dl_url:
                    import tempfile
                    import xarray as xr
                    import asf_search as asf
                    tmp = Path(tempfile.mkdtemp(prefix="opera_dl_"))
                    try:
                        asf.download_url(url=dl_url, path=str(tmp),
                                         filename=Path(dl_url).name, session=asf_session)
                        ncs = list(tmp.glob("*.nc"))
                        if ncs and ncs[0].stat().st_size > 1_000_000:
                            def _local_factory(p=ncs[0]):
                                return xr.open_dataset(p, engine="h5netcdf")
                            result = _read_one_window(_local_factory, lon, lat, half, coherence_threshold)
                    except Exception as exc:
                        logger.warning("download fallback failed %s: %s: %s",
                                       sd, type(exc).__name__, str(exc)[:80])
                        result = None
                    finally:
                        import shutil
                        shutil.rmtree(tmp, ignore_errors=True)

            if result is None:
                logger.warning("skip %s (lazy+download both failed)", sd)
                skipped += 1
            if result is not None and cache_path is not None:
                try:
                    np.savez_compressed(cache_path, disp=result[0], x=result[1],
                                        y=result[2], crs=np.str_(result[3]))
                except Exception:
                    pass

        if result is None:
            continue
        disp, xsub, ysub, wkt = result
        if crs_wkt is None:
            crs_wkt = wkt
        if ref_shape is None:
            ref_shape = disp.shape
            xcoord, ycoord = xsub, ysub
        elif disp.shape != ref_shape:
            skipped += 1
            continue
        cube.append(disp)
        sec_dates.append(sd)
        ref_dates.append(rd)
        if progress and (i % 25 == 0):
            logger.info("  streamed %d/%d (%s), %d in cube, %d skipped",
                        i + 1, len(granules), sd, len(cube), skipped)

    if not cube:
        raise RuntimeError("No OPERA epochs overlapped the AOI window")
    logger.info("assembled %d epochs (%d skipped)", len(cube), skipped)
    arr = np.stack(cube, axis=0)

    from deformation_intel.timeseries import stitch_reference_eras
    stitched = stitch_reference_eras(arr, sec_dates, ref_dates)
    t = np.array([decimal_year(d) for d in sorted(sec_dates)])
    return {
        "cube": stitched,
        "t": t,
        "sec_dates": sorted(sec_dates),
        "ref_dates": ref_dates,
        "x": xcoord,
        "y": ycoord,
        "crs_wkt": crs_wkt,
        "frame": frame,
        "coherence_threshold": coherence_threshold,
        "center_lat": lat,
        "center_lon": lon,
    }


def assemble_cube(
    granule_paths: Sequence[Path],
    times: Sequence[float],
    bbox: Tuple[float, float, float, float],
    *,
    coherence_threshold: float = 0.5,
    apply_recommended_mask: bool = True,
) -> dict:
    """Assemble a coherence-masked (T,H,W) LOS displacement cube over a bbox.

    bbox = (lon_min, lat_min, lon_max, lat_max) in the granule CRS' geographic
    coordinates. Granules on the same OPERA frame share a grid, so we subset by
    coordinate index once and stack. Returns dict with cube, t, lats, lons.
    """
    import xarray as xr

    if len(granule_paths) != len(times):
        raise ValueError("granule_paths and times length mismatch")
    lon_min, lat_min, lon_max, lat_max = bbox

    cube = None
    lats = lons = None
    kept_t: List[float] = []
    for path, t in zip(granule_paths, times):
        try:
            ds = xr.open_dataset(path, engine="h5netcdf")
        except Exception:
            ds = xr.open_dataset(path)
        try:
            disp = ds["displacement"]
            ycoord = "y" if "y" in disp.dims else ("latitude" if "latitude" in disp.dims else disp.dims[-2])
            xcoord = "x" if "x" in disp.dims else ("longitude" if "longitude" in disp.dims else disp.dims[-1])
            yv = ds[ycoord].values
            xv = ds[xcoord].values
            # Support descending y.
            y_asc = yv[0] < yv[-1]
            ys = np.where((yv >= lat_min) & (yv <= lat_max))[0]
            xs = np.where((xv >= lon_min) & (xv <= lon_max))[0]
            if ys.size == 0 or xs.size == 0:
                ds.close()
                continue
            sl = disp.isel({ycoord: slice(ys.min(), ys.max() + 1),
                            xcoord: slice(xs.min(), xs.max() + 1)}).values.astype(np.float32)
            mask = np.ones_like(sl, dtype=bool)
            coh = _read_layer(ds, ["temporal_coherence", "phase_similarity"])
            if coh is not None:
                cv = ds[coh.name].isel({ycoord: slice(ys.min(), ys.max() + 1),
                                        xcoord: slice(xs.min(), xs.max() + 1)}).values
                mask &= np.isfinite(cv) & (cv >= coherence_threshold)
            if apply_recommended_mask and "recommended_mask" in ds.variables:
                rm = ds["recommended_mask"].isel({ycoord: slice(ys.min(), ys.max() + 1),
                                                  xcoord: slice(xs.min(), xs.max() + 1)}).values
                mask &= (rm > 0)
            sl = np.where(mask, sl, np.nan)
            if not y_asc:
                sl = sl[::-1]
                yv_sub = yv[ys.min():ys.max() + 1][::-1]
            else:
                yv_sub = yv[ys.min():ys.max() + 1]
            xv_sub = xv[xs.min():xs.max() + 1]
            if cube is None:
                lats, lons = yv_sub, xv_sub
                cube = [sl]
            else:
                if sl.shape != cube[0].shape:
                    ds.close()
                    continue
                cube.append(sl)
            kept_t.append(float(t))
        finally:
            ds.close()

    if not cube:
        raise RuntimeError("No granules overlapped the requested bbox")
    arr = np.stack(cube, axis=0)
    order = np.argsort(kept_t)
    return {
        "cube": arr[order],
        "t": np.asarray(kept_t)[order],
        "lats": lats,
        "lons": lons,
        "coherence_threshold": coherence_threshold,
    }
