"""CORONA declassified satellite imagery (1960s-70s, ~1.8-2.7 m): find,
preview, read, georeference, and export — from the open CAST archive, no
account required.

Why CORONA
----------
CORONA (public cover name "Discoverer", cameras KH-1..KH-4B) was the first US
reconnaissance satellite program (1960-1972); the film was declassified in
1995. Two properties make it uniquely valuable for landscape archaeology and
long-baseline change detection:

* resolution ~1.8-2.7 m (KH-4B) — 4-15x finer than today's free optical data;
* acquisition date — the frames pre-date mechanized agriculture, dams, urban
  sprawl, and recent conflict, so features long erased on the ground (small
  tells, hollow-way road networks, canals) are often crisply visible.

Data source and credit
----------------------
Imagery is streamed from the University of Arkansas CAST "CORONA Atlas of the
Middle East" project archive (https://corona.cast.uark.edu — Casana &
Cothren's team), which serves mission film scans as NITF + overview pyramids:

    https://cast-ftp1.cast.uark.edu/coronaftp/    (217 missions)
    https://cast-ftp1.cast.uark.edu/coronaftp2/

If you use this data, credit the CORONA Atlas project (CC BY-SA per their
site; the underlying USGS declassified imagery is public domain) and cite:
Casana, J. & Cothren, J., "The CORONA Atlas Project", in Comer & Harrower
(eds.), Mapping Archaeological Landscapes from Space (2013).

Layout: <mission>/<mission><a|f>/ds<mission><a|f><strip>.ntf
  'a' = aft camera, 'f' = forward camera (KH-4B is a stereo pair); each .ntf
  is a raw film-scan strip (~107k x 7.4k px, ~800 MB, uint8) with a tiled
  .ovr overview pyramid beside it — so a whole-strip quicklook is a ~2 s
  range-read and a full-resolution window costs only what you crop.

The strips are raw film scans, NOT georeferenced. This module's workflow:

    1. list_missions() / list_strips()          find film
    2. quicklook(url)                           preview a whole strip
    3. read_window(url, ...)                    full-res crop of an area
    4. add_gcp(gcps, ...) x >=3 (6+ for order=2), matching landmarks
       between the strip and any modern map/imagery
    5. fit_report(gcps)                         check residuals, drop bad GCPs
    6. warp_to_grid(url, gcps, ...)             north-up lon/lat image
    7. save_geotiff(path, img, bbox)            open it in QGIS/Google Earth

Film distortion note: KH-4B optics are panoramic; scale varies along the
strip. An affine fit (order=1) is good over ~10-20 km windows; use order=2
and well-spread GCPs for larger areas, and prefer several local fits over
one global fit.

CLI:  python -m archaeo_intel.corona missions
      python -m archaeo_intel.corona strips 1102-1025d
      python -m archaeo_intel.corona quicklook <ntf_url> -o strip.png
      python -m archaeo_intel.corona warp <ntf_url> gcps.json -o site.tif
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

BASES = ("https://cast-ftp1.cast.uark.edu/coronaftp/",
         "https://cast-ftp1.cast.uark.edu/coronaftp2/")

_HREF = re.compile(r'href="([^"?/][^"]*)"')
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQ = 111_320.0


# --------------------------------------------------------------- archive I/O

def _listing(url: str, retries: int = 3) -> list[str]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "archaeo-intel-corona/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
            return _HREF.findall(html)
        except Exception as exc:            # transient university-server hiccups
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def list_missions() -> list[str]:
    """All mission directory names across both archive roots."""
    out: list[str] = []
    for base in BASES:
        try:
            out += [h.rstrip("/") for h in _listing(base) if re.match(r"\d{4}-", h)]
        except Exception:
            continue
    return sorted(set(out))


def list_strips(mission: str) -> list[str]:
    """Full .ntf URLs for every strip/camera of a mission."""
    urls: list[str] = []
    for base in BASES:
        try:
            cams = [h for h in _listing(base + mission + "/") if h.endswith("/")]
        except Exception:
            continue
        for cam in cams:
            try:
                for f in _listing(base + mission + "/" + cam):
                    if f.endswith(".ntf"):
                        urls.append(base + mission + "/" + cam + f)
            except Exception:
                continue
        if urls:
            break
    return urls


def _open_remote(ntf_url: str):
    import rasterio
    # GDAL must see the sidecar .ovr; only allow exactly the extensions needed
    os.environ.pop("GDAL_DISABLE_READDIR_ON_OPEN", None)
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".ntf,.ovr,.aux,.xml"
    return rasterio.open("/vsicurl/" + ntf_url)


def strip_shape(ntf_url: str) -> tuple[int, int]:
    """(height, width) of a strip in native pixels."""
    with _open_remote(ntf_url) as ds:
        return ds.height, ds.width


def quicklook(ntf_url: str, factor: int = 64) -> np.ndarray:
    """Decimated whole-strip image via the .ovr pyramid (~2 s, a few MB)."""
    with _open_remote(ntf_url) as ds:
        return ds.read(1, out_shape=(1, ds.height // factor, ds.width // factor))


def read_window(ntf_url: str, col_off: int, row_off: int,
                width: int, height: int, factor: int = 1) -> np.ndarray:
    """Range-read a rectangular window from a strip.

    factor > 1 returns the window decimated by that factor (served from the
    overview pyramid where possible — much faster for large windows).
    """
    import rasterio
    from rasterio.windows import Window
    with _open_remote(ntf_url) as ds:
        return ds.read(1, window=Window(col_off, row_off, width, height),
                       out_shape=(1, max(1, height // factor),
                                  max(1, width // factor)))


# ------------------------------------------------------------ GCPs & fitting

def add_gcp(gcps: list | None, col: float, row: float,
            lon: float, lat: float, note: str = "") -> list:
    """Append one ground control point; returns the (new) list.

    A GCP pairs a strip pixel (col,row) with its real-world (lon,lat) —
    typically a road junction, wadi confluence, or building corner you can
    identify in both the strip and modern imagery.
    """
    gcps = list(gcps) if gcps else []
    gcps.append({"col": float(col), "row": float(row),
                 "lon": float(lon), "lat": float(lat), "note": note})
    return gcps


def save_gcps(path: str | Path, gcps: list, ntf_url: str = "") -> None:
    Path(path).write_text(json.dumps(
        {"ntf_url": ntf_url, "gcps": gcps}, indent=1), encoding="utf-8")


def load_gcps(path: str | Path) -> tuple[list, str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["gcps"], d.get("ntf_url", "")


def _design(px: np.ndarray, order: int) -> np.ndarray:
    c, r = px[:, 0], px[:, 1]
    cols = [np.ones_like(c), c, r]
    if order == 2:
        cols += [c * c, c * r, r * r]
    return np.column_stack(cols)


def fit_transform(gcps: list, order: int = 1) -> dict:
    """Fit pixel->lon/lat (and the inverse) from GCPs.

    order=1: affine (needs >=3 GCPs). order=2: quadratic, handles panoramic
    distortion over larger windows (needs >=6). Returns a model dict for
    px_to_ll / ll_to_px / warp_to_grid.
    """
    need = 3 if order == 1 else 6
    if len(gcps) < need:
        raise ValueError(f"order={order} needs >= {need} GCPs, got {len(gcps)}")
    px = np.array([[g["col"], g["row"]] for g in gcps], float)
    ll = np.array([[g["lon"], g["lat"]] for g in gcps], float)
    A_fwd = _design(px, order)
    A_inv = _design(ll, order)
    fwd = np.linalg.lstsq(A_fwd, ll, rcond=None)[0]      # px -> ll
    inv = np.linalg.lstsq(A_inv, px, rcond=None)[0]      # ll -> px
    return {"order": order, "fwd": fwd, "inv": inv}


def px_to_ll(model: dict, col, row) -> tuple[np.ndarray, np.ndarray]:
    p = np.column_stack([np.atleast_1d(col).astype(float),
                         np.atleast_1d(row).astype(float)])
    out = _design(p, model["order"]) @ model["fwd"]
    return out[:, 0], out[:, 1]


def ll_to_px(model: dict, lon, lat) -> tuple[np.ndarray, np.ndarray]:
    p = np.column_stack([np.atleast_1d(lon).astype(float),
                         np.atleast_1d(lat).astype(float)])
    out = _design(p, model["order"]) @ model["inv"]
    return out[:, 0], out[:, 1]


def fit_report(gcps: list, order: int = 1) -> dict:
    """Fit + per-GCP residuals in meters. Use to spot a mis-clicked GCP:
    one point with residuals several times the rest is a bad click."""
    model = fit_transform(gcps, order)
    lon_p, lat_p = px_to_ll(model, [g["col"] for g in gcps],
                            [g["row"] for g in gcps])
    res_m = []
    for g, lo, la in zip(gcps, lon_p, lat_p):
        dx = (lo - g["lon"]) * _M_PER_DEG_LON_EQ * np.cos(np.radians(g["lat"]))
        dy = (la - g["lat"]) * _M_PER_DEG_LAT
        res_m.append(float(np.hypot(dx, dy)))
    return {"model": model, "residuals_m": res_m,
            "rms_m": float(np.sqrt(np.mean(np.square(res_m))))}


# ------------------------------------------------------------------- warping

def warp_to_grid(src, gcps: list, res_m: float = 4.0, order: int = 1,
                 bbox: tuple | None = None, margin_frac: float = 0.03,
                 read_factor: int = 1):
    """Resample a strip (or array) onto a north-up EPSG:4326 grid.

    src      : strip URL, or a 2-D ndarray (full-strip pixel space) for
               offline use/testing.
    gcps     : ground control points (see add_gcp) tying strip px to lon/lat.
    res_m    : output pixel size in meters.
    bbox     : (lon_min, lat_min, lon_max, lat_max); default = GCP extent
               plus margin_frac.
    returns  : (image, bbox) — image is float32, NaN outside the source;
               pass both to save_geotiff().
    """
    from scipy.ndimage import map_coordinates
    model = fit_transform(gcps, order)
    if bbox is None:
        lons = [g["lon"] for g in gcps]; lats = [g["lat"] for g in gcps]
        mx = (max(lons) - min(lons)) * margin_frac
        my = (max(lats) - min(lats)) * margin_frac
        bbox = (min(lons) - mx, min(lats) - my, max(lons) + mx, max(lats) + my)
    lon0, lat0, lon1, lat1 = bbox
    lat_c = 0.5 * (lat0 + lat1)
    W = max(2, int(round((lon1 - lon0) * _M_PER_DEG_LON_EQ *
                         np.cos(np.radians(lat_c)) / res_m)))
    H = max(2, int(round((lat1 - lat0) * _M_PER_DEG_LAT / res_m)))
    lon_g = lon0 + (np.arange(W) + 0.5) / W * (lon1 - lon0)
    lat_g = lat1 - (np.arange(H) + 0.5) / H * (lat1 - lat0)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    cols, rows = ll_to_px(model, LON.ravel(), LAT.ravel())
    cols = cols.reshape(H, W); rows = rows.reshape(H, W)

    if isinstance(src, np.ndarray):
        band = src.astype("float32")
        c_off = r_off = 0
        f = 1
    else:
        pad = 8 * read_factor
        c0 = int(max(np.nanmin(cols) - pad, 0))
        r0 = int(max(np.nanmin(rows) - pad, 0))
        c1 = int(np.nanmax(cols) + pad); r1 = int(np.nanmax(rows) + pad)
        if c1 <= c0 or r1 <= r0:
            raise ValueError("GCP model maps the bbox outside the strip")
        band = read_window(src, c0, r0, c1 - c0, r1 - r0,
                           factor=read_factor).astype("float32")
        c_off, r_off, f = c0, r0, read_factor

    cc = (cols - c_off) / f
    rr = (rows - r_off) / f
    img = map_coordinates(band, [rr.ravel(), cc.ravel()], order=1,
                          mode="constant", cval=np.nan).reshape(H, W)
    inside = (cc >= 0) & (cc <= band.shape[1] - 1) & \
             (rr >= 0) & (rr <= band.shape[0] - 1)
    img[~inside] = np.nan
    return img.astype("float32"), bbox


def save_geotiff(path: str | Path, img: np.ndarray, bbox: tuple) -> None:
    """Write a warp_to_grid() result as an EPSG:4326 GeoTIFF (QGIS-ready)."""
    import rasterio
    from rasterio.transform import from_bounds
    lon0, lat0, lon1, lat1 = bbox
    h, w = img.shape
    with rasterio.open(
            str(path), "w", driver="GTiff", width=w, height=h, count=1,
            dtype="float32", crs="EPSG:4326",
            transform=from_bounds(lon0, lat0, lon1, lat1, w, h),
            nodata=float("nan"), compress="deflate") as ds:
        ds.write(img.astype("float32"), 1)


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Contrast-stretched 8-bit PNG of any array from this module."""
    from matplotlib.image import imsave
    p2, p98 = np.nanpercentile(img, [2, 98])
    out = np.clip((img - p2) / (p98 - p2 + 1e-9), 0, 1)
    imsave(str(path), np.nan_to_num(out, nan=0.0), cmap="gray")


# ----------------------------------------------------- backward compatibility

def fit_affine(pixel_pts, geo_pts):
    """Legacy helper: least-squares affine (col,row)->(lon,lat), >=3 points.
    Prefer fit_transform()/fit_report() for new code."""
    px = np.asarray(pixel_pts, float)
    geo = np.asarray(geo_pts, float)
    if len(px) < 3:
        raise ValueError("need >=3 GCPs")
    A = np.column_stack([px, np.ones(len(px))])
    m_lon, *_ = np.linalg.lstsq(A, geo[:, 0], rcond=None)
    m_lat, *_ = np.linalg.lstsq(A, geo[:, 1], rcond=None)
    return np.vstack([m_lon, m_lat])


def apply_affine(M, col, row):
    """Legacy helper: map one strip pixel to (lon, lat)."""
    v = np.array([col, row, 1.0])
    return float(M[0] @ v), float(M[1] @ v)


# --------------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m archaeo_intel.corona",
        description="CORONA open-archive access (see module docstring)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("missions", help="list all mission ids")
    p = sub.add_parser("strips", help="list strip URLs of a mission")
    p.add_argument("mission")
    p = sub.add_parser("quicklook", help="whole-strip preview PNG")
    p.add_argument("url"); p.add_argument("-o", default="quicklook.png")
    p.add_argument("--factor", type=int, default=64)
    p = sub.add_parser("window", help="crop PNG (native px coords)")
    p.add_argument("url")
    p.add_argument("col", type=int); p.add_argument("row", type=int)
    p.add_argument("width", type=int); p.add_argument("height", type=int)
    p.add_argument("-o", default="window.png")
    p.add_argument("--factor", type=int, default=1)
    p = sub.add_parser("warp", help="GCP-warp to GeoTIFF")
    p.add_argument("url"); p.add_argument("gcps", help="gcps.json (save_gcps)")
    p.add_argument("-o", default="corona_warp.tif")
    p.add_argument("--res-m", type=float, default=4.0)
    p.add_argument("--order", type=int, default=1, choices=(1, 2))
    a = ap.parse_args(argv)

    if a.cmd == "missions":
        print("\n".join(list_missions()))
    elif a.cmd == "strips":
        print("\n".join(list_strips(a.mission)))
    elif a.cmd == "quicklook":
        save_png(a.o, quicklook(a.url, a.factor))
        print(a.o)
    elif a.cmd == "window":
        save_png(a.o, read_window(a.url, a.col, a.row, a.width, a.height,
                                  factor=a.factor))
        print(a.o)
    elif a.cmd == "warp":
        gcps, _ = load_gcps(a.gcps)
        rep = fit_report(gcps, a.order)
        print("GCP residuals (m):",
              " ".join(f"{r:.1f}" for r in rep["residuals_m"]),
              f"| RMS {rep['rms_m']:.1f} m")
        img, bbox = warp_to_grid(a.url, gcps, res_m=a.res_m, order=a.order)
        save_geotiff(a.o, img, bbox)
        print(a.o)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
