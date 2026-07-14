"""CORONA declassified spy imagery (1960s-70s, ~1.8-2.7 m) — open access.

DISCOVERY (2026-07-13): the University of Arkansas CAST CORONA Atlas project
serves its complete mission archive as an OPEN https directory — no account:

    https://cast-ftp1.cast.uark.edu/coronaftp/   (217 missions)
    https://cast-ftp1.cast.uark.edu/coronaftp2/  (more missions)

Layout: <mission>/<mission><a|f>/ds<mission><a|f><strip>.ntf
  - 'a' = aft camera, 'f' = forward camera (KH-4B stereo pair)
  - each .ntf is a raw film-scan strip (~107k x 7.4k px, ~800 MB, uint8)
  - .ntf.ovr sidecar carries a tiled overview pyramid (levels 4-128); GDAL
    range-reads make a whole-strip quicklook cost ~2 s / few MB (validated).

The raw strips are NOT georeferenced. Georeferencing paths:
  1. quicklook -> identify coverage by eye/VLM -> pick GCPs against Sentinel-2
     -> fit_affine() below -> warp locally. Fully account-free.
  2. USGS EarthExplorer 'Declass 1/2/3' datasets have footprint search and
     (for many strips) georeferenced products — requires a free USGS account
     (user action; account creation is out of scope for automation).

Why CORONA matters (measured): our free-data ceiling for small sites is AUC
~0.62 at 10-30 m. CORONA is 4-15x finer AND pre-dates mechanized agriculture —
hollow-ways and small tells that are erased today are intact in these frames
(Ur 2003 mapped hollow-ways from exactly this imagery).
"""
import os
import re
import urllib.request

import numpy as np

BASES = ("https://cast-ftp1.cast.uark.edu/coronaftp/",
         "https://cast-ftp1.cast.uark.edu/coronaftp2/")

_HREF = re.compile(r'href="([^"?/][^"]*)"')


def _listing(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "archaeo-intel/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return _HREF.findall(html)


def list_missions() -> list[str]:
    """All mission directory names across both archive roots."""
    out = []
    for base in BASES:
        try:
            out += [h.rstrip("/") for h in _listing(base) if re.match(r"\d{4}-", h)]
        except Exception:
            continue
    return sorted(set(out))


def list_strips(mission: str) -> list[str]:
    """Full .ntf URLs for every strip/camera of a mission."""
    urls = []
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


def quicklook(ntf_url: str, factor: int = 64) -> np.ndarray:
    """Decimated whole-strip image via the .ovr pyramid (~2 s, few MB)."""
    import rasterio
    os.environ.pop("GDAL_DISABLE_READDIR_ON_OPEN", None)  # sidecar .ovr needed
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".ntf,.ovr,.aux,.xml"
    with rasterio.open("/vsicurl/" + ntf_url) as ds:
        return ds.read(1, out_shape=(1, ds.height // factor, ds.width // factor))


def read_window(ntf_url: str, col_off: int, row_off: int,
                width: int, height: int, factor: int = 1) -> np.ndarray:
    """Range-read a full- or reduced-resolution window from a strip."""
    import rasterio
    from rasterio.windows import Window
    os.environ.pop("GDAL_DISABLE_READDIR_ON_OPEN", None)
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".ntf,.ovr,.aux,.xml"
    with rasterio.open("/vsicurl/" + ntf_url) as ds:
        return ds.read(1, window=Window(col_off, row_off, width, height),
                       out_shape=(1, height // factor, width // factor))


def fit_affine(pixel_pts, geo_pts):
    """Least-squares affine (px,row)->(lon,lat) from >=3 ground control points.

    pixel_pts: [(col,row), ...]; geo_pts: [(lon,lat), ...]. Returns a 2x3
    matrix M with [lon,lat]' = M @ [col,row,1]'. For strip-scale accuracy use
    >=6 well-spread GCPs; film distortion means local (per-AOI) fits beat one
    global fit.
    """
    px = np.asarray(pixel_pts, float)
    geo = np.asarray(geo_pts, float)
    if len(px) < 3:
        raise ValueError("need >=3 GCPs")
    A = np.column_stack([px, np.ones(len(px))])
    m_lon, *_ = np.linalg.lstsq(A, geo[:, 0], rcond=None)
    m_lat, *_ = np.linalg.lstsq(A, geo[:, 1], rcond=None)
    return np.vstack([m_lon, m_lat])


def apply_affine(M, col, row):
    """Map strip pixel coords to (lon, lat) with a fit_affine matrix."""
    v = np.array([col, row, 1.0])
    return float(M[0] @ v), float(M[1] @ v)
