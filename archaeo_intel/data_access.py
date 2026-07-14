"""Free public imagery access: Earth Search STAC (AWS) + Microsoft Planetary
Computer (Azure SAS signing, for collections that are requester-pays on AWS,
e.g. Landsat C2L2 thermal).

All reads are lazy windowed /vsicurl range-reads — no bulk downloads.
"""
import os
import time
from urllib.parse import quote

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MPC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

_mpc_tokens: dict = {}


def vsicurl(url: str) -> str:
    """GDAL /vsicurl path for an https or s3 asset URL (anonymous AWS)."""
    if url.startswith("s3://"):
        bucket, key = url[5:].split("/", 1)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
    scheme, rest = url.split("//", 1)
    return "/vsicurl/" + scheme + "//" + quote(rest)


def stac_search(collection: str, bbox, datetime: str | None = None,
                query: dict | None = None, limit: int = 50,
                sortby=None, endpoint: str = EARTH_SEARCH):
    body: dict = {"collections": [collection], "bbox": list(bbox), "limit": limit}
    if datetime:
        body["datetime"] = datetime
    if query:
        body["query"] = query
    if sortby:
        body["sortby"] = sortby
    r = requests.post(endpoint, json=body, timeout=90)
    r.raise_for_status()
    return r.json().get("features", [])


def mpc_sign(href: str) -> str:
    """Append a Planetary Computer SAS token (cached per container)."""
    account = href.split("//", 1)[1].split(".", 1)[0]
    container = href.split(".blob.core.windows.net/", 1)[1].split("/", 1)[0]
    key = f"{account}/{container}"
    now = time.time()
    if key not in _mpc_tokens or _mpc_tokens[key][1] - now < 120:
        r = requests.get(f"{MPC_SIGN}/{account}/{container}", timeout=60)
        r.raise_for_status()
        j = r.json()
        expiry = time.mktime(time.strptime(j["msft:expiry"], "%Y-%m-%dT%H:%M:%SZ"))
        _mpc_tokens[key] = (j["token"], expiry)
    return href + "?" + _mpc_tokens[key][0]


def read_grid(url: str, grid, width: int, height: int,
              resampling: Resampling = Resampling.bilinear,
              signed: bool = False, retries: int = 3) -> np.ndarray:
    """Reproject-read one band of a remote COG onto an EPSG:4326 grid.

    grid = (lon_min, lat_min, lon_max, lat_max). Anonymous AWS by default;
    signed=True routes through Planetary Computer SAS signing.
    """
    transform = from_bounds(*grid, width, height)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            path = "/vsicurl/" + mpc_sign(url) if signed else vsicurl(url)
            if not signed:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            with rasterio.open(path) as ds:
                out = np.full((height, width), np.nan, "float32")
                reproject(rasterio.band(ds, 1), out, dst_transform=transform,
                          dst_crs="EPSG:4326", resampling=resampling,
                          src_nodata=ds.nodata, dst_nodata=np.nan)
            return out
        except Exception as exc:                     # transient object-store errors
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]
