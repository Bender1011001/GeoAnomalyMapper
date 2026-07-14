"""Ground-truth site catalogs for validation and novelty determination.

Menze & Ur 2012 (PNAS) Upper Khabur catalog: 14,324 sites over 23,000 km2 —
the most complete archaeological inventory of any Mesopotamian landscape.
Source: Harvard Dataverse doi:10.7910/DVN/7H8K3N (CSV of UTM-37N coordinates;
column 1 = northing, column 2 = easting, despite the lat/lon header names).

Validation status (2026-07): Tell Brak matches at 117 m; 4/4 of this system's
confident VLM-triaged Khabur detections matched catalog sites at 41-373 m.
"""
import csv
from pathlib import Path

import numpy as np

DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/2373110"
DEFAULT_NPZ = Path(__file__).resolve().parent.parent / "data" / "archaeo" / \
    "menze_ur_catalog.npz"


def load_menze_ur(npz_path: Path = DEFAULT_NPZ):
    """Load the cached catalog -> (lat, lon) arrays (EPSG:4326)."""
    d = np.load(npz_path)
    return d["lat"], d["lon"]


def build_menze_ur_npz(csv_path: Path, npz_path: Path = DEFAULT_NPZ):
    """Convert the Dataverse CSV (UTM 37N) to a lat/lon npz cache."""
    from pyproj import Transformer
    northing, easting = [], []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)                                  # header
        for row in reader:
            try:
                northing.append(float(row[0]))
                easting.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    tr = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.array(easting), np.array(northing))
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, lat=np.asarray(lat), lon=np.asarray(lon))
    return np.asarray(lat), np.asarray(lon)


def nearest_site_km(lat: float, lon: float, cat_lat: np.ndarray,
                    cat_lon: np.ndarray) -> float:
    """Distance (km) to the nearest catalog site (equirectangular)."""
    d = 111.0 * np.hypot(cat_lat - lat,
                         (cat_lon - lon) * np.cos(np.radians(lat)))
    return float(d.min())


def classify_hit(lat: float, lon: float, cat_lat: np.ndarray,
                 cat_lon: np.ndarray, in_km: float = 0.5,
                 near_km: float = 1.5) -> tuple[str, float]:
    """Label a detection vs the catalog: 'in-catalog' / 'near' / 'novel-candidate'.

    'novel-candidate' means not in THIS catalog — inside the surveyed footprint
    that is defensible novelty; outside it, it only means uncatalogued-here.
    """
    d = nearest_site_km(lat, lon, cat_lat, cat_lon)
    if d < in_km:
        return "in-catalog", d
    if d < near_km:
        return "near", d
    return "novel-candidate", d
