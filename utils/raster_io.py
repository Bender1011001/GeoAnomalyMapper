"""Raster I/O utilities for GeoAnomalyMapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import rasterio


def write_raster_with_metadata(raster: np.ndarray, path: Path, metadata_dict: Dict[str, Any]) -> None:
    """Write a raster to GeoTIFF along with a JSON metadata sidecar."""

    if raster.ndim != 2:
        raise ValueError("Raster array must be two-dimensional")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata_dict.copy()
    profile = meta.pop('raster_profile', None)
    if profile is None:
        raise ValueError("metadata_dict must include a 'raster_profile' entry")

    profile = profile.copy()
    profile.setdefault('driver', 'GTiff')
    profile.setdefault('count', 1)
    profile['dtype'] = 'float32'

    raster_to_write = raster.astype(np.float32, copy=False)

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(raster_to_write, 1)
        if meta.get('statistic'):
            dst.set_band_description(1, str(meta['statistic']))

    meta_path = path.with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

