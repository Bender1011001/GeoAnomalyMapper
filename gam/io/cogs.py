"""COG writing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import rasterio
from rasterio.shutil import copy as rio_copy


def write_cog(src_dataset: rasterio.io.DatasetReader, dst_path: Path, **options: Any) -> Path:
    """Persist ``src_dataset`` as a Cloud Optimized GeoTIFF.

    The function first attempts to use the COG driver. If unavailable, it
    gracefully falls back to writing a standard GeoTIFF with tiling and
    compression enabled.
    """
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    profile = src_dataset.profile.copy()
    profile.update({"driver": "GTiff", "tiled": True, "compress": "deflate"})
    tmp_path = dst.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as sink:
        for idx in range(1, src_dataset.count + 1):
            sink.write(src_dataset.read(idx), idx)
        sink.set_transform(src_dataset.transform)
        sink.set_crs(src_dataset.crs)
        sink.update_tags(**src_dataset.tags())
    try:
        rio_copy(
            tmp_path,
            dst,
            driver="COG",
            forward_anywhere=True,
            **{"compress": "deflate", "blocksize": 512, **options},
        )
    except Exception:
        rio_copy(
            tmp_path,
            dst,
            driver="GTiff",
            copy_src_overviews=True,
            **{"compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512, **options},
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return dst


__all__ = ["write_cog"]
