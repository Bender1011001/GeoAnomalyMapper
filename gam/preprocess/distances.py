"""Distance raster generation utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from ..io.grid import ZoneGrid
from ..io.cogs import write_cog
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _read_geometries(path: Path) -> List[BaseGeometry]:
    geoms: List[BaseGeometry] = []
    with fiona.open(path) as src:
        for feature in src:
            geoms.append(shape(feature["geometry"]))
    return geoms


def _distance_from_geometries(
    geoms: Iterable[BaseGeometry],
    transform: rasterio.Affine,
    width: int,
    height: int,
    nodata: float,
) -> np.ndarray:
    if not geoms:
        return np.full((height, width), nodata, dtype="float32")
    affine = transform
    pixel_size = float(np.hypot(affine.a, affine.e))
    shapes = [(geom, 1) for geom in geoms]
    mask = rasterize(shapes, out_shape=(height, width), transform=affine, fill=0, dtype="uint8")
    if np.all(mask == 0):
        return np.full((height, width), nodata, dtype="float32")
    distance = distance_transform_edt(1 - mask) * pixel_size
    distance = distance.astype("float32")
    distance[mask == 1] = 0.0
    return distance


def build_distances(
    faults_path: Path,
    basins_path: Path,
    tiling_config: Path,
    output_dir: Path,
    zones: Optional[List[str]] = None,
) -> None:
    grid = ZoneGrid(tiling_config)
    faults = _read_geometries(faults_path)
    basins = [geom.boundary for geom in _read_geometries(basins_path)]
    zone_list = zones if zones else list(grid.zones.keys())
    for zone_epsg in zone_list:
        zone = grid.zones[zone_epsg]
        dst_dir = output_dir / zone_epsg
        dst_dir.mkdir(parents=True, exist_ok=True)
        dummy_tile = next(grid.iter_tiles(zone_epsg))
        width = int(dummy_tile.width / grid.pixel_size)
        height = int(dummy_tile.height / grid.pixel_size)
        transform = dummy_tile.affine_transform
        memfile = rasterio.io.MemoryFile()
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 2,
            "dtype": "float32",
            "transform": transform,
            "crs": f"EPSG:{zone_epsg}",
            "nodata": -9999.0,
        }
        with memfile.open(**profile) as dataset:
            dataset.write(_distance_from_geometries(faults, transform, width, height, -9999.0), 1)
            dataset.update_tags(1, name="dist_to_fault_m")
            dataset.write(_distance_from_geometries(basins, transform, width, height, -9999.0), 2)
            dataset.update_tags(2, name="dist_to_basin_edge_m")
        dst_path = dst_dir / "context_distances.tif"
        with memfile.open() as dataset:
            write_cog(dataset, dst_path)
        LOGGER.info("Wrote %s", dst_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build contextual distance rasters")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Build rasters")
    build_parser.add_argument("--faults", type=Path, required=True, help="Path to faults vector file")
    build_parser.add_argument("--basins", type=Path, required=True, help="Path to basins vector file")
    build_parser.add_argument("--tiling", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, default=Path("data/interim"))
    build_parser.add_argument("--zones", nargs="*")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        build_distances(args.faults, args.basins, args.tiling, args.out, args.zones)


if __name__ == "__main__":
    main()
