"""Zone-aware raster reprojection pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from pyproj import CRS

from .grid import ZoneGrid
from .cogs import write_cog
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def intersects(bounds_a: Iterable[float], bounds_b: Iterable[float]) -> bool:
    ax1, ay1, ax2, ay2 = bounds_a
    bx1, by1, bx2, by2 = bounds_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def reproject_to_zone(
    src_path: Path,
    dst_dir: Path,
    grid: ZoneGrid,
    zone_epsg: str,
    resampling: Resampling = Resampling.bilinear,
) -> Optional[Path]:
    zone_cfg = grid.zones[zone_epsg]
    dst_crs = CRS.from_epsg(int(zone_epsg))
    dst_bounds = transform_bounds("epsg:4326", dst_crs, *zone_cfg.bbox4326, densify_pts=21)
    dst_x_min, dst_y_min, dst_x_max, dst_y_max = dst_bounds
    pixel_size = grid.pixel_size
    dst_width = int(np.ceil((dst_x_max - dst_x_min) / pixel_size))
    dst_height = int(np.ceil((dst_y_max - dst_y_min) / pixel_size))
    dst_transform = Affine(pixel_size, 0, dst_x_min, 0, -pixel_size, dst_y_max)

    with rasterio.open(src_path) as src:
        src_bounds = transform_bounds(src.crs, "epsg:4326", *src.bounds, densify_pts=21)
        if not intersects(src_bounds, zone_cfg.bbox4326):
            return None
        dst_array = np.full(
            (src.count, dst_height, dst_width),
            grid.nodata,
            dtype=src.dtypes[0],
        )
        for band in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=dst_array[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=grid.nodata,
            )
        if np.all(dst_array == grid.nodata):
            LOGGER.debug("Skipping %s for zone %s (no overlap after reprojection)", src_path.name, zone_epsg)
            return None
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "crs": dst_crs,
                "nodata": grid.nodata,
            }
        )
        memfile = rasterio.io.MemoryFile()
        with memfile.open(**profile) as dataset:
            dataset.write(dst_array)
        dst_path = dst_dir / zone_epsg / f"{src_path.stem}_{zone_cfg.name}.tif"
        with memfile.open() as dataset:
            write_cog(dataset, dst_path)
        memfile.close()
        LOGGER.info("Wrote %s", dst_path)
        return dst_path


def run_reprojection(
    source_dir: Path,
    dst_dir: Path,
    grid: ZoneGrid,
    zones: Optional[List[str]] = None,
) -> List[Path]:
    paths: List[Path] = []
    zone_list = zones if zones else list(grid.zones.keys())
    for src_path in sorted(source_dir.rglob("*.tif")):
        for zone_epsg in zone_list:
            result = reproject_to_zone(src_path, dst_dir, grid, zone_epsg)
            if result:
                paths.append(result)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproject rasters into UTM zones.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute reprojection over a directory")
    run_parser.add_argument("--tiling", type=Path, required=True, help="Path to tiling_zones.yaml")
    run_parser.add_argument("--source", type=Path, default=Path("data/raw"), help="Directory of source rasters")
    run_parser.add_argument("--dest", type=Path, default=Path("data/interim"), help="Destination directory")
    run_parser.add_argument("--zones", nargs="*", help="Specific EPSG codes to process")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        grid = ZoneGrid(args.tiling)
        run_reprojection(args.source, args.dest, grid, args.zones)


if __name__ == "__main__":
    main()
