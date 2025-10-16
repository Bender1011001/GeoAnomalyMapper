"""Multi-scale rolling feature generation."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import sobel, laplace, uniform_filter

from ..io.grid import ZoneGrid, TileDefinition
from ..io.cogs import write_cog
from ..utils.logging import get_logger
from .schema import ensure_schema

LOGGER = get_logger(__name__)


def _rolling_mean_std(data: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isnan(data)
    filled = np.nan_to_num(data, nan=0.0)
    valid = (~mask).astype(np.float32)
    sum_ = uniform_filter(filled, size=size, mode="reflect") * (size ** 2)
    count = uniform_filter(valid, size=size, mode="reflect") * (size ** 2)
    mean = np.divide(sum_, count, out=np.full_like(sum_, np.nan), where=count > 0)
    sq = uniform_filter(filled ** 2, size=size, mode="reflect") * (size ** 2)
    var = np.divide(sq, count, out=np.full_like(sum_, np.nan), where=count > 0) - mean ** 2
    var[var < 0] = 0
    std = np.sqrt(var)
    return mean, std, count


def _rolling_percentile(data: np.ndarray, size: int, percentile: float) -> np.ndarray:
    pad = size // 2
    padded = np.pad(data, pad_width=pad, mode="reflect")
    h, w = data.shape
    result = np.empty((h, w), dtype=data.dtype)
    for i in range(h):
        window_rows = padded[i : i + size]
        windows = np.lib.stride_tricks.sliding_window_view(window_rows, size, axis=1)
        flattened = windows.reshape(size * size, w)
        result[i] = np.nanpercentile(flattened, percentile, axis=0)
    return result


def _gradient_magnitude(data: np.ndarray) -> np.ndarray:
    gx = sobel(data, axis=1, mode="reflect")
    gy = sobel(data, axis=0, mode="reflect")
    return np.sqrt(gx ** 2 + gy ** 2)


def _laplacian(data: np.ndarray) -> np.ndarray:
    return laplace(data, mode="reflect")


class RollingFeatureGenerator:
    def __init__(
        self,
        scales: Sequence[int],
        schema_path: Path,
        output_dir: Path,
        nodata: float = -9999.0,
    ) -> None:
        self.scales = list(scales)
        self.schema_path = schema_path
        self.output_dir = output_dir
        self.nodata = nodata
        self._band_names: Optional[List[str]] = None

    def _process_layer(self, base_name: str, data: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        arrays: List[np.ndarray] = []
        names: List[str] = []
        is_context = base_name.startswith("dist_")
        if is_context:
            arrays.append(data)
            names.append(base_name)
            return arrays, names
        mask = np.isnan(data)
        for scale in self.scales:
            mean, std, count = _rolling_mean_std(data, scale)
            p10 = _rolling_percentile(data, scale, 10)
            p50 = _rolling_percentile(data, scale, 50)
            p90 = _rolling_percentile(data, scale, 90)
            smooth = np.nan_to_num(mean, nan=0.0)
            grad = _gradient_magnitude(smooth)
            lap = _laplacian(smooth)
            grad[mask] = np.nan
            lap[mask] = np.nan
            z = np.divide(
                data - mean,
                std,
                out=np.zeros_like(data),
                where=(std > 0) & ~np.isnan(data),
            )
            arrays.extend([mean, std, p10, p50, p90, grad, lap, z])
            names.extend(
                [
                    f"{base_name}_s{scale}_mean",
                    f"{base_name}_s{scale}_std",
                    f"{base_name}_s{scale}_p10",
                    f"{base_name}_s{scale}_p50",
                    f"{base_name}_s{scale}_p90",
                    f"{base_name}_s{scale}_grad",
                    f"{base_name}_s{scale}_lap",
                    f"{base_name}_s{scale}_z",
                ]
            )
        return arrays, names

    def _finalize(self, arrays: List[np.ndarray]) -> np.ndarray:
        stack = np.stack(arrays, axis=0).astype("float32")
        stack = np.where(np.isnan(stack), self.nodata, stack)
        return stack

    def _update_schema(self, names: List[str]) -> None:
        if self._band_names is None:
            schema = ensure_schema(self.schema_path, names)
            self._band_names = schema.bands
        else:
            ensure_schema(self.schema_path, names)

    def process_tile(
        self,
        zone_epsg: str,
        tile: TileDefinition,
        datasets: Dict[str, rasterio.DatasetReader],
    ) -> Path:
        arrays: List[np.ndarray] = []
        names: List[str] = []
        for name, ds in datasets.items():
            window = from_bounds(tile.x_min, tile.y_min, tile.x_max, tile.y_max, ds.transform)
            data = ds.read(1, window=window, boundless=True, fill_value=self.nodata).astype("float32")
            nodata_values = {self.nodata}
            if ds.nodata is not None:
                nodata_values.add(float(ds.nodata))
            for nv in nodata_values:
                data[data == nv] = np.nan
            layer_arrays, layer_names = self._process_layer(name, data)
            arrays.extend(layer_arrays)
            names.extend(layer_names)
        self._update_schema(names)
        stack = self._finalize(arrays)
        profile = {
            "driver": "GTiff",
            "height": stack.shape[1],
            "width": stack.shape[2],
            "count": stack.shape[0],
            "dtype": "float32",
            "transform": tile.affine_transform,
            "crs": f"EPSG:{zone_epsg}",
            "nodata": self.nodata,
        }
        memfile = rasterio.io.MemoryFile()
        with memfile.open(**profile) as dataset:
            dataset.write(stack)
            for idx, band_name in enumerate(names, start=1):
                dataset.update_tags(idx, name=band_name)
        dst_dir = self.output_dir / zone_epsg
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{tile.tile_id}.tif"
        with memfile.open() as dataset:
            write_cog(dataset, dst_path)
        memfile.close()
        LOGGER.info("Wrote features %s", dst_path)
        return dst_path


def load_zone_datasets(zone_dir: Path) -> Dict[str, rasterio.DatasetReader]:
    datasets: Dict[str, rasterio.DatasetReader] = {}
    for path in sorted(zone_dir.glob("*.tif")):
        ds = rasterio.open(path)
        name = ds.tags().get("name") or path.stem
        datasets[name] = ds
    if not datasets:
        raise FileNotFoundError(f"No rasters found in {zone_dir}")
    return datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-scale rolling features")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute feature generation")
    run_parser.add_argument("--tiling", type=Path, required=True)
    run_parser.add_argument("--schema", type=Path, required=True)
    run_parser.add_argument("--input", type=Path, default=Path("data/interim"))
    run_parser.add_argument("--output", type=Path, default=Path("data/features"))
    run_parser.add_argument("--scales", nargs="*", type=int, default=[3, 15, 61])
    run_parser.add_argument("--zones", nargs="*")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        grid = ZoneGrid(args.tiling)
        generator = RollingFeatureGenerator(args.scales, args.schema, args.output, nodata=grid.nodata)
        zone_list = args.zones if args.zones else list(grid.zones.keys())
        for zone_epsg in zone_list:
            zone_dir = args.input / zone_epsg
            datasets = load_zone_datasets(zone_dir)
            for tile in grid.iter_tiles(zone_epsg):
                generator.process_tile(zone_epsg, tile, datasets)
            for ds in datasets.values():
                ds.close()


if __name__ == "__main__":
    main()
