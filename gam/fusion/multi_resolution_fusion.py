"""Multi-resolution fusion driver."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import rasterio
import yaml

from .weight_calculator import physics_weighting
from ..io.cogs import write_cog
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_config(path: Path) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def _open_dataset(layer_cfg: Dict[str, Any]) -> rasterio.DatasetReader:
    dataset = rasterio.open(Path(layer_cfg["path"]))
    if dataset.count != 1:
        raise ValueError(f"Layer '{layer_cfg['name']}' must be single-band GeoTIFF")
    return dataset


def _harmonise_profile(datasets: Dict[str, rasterio.DatasetReader]) -> Dict[str, Any]:
    reference = next(iter(datasets.values()))
    profile = reference.profile.copy()
    profile.update({"dtype": "float32", "count": 1, "nodata": np.float32(np.nan)})
    height, width = reference.shape
    profile["height"] = height
    profile["width"] = width
    return profile


def _read_band(dataset: rasterio.DatasetReader) -> np.ma.MaskedArray:
    data = dataset.read(1, masked=True).astype("float32")
    if data.mask.all():
        raise ValueError(f"Dataset {dataset.name} contains only nodata values")
    return data


def fuse_layers(layers: Dict[str, Dict[str, Any]], output_path: Path) -> Path:
    weight_result = physics_weighting(layers)
    datasets = {name: _open_dataset(cfg) for name, cfg in layers.items()}
    try:
        weights = weight_result.weights
        profile = _harmonise_profile(datasets)
        height = profile["height"]
        width = profile["width"]
        accum = np.zeros((height, width), dtype="float32")
        weight_sum = np.zeros((height, width), dtype="float32")

        for name, dataset in datasets.items():
            band = _read_band(dataset)
            weight = np.float32(weights[name])
            mask = np.ma.getmaskarray(band)
            data = band.filled(0.0)
            accum += data * weight
            weight_sum += weight * (~mask).astype("float32")

        fused = np.full((height, width), np.nan, dtype="float32")
        valid = weight_sum > 0
        fused[valid] = accum[valid] / weight_sum[valid]

        memfile = rasterio.io.MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(fused.astype("float32"), 1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with memfile.open() as dataset:
            write_cog(dataset, output_path)
        memfile.close()
        LOGGER.info("Wrote fused raster %s", output_path)
        LOGGER.info("Layer weights: %s", ", ".join(f"%s=%.3f" % item for item in weights.items()))
        return output_path
    finally:
        for dataset in datasets.values():
            dataset.close()


def run(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    for product in config.get("products", []):
        name = product["name"]
        layers = {layer["name"]: layer for layer in product["layers"]}
        output_path = output_dir / f"{name}.tif"
        fuse_layers(layers, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse rasters using weighted average")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run fusion")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, default=Path("data/fused"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.config, args.output)


if __name__ == "__main__":
    main()
