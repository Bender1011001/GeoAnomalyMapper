"""Multi-resolution fusion driver."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import rasterio
import yaml

from .weight_calculator import resolution_weighting
from ..io.cogs import write_cog
from ..utils.logging import get_logger
from utils.config import ConfigManager

LOGGER = get_logger(__name__)


def load_config(path: Path) -> Dict:
    return yaml.safe_load(Path(path).read_text())


def fuse_layers(
    layers: Dict[str, Dict[str, Any]],
    output_path: Path,
    *,
    dynamic: bool = True,
    temperature: float = 1.0,
) -> Path:
    if not layers:
        raise ValueError("No layers provided for fusion")

    if dynamic:
        resolutions = {name: cfg["resolution"] for name, cfg in layers.items()}
        weight_result = resolution_weighting(resolutions, temperature=temperature)
        weights = weight_result.weights
    else:
        uniform_weight = 1.0 / len(layers)
        weights = {name: uniform_weight for name in layers}

    datasets = {name: rasterio.open(Path(cfg["path"])) for name, cfg in layers.items()}
    try:
        sample = next(iter(datasets.values()))
        profile = sample.profile.copy()
        stack = np.zeros((sample.height, sample.width), dtype="float32")
        weight_sum = 0.0
        for name, dataset in datasets.items():
            data = dataset.read(1).astype("float32")
            stack += data * weights[name]
            weight_sum += weights[name]
        stack /= weight_sum
        profile.update({"dtype": "float32", "count": 1})
        memfile = rasterio.io.MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(stack, 1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with memfile.open() as dataset:
            write_cog(dataset, output_path)
        memfile.close()
        LOGGER.info(
            "Wrote fused raster %s (dynamic=%s, temperature=%.3f, weights=%s)",
            output_path,
            dynamic,
            temperature,
            weights,
        )
        return output_path
    finally:
        for dataset in datasets.values():
            dataset.close()


def run(config_path: Path, output_dir: Path, *, dynamic: bool, temperature: float) -> None:
    config = load_config(config_path)
    for product in config.get("products", []):
        name = product["name"]
        layers = {layer["name"]: layer for layer in product["layers"]}
        output_path = output_dir / f"{name}.tif"
        fuse_layers(layers, output_path, dynamic=dynamic, temperature=temperature)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse rasters using weighted average")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run fusion")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, default=Path("data/fused"))
    mode_group = run_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dynamic", dest="dynamic", action="store_true")
    mode_group.add_argument("--static", dest="dynamic", action="store_false")
    run_parser.set_defaults(dynamic=None)
    run_parser.add_argument("--temperature", type=float, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = ConfigManager()
        config_dynamic = cfg.get("fusion.dynamic_weighting", True)
        config_temperature = cfg.get("fusion.temperature", 1.0)
        dynamic = config_dynamic if args.dynamic is None else args.dynamic
        temperature = args.temperature if args.temperature is not None else config_temperature
        run(args.config, args.output, dynamic=dynamic, temperature=temperature)


if __name__ == "__main__":
    main()
