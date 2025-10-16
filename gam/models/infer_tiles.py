"""Tile-based inference from feature rasters."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import rasterio

from ..features.schema import FeatureSchema
from ..io.cogs import write_cog
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_model(model_path: Path):
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not implement predict_proba")
    return model


def read_feature_raster(path: Path) -> tuple[np.ndarray, dict, float]:
    with rasterio.open(path) as ds:
        data = ds.read().astype("float32")
        nodata = ds.nodata if ds.nodata is not None else -9999.0
        meta = {
            "transform": ds.transform,
            "crs": ds.crs,
            "height": ds.height,
            "width": ds.width,
        }
    return data, meta, nodata


def run_inference_on_tile(
    feature_path: Path,
    model,
    schema: FeatureSchema,
    output_dir: Path,
) -> Path:
    data, meta, nodata = read_feature_raster(feature_path)
    feature_order = [schema_band for schema_band in schema.bands]
    band_names = []
    with rasterio.open(feature_path) as ds:
        for idx in range(1, ds.count + 1):
            band_names.append(ds.tags(idx).get("name") or f"band_{idx}")
    if band_names != feature_order:
        raise ValueError("Feature band order does not match schema")
    stack = np.moveaxis(data, 0, -1)
    flat = stack.reshape(-1, stack.shape[-1])
    mask = np.any(data == nodata, axis=0).ravel()
    preds = np.full(flat.shape[0], np.nan, dtype="float32")
    valid_idx = ~mask
    if np.any(valid_idx):
        preds[valid_idx] = model.predict_proba(flat[valid_idx])[:, 1]
    grid = preds.reshape(meta["height"], meta["width"])
    grid = np.where(np.isnan(grid), -9999.0, grid).astype("float32")
    profile = {
        "driver": "GTiff",
        "height": meta["height"],
        "width": meta["width"],
        "count": 1,
        "dtype": "float32",
        "transform": meta["transform"],
        "crs": meta["crs"],
        "nodata": -9999.0,
    }
    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(grid, 1)
        dataset.update_tags(model="GeoAnomalyMapper", role="probability")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = output_dir / feature_path.name.replace(".tif", "_proba.tif")
    with memfile.open() as dataset:
        write_cog(dataset, dst_path)
    memfile.close()
    LOGGER.info("Wrote probabilities %s", dst_path)
    return dst_path


def run_inference(
    features_dir: Path,
    model_path: Path,
    schema_path: Path,
    output_dir: Path,
    zones: Optional[Sequence[str]] = None,
) -> None:
    model = load_model(model_path)
    schema = FeatureSchema.from_file(schema_path)
    zone_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
    for zone_dir in zone_dirs:
        zone = zone_dir.name
        if zones and zone not in zones:
            continue
        for feature_path in sorted(zone_dir.glob("*.tif")):
            run_inference_on_tile(feature_path, model, schema, output_dir / zone)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference over feature tiles")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute inference")
    run_parser.add_argument("--features", type=Path, default=Path("data/features"))
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--schema", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, default=Path("data/products"))
    run_parser.add_argument("--zones", nargs="*")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run_inference(args.features, args.model, args.schema, args.output, args.zones)


if __name__ == "__main__":
    main()
