"""Post-processing of probability rasters into vector polygons."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mlflow
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.segmentation import watershed
import fiona

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def resolve_threshold(threshold: str, mlflow_experiment: Optional[str] = None) -> float:
    if not threshold.startswith("from:mlflow"):
        return float(threshold)
    client = mlflow.tracking.MlflowClient()
    experiment_id = None
    if mlflow_experiment:
        experiment = client.get_experiment_by_name(mlflow_experiment)
        if experiment:
            experiment_id = experiment.experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id] if experiment_id else None,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        LOGGER.warning("No MLflow runs found; defaulting threshold to 0.5")
        return 0.5
    run = runs[0]
    params = run.data.params
    if "operating_threshold" in params:
        return float(params["operating_threshold"])
    metrics = run.data.metrics
    for key in ("operating_threshold", "recall_at_1fpr_threshold"):
        if key in metrics:
            return float(metrics[key])
    LOGGER.warning("MLflow run missing operating_threshold; defaulting to 0.5")
    return 0.5


def segment(prob: np.ndarray, thr: float, sigma: float, min_distance: int) -> np.ndarray:
    mask = prob >= thr
    if not np.any(mask):
        return np.zeros_like(prob, dtype=np.int32)
    smoothed = gaussian(prob, sigma=sigma, preserve_range=True)
    coords = peak_local_max(smoothed, labels=mask, min_distance=min_distance)
    markers = np.zeros_like(prob, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labels = watershed(-smoothed, markers, mask=mask)
    return labels.astype(np.int32)


def raster_to_polygons(labels: np.ndarray, prob: np.ndarray, transform, zone: str, tile: str) -> List[Dict]:
    results: List[Dict] = []
    if labels.max() == 0:
        return results
    for geom, value in shapes(labels, mask=labels > 0, transform=transform):
        if value == 0:
            continue
        polygon = shape(geom)
        mask = labels == value
        max_prob = float(prob[mask].max())
        area_km2 = float(polygon.area / 1_000_000.0)
        results.append(
            {
                "geometry": polygon,
                "properties": {
                    "zone": zone,
                    "tile": tile,
                    "max_proba": max_prob,
                    "area_km2": area_km2,
                },
            }
        )
    return results


def write_geojson(features: List[Dict], crs, output_path: Path) -> None:
    if not features:
        LOGGER.info("No features found for %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "geometry": "Polygon",
        "properties": {
            "zone": "str",
            "tile": "str",
            "max_proba": "float",
            "area_km2": "float",
        },
    }
    with fiona.open(
        output_path,
        "w",
        driver="GeoJSON",
        schema=schema,
        crs=crs,
    ) as dst:
        for feature in features:
            dst.write({"geometry": mapping(feature["geometry"]), "properties": feature["properties"]})
    LOGGER.info("Wrote vectors %s", output_path)


def process_probability_raster(
    path: Path,
    threshold: float,
    sigma: float,
    min_distance: int,
    output_dir: Path,
) -> None:
    with rasterio.open(path) as src:
        prob = src.read(1)
        labels = segment(prob, threshold, sigma, min_distance)
        zone = path.parent.name
        tile = path.stem.replace("_proba", "")
        features = raster_to_polygons(labels, prob, src.transform, zone, tile)
        out_path = output_dir / zone / f"{tile}_vectors.geojson"
        write_geojson(features, src.crs, out_path)


def run_postprocess(
    probabilities_dir: Path,
    output_dir: Path,
    threshold: float,
    sigma: float,
    min_distance: int,
    zones: Optional[Sequence[str]] = None,
) -> None:
    for zone_dir in probabilities_dir.iterdir():
        if not zone_dir.is_dir():
            continue
        if zones and zone_dir.name not in zones:
            continue
        for path in sorted(zone_dir.glob("*_proba.tif")):
            process_probability_raster(path, threshold, sigma, min_distance, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vectorize probability rasters using watershed")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute post-processing")
    run_parser.add_argument("--probabilities", type=Path, default=Path("data/products"))
    run_parser.add_argument("--output", type=Path, default=Path("data/products/vectors"))
    run_parser.add_argument("--threshold", required=True)
    run_parser.add_argument("--sigma", type=float, default=1.0)
    run_parser.add_argument("--min-distance", type=int, default=5)
    run_parser.add_argument("--mlflow-experiment", type=str, default=None)
    run_parser.add_argument("--zones", nargs="*")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        threshold = resolve_threshold(args.threshold, args.mlflow_experiment)
        run_postprocess(args.probabilities, args.output, threshold, args.sigma, args.min_distance, args.zones)


if __name__ == "__main__":
    main()
