"""Extract point features from feature rasters."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import rasterio

from ..io.catalog import CatalogClient
from ..utils.logging import get_logger
from .schema import FeatureSchema

LOGGER = get_logger(__name__)


@dataclass
class LabeledPoint:
    lon: float
    lat: float
    label: int


def load_points(path: Path) -> List[LabeledPoint]:
    df = pd.read_csv(path)
    required = {"lon", "lat", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Point file must contain columns {required}")
    return [LabeledPoint(float(row.lon), float(row.lat), int(row.label)) for row in df.itertuples()]


def sample_feature_vector(asset_path: Path, lon: float, lat: float) -> np.ndarray:
    with rasterio.open(asset_path) as src:
        row, col = src.index(lon, lat)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            raise ValueError(f"Point ({lon}, {lat}) falls outside raster extent {asset_path}")
        data = src.read(window=rasterio.windows.Window(col, row, 1, 1), boundless=False)
        return data[:, 0, 0]


def extract_features(
    catalog_path: Path,
    schema_path: Path,
    points_path: Path,
    role: str = "feature",
) -> pd.DataFrame:
    client = CatalogClient(catalog_path)
    schema = FeatureSchema.from_file(schema_path)
    points = load_points(points_path)
    records = []
    for point in points:
        asset = client.locate_tile(point.lon, point.lat, role=role)
        values = sample_feature_vector(Path(asset.href), point.lon, point.lat)
        record = {"lon": point.lon, "lat": point.lat, "label": point.label}
        for idx, band_name in enumerate(schema.bands):
            record[band_name] = float(values[idx])
        records.append(record)
    return pd.DataFrame(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract features at labeled points")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True, help="CSV file with lon, lat, label")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV for model training")
    parser.add_argument("--role", default="feature")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    df = extract_features(args.catalog, args.schema, args.points, args.role)
    df.to_csv(args.out, index=False)
    LOGGER.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
