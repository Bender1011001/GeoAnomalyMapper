"""FastAPI service for GeoAnomalyMapper."""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..features.schema import FeatureSchema
from ..io.catalog import CatalogClient
from ..models.infer_tiles import load_model
from ..io.cogs import write_cog
from ..utils.logging import configure_logging, get_logger

configure_logging()
LOGGER = get_logger(__name__)

DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "stac" / "catalog.json"
MODEL_PATH = Path("artifacts/selected_model.pkl")
SCHEMA_PATH = Path("data/feature_schema.json")
OUTPUT_DIR = Path("api_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PointRequest(BaseModel):
    lon: float = Field(..., description="Longitude in degrees")
    lat: float = Field(..., description="Latitude in degrees")


class BBoxResponse(BaseModel):
    width: int
    height: int
    min: float
    max: float
    path: str


class ApiState:
    def __init__(self) -> None:
        if not CATALOG_PATH.exists():
            raise RuntimeError(f"STAC catalog not found at {CATALOG_PATH}")
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        if not SCHEMA_PATH.exists():
            raise RuntimeError(f"Feature schema not found at {SCHEMA_PATH}")
        self.catalog = CatalogClient(CATALOG_PATH)
        self.model = load_model(MODEL_PATH)
        self.schema = FeatureSchema.from_file(SCHEMA_PATH)

    @lru_cache(maxsize=32)
    def _open_dataset(self, href: str) -> rasterio.io.DatasetReader:
        return rasterio.open(href)

    def sample_features(self, lon: float, lat: float) -> np.ndarray:
        asset = self.catalog.locate_tile(lon, lat, role="feature")
        dataset = self._open_dataset(asset.href)
        row, col = dataset.index(lon, lat)
        if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
            raise HTTPException(status_code=404, detail="Point outside feature tile")
        values = dataset.read(window=rasterio.windows.Window(col, row, 1, 1))[:, 0, 0]
        band_names = [dataset.tags(i).get("name") or f"band_{i}" for i in range(1, dataset.count + 1)]
        if band_names != self.schema.bands:
            raise HTTPException(status_code=500, detail="Feature schema mismatch")
        if dataset.nodata is not None:
            mask = values == dataset.nodata
            if mask.any():
                raise HTTPException(status_code=422, detail="Features contain nodata")
        return values.astype("float32")

    def predict_points(self, points: List[PointRequest]) -> Dict[str, List[Dict[str, float]]]:
        results: List[Dict[str, float]] = []
        for point in points:
            features = self.sample_features(point.lon, point.lat)
            proba = float(self.model.predict_proba(features.reshape(1, -1))[:, 1][0])
            results.append({"lon": point.lon, "lat": point.lat, "p": proba})
        return {"results": results}

    def predict_bbox(self, minx: float, miny: float, maxx: float, maxy: float, res_m: int) -> BBoxResponse:
        assets = self.catalog.tiles_for_bbox((minx, miny, maxx, maxy), role="feature")
        if not assets:
            raise HTTPException(status_code=404, detail="No tiles cover requested bbox")
        stack, transform, crs, nodata = self.catalog.load_feature_stack(assets, (minx, miny, maxx, maxy))
        mask = np.any(stack == nodata, axis=2)
        flat = stack.reshape(-1, stack.shape[-1])
        preds = np.full(flat.shape[0], np.nan, dtype="float32")
        valid_idx = ~mask.ravel()
        if np.any(valid_idx):
            preds[valid_idx] = self.model.predict_proba(flat[valid_idx])[:, 1]
        grid = preds.reshape(stack.shape[0], stack.shape[1])
        grid = np.where(np.isnan(grid), nodata, grid).astype("float32")
        profile = {
            "driver": "GTiff",
            "height": grid.shape[0],
            "width": grid.shape[1],
            "count": 1,
            "dtype": "float32",
            "transform": transform,
            "crs": crs,
            "nodata": nodata,
        }
        memfile = rasterio.io.MemoryFile()
        with memfile.open(**profile) as dataset:
            dataset.write(grid, 1)
            dataset.update_tags(model="GeoAnomalyMapper", role="probability")
        timestamp = int(time.time())
        out_path = OUTPUT_DIR / f"bbox_{timestamp}.tif"
        with memfile.open() as dataset:
            write_cog(dataset, out_path)
        memfile.close()
        valid_mask = grid > nodata
        return BBoxResponse(
            width=int(grid.shape[1]),
            height=int(grid.shape[0]),
            min=float(np.nanmin(grid[valid_mask])) if np.any(valid_mask) else float(nodata),
            max=float(np.nanmax(grid[valid_mask])) if np.any(valid_mask) else float(nodata),
            path=str(out_path),
        )


state = ApiState()
app = FastAPI(title="GeoAnomalyMapper API", version="3.0")


@app.post("/predict/points")
def predict_points(points: List[PointRequest]):
    return state.predict_points(points)


@app.get("/predict/bbox", response_model=BBoxResponse)
def predict_bbox(minx: float, miny: float, maxx: float, maxy: float, res_m: int = 1000):
    return state.predict_bbox(minx, miny, maxx, maxy, res_m)
