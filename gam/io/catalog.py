"""STAC catalog client utilities."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pystac
import rasterio
from pyproj import Transformer
from affine import Affine
from rasterio.merge import merge
from shapely.geometry import box
from pystac.extensions.projection import ProjectionExtension

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AssetRecord:
    item: pystac.Item
    asset: pystac.Asset

    @property
    def href(self) -> str:
        return self.asset.get_absolute_href() or self.asset.href

    @property
    def role(self) -> str:
        return ",".join(self.asset.roles or [])

    @property
    def zone_epsg(self) -> Optional[int]:
        proj = ProjectionExtension.ext(self.item, add_if_missing=False)
        if proj and proj.epsg:
            return proj.epsg
        return None


class CatalogClient:
    def __init__(self, catalog_root: Path):
        self.catalog_root = Path(catalog_root)
        catalog_path = self.catalog_root
        if catalog_path.is_dir():
            catalog_path = catalog_path / "catalog.json"
        self.catalog = pystac.Catalog.from_file(str(catalog_path))

    def _iter_items(self) -> Iterator[pystac.Item]:
        yield from self.catalog.get_all_items()

    def find_assets(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        time: Optional[Tuple[str, str]] = None,
        role: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> List[AssetRecord]:
        bbox_geom = box(*bbox) if bbox else None
        results: List[AssetRecord] = []
        for item in self._iter_items():
            if time and item.datetime:
                start, end = time
                if start and item.datetime.isoformat() < start:
                    continue
                if end and item.datetime.isoformat() > end:
                    continue
            if bbox_geom and not bbox_geom.intersects(box(*item.bbox)):
                continue
            proj = pystac.extensions.projection.ProjectionExtension.ext(item, add_if_missing=False)
            zone_epsg = str(proj.epsg) if proj and proj.epsg else None
            if zone and zone_epsg != zone:
                continue
            for key, asset in item.assets.items():
                if role and role not in (asset.roles or []):
                    continue
                results.append(AssetRecord(item=item, asset=asset))
        return results

    def latest_product(self, role: str, zone: Optional[str] = None) -> Optional[AssetRecord]:
        candidates = self.find_assets(role=role, zone=zone)
        if not candidates:
            return None
        return max(candidates, key=lambda rec: rec.item.datetime or dt.datetime.min)  # type: ignore[name-defined]

    def locate_tile(self, lon: float, lat: float, role: str = "feature") -> AssetRecord:
        point_box = (lon, lat, lon, lat)
        assets = self.find_assets(bbox=point_box, role=role)
        if not assets:
            raise ValueError(f"No assets found covering point ({lon}, {lat})")
        if len(assets) == 1:
            return assets[0]
        # Prefer highest resolution (assume smaller pixel size -> more rows)
        def resolution(rec: AssetRecord) -> float:
            with rasterio.open(rec.href) as src:
                return abs(src.transform.a)
        return min(assets, key=resolution)

    def tiles_for_bbox(self, bbox: Tuple[float, float, float, float], role: str = "feature") -> List[AssetRecord]:
        return self.find_assets(bbox=bbox, role=role)

    def load_feature_stack(
        self,
        tiles: Sequence[AssetRecord],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Affine, rasterio.crs.CRS, float]:
        if not tiles:
            raise ValueError("No tiles provided")
        datasets = [rasterio.open(tile.href) for tile in tiles]
        try:
            ref_crs = datasets[0].crs
            nodata = datasets[0].nodata if datasets[0].nodata is not None else -9999.0
            transformer = Transformer.from_crs("epsg:4326", ref_crs, always_xy=True)
            x1, y1 = transformer.transform(bbox[0], bbox[1])
            x2, y2 = transformer.transform(bbox[2], bbox[3])
            target_bounds = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            merged, transform = merge(datasets, bounds=target_bounds, nodata=nodata)
            return merged.transpose(1, 2, 0), transform, ref_crs, nodata
        finally:
            for ds in datasets:
                ds.close()


__all__ = ["CatalogClient", "AssetRecord"]
