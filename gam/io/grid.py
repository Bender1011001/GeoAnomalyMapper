"""UTM zone grid utilities for GeoAnomalyMapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import yaml
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class ZoneConfig:
    epsg: str
    name: str
    bbox4326: Tuple[float, float, float, float]


@dataclass(frozen=True)
class TileDefinition:
    zone_epsg: str
    tile_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    pixel_size: float
    tile_size_px: int
    overlap_px: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def affine_transform(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.pixel_size,
            0.0,
            self.x_min,
            0.0,
            -self.pixel_size,
            self.y_max,
        )


class ZoneGrid:
    """Helper class for enumerating tiles in a zone."""

    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf8") as stream:
            raw = yaml.safe_load(stream)
        self.pixel_size = float(raw["pixel_size_m"])
        self.tile_size_px = int(raw["tile_size_px"])
        self.overlap_px = int(raw.get("overlap_px", 0))
        self.nodata = raw.get("nodata", -9999)
        self.zones: Dict[str, ZoneConfig] = {
            epsg: ZoneConfig(epsg, entry["name"], tuple(entry["bbox4326"]))
            for epsg, entry in raw["zones"].items()
        }

    def iter_tiles(self, zone_epsg: str) -> Iterator[TileDefinition]:
        zone = self.zones[zone_epsg]
        crs = CRS.from_epsg(int(zone_epsg))
        transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
        minx, miny, maxx, maxy = zone.bbox4326
        xs = np.array([minx, maxx, maxx, minx])
        ys = np.array([miny, miny, maxy, maxy])
        utm_x, utm_y = transformer.transform(xs, ys)
        x_min, x_max = float(np.min(utm_x)), float(np.max(utm_x))
        y_min, y_max = float(np.min(utm_y)), float(np.max(utm_y))
        tile_size_m = self.pixel_size * self.tile_size_px
        step = tile_size_m - self.overlap_px * self.pixel_size
        x = x_min
        tile_index = 0
        while x < x_max:
            y = y_min
            while y < y_max:
                tile_id = f"{zone.name}_{tile_index:05d}"
                tile_index += 1
                yield TileDefinition(
                    zone_epsg=zone_epsg,
                    tile_id=tile_id,
                    x_min=x,
                    y_min=y,
                    x_max=min(x + tile_size_m, x_max),
                    y_max=min(y + tile_size_m, y_max),
                    pixel_size=self.pixel_size,
                    tile_size_px=self.tile_size_px,
                    overlap_px=self.overlap_px,
                )
                y += step
            x += step

    def enumerate_all_tiles(self) -> List[TileDefinition]:
        tiles: List[TileDefinition] = []
        for epsg in self.zones:
            tiles.extend(list(self.iter_tiles(epsg)))
        return tiles


__all__ = ["ZoneConfig", "TileDefinition", "ZoneGrid"]
