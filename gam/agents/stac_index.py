"""STAC catalog management for GeoAnomalyMapper."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

import rasterio
import pystac
from shapely.geometry import Polygon, mapping
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension
from pystac import Link

from ..utils.hashing import sha256_path
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class StacIndex:
    """Wrapper around pystac for simplified catalog authoring."""

    def __init__(self, catalog_root: Path):
        self.catalog_root = catalog_root
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self._catalog: Optional[pystac.Catalog] = None

    def init_catalog(self) -> pystac.Catalog:
        catalog = pystac.Catalog(
            id="geoanomalymapper",
            description="GeoAnomalyMapper data products",
            title="GeoAnomalyMapper",
        )
        catalog.normalize_and_save(self.catalog_root, catalog_type=pystac.CatalogType.SELF_CONTAINED)
        LOGGER.info("Initialized STAC catalog at %s", self.catalog_root)
        self._catalog = catalog
        return catalog

    def load(self) -> pystac.Catalog:
        if self._catalog is None:
            catalog_path = self.catalog_root / "catalog.json"
            if not catalog_path.exists():
                raise FileNotFoundError(f"Catalog not initialized at {catalog_path}")
            self._catalog = pystac.Catalog.from_file(str(catalog_path))
        return self._catalog

    def ensure_collection(self, collection_id: str, title: str, description: str) -> pystac.Collection:
        catalog = self.load()
        collection = next((c for c in catalog.get_children() if c.id == collection_id), None)
        if collection is None:
            extent = pystac.Extent(
                spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                temporal=pystac.TemporalExtent([[dt.datetime(1970, 1, 1), None]]),
            )
            collection = pystac.Collection(
                id=collection_id,
                title=title,
                description=description,
                extent=extent,
            )
            catalog.add_child(collection)
            catalog.normalize_hrefs(str(self.catalog_root))
            catalog.save()
            LOGGER.info("Created collection %s", collection_id)
        return collection

    def register_raster(
        self,
        asset_path: Path,
        collection_id: str,
        asset_role: str,
        properties: Optional[Dict[str, object]] = None,
        derived_from: Optional[List[str]] = None,
    ) -> pystac.Item:
        collection = self.ensure_collection(collection_id, collection_id, f"Collection for {collection_id}")
        asset_path = asset_path.resolve()
        props = dict(properties or {})
        props.setdefault("processing:timestamp", dt.datetime.utcnow().isoformat())
        props.setdefault("processing:role", asset_role)
        checksum = sha256_path(asset_path)
        props["checksum:multihash"] = checksum

        with rasterio.open(asset_path) as src:
            bounds = src.bounds
            geom = Polygon(
                [
                    (bounds.left, bounds.bottom),
                    (bounds.right, bounds.bottom),
                    (bounds.right, bounds.top),
                    (bounds.left, bounds.top),
                    (bounds.left, bounds.bottom),
                ]
            )
            item = pystac.Item(
                id=f"{asset_role}-{asset_path.stem}",
                geometry=mapping(geom),
                bbox=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                datetime=dt.datetime.utcnow(),
                properties=props,
            )
            ProjectionExtension.add_to(item)
            RasterExtension.add_to(item)
            ProjectionExtension.ext(item).apply(crs=src.crs, transform=src.transform, shape=src.shape)
            bands = [
                {
                    "nodata": src.nodata,
                    "data_type": src.dtypes[idx - 1],
                    "sampling": "area",
                }
                for idx in range(1, src.count + 1)
            ]
            RasterExtension.ext(item).bands = bands
            asset = pystac.Asset(
                href=str(asset_path),
                media_type=pystac.MediaType.GEOTIFF,
                roles=[asset_role],
                extra_fields={"checksum:multihash": checksum},
            )
            item.add_asset(asset_role, asset)
        if derived_from:
            for href in derived_from:
            item.add_link(Link(rel="derived_from", target=href))
        collection.add_item(item, strategy=pystac.RelinkStrategy.REF_RESOLVE)
        collection.normalize_hrefs(str(self.catalog_root))
        collection.save()
        LOGGER.info("Registered %s under %s", asset_path, collection_id)
        return item


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the GeoAnomalyMapper STAC catalog")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Create a fresh catalog")
    init_parser.add_argument("--out", type=Path, required=True, help="Output directory for the catalog")

    reg_parser = sub.add_parser("register", help="Register a raster asset")
    reg_parser.add_argument("--catalog", type=Path, required=True)
    reg_parser.add_argument("--path", type=Path, required=True)
    reg_parser.add_argument("--collection", required=True)
    reg_parser.add_argument("--role", required=True)
    reg_parser.add_argument("--properties", type=Path, help="Optional JSON file with additional properties")
    reg_parser.add_argument("--derived-from", nargs="*", dest="derived_from")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        index = StacIndex(args.out)
        index.init_catalog()
    elif args.command == "register":
        index = StacIndex(args.catalog)
        props = None
        if args.properties:
            props = json.loads(Path(args.properties).read_text())
        index.register_raster(args.path, args.collection, args.role, props, args.derived_from)


if __name__ == "__main__":
    main()
