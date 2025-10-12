#!/usr/bin/env python3
"""GeoAnomalyMapper Data Agent.

This module provides a command line "data agent" that understands the core
geophysical datasets required by the project, inspects which of them are
already present on disk, and coordinates the downloads of the ones that are
missing.  The agent wraps the existing downloader utilities in this repository,
adds status tracking, and surfaces clear manual follow-up instructions whenever
automation is not possible.

Example usage (defaults to contiguous United States):

    # Show status of all datasets
    python data_agent.py

    # Download every dataset in the high priority phases
    python data_agent.py --download all --phases 1 2

    # Override the area of interest and download Copernicus DEM only
    python data_agent.py --preset europe --download copernicus_dem
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """Geographic bounding box expressed as lon/lat extents."""

    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


DownloadCallback = Callable[["Dataset"], bool]
ExistenceCheck = Callable[["DataAgent", "Dataset"], bool]


@dataclass
class Dataset:
    """Metadata for a dataset managed by the agent."""

    key: str
    title: str
    phase: int
    description: str
    targets: Sequence[Path]
    download: Optional[DownloadCallback] = None
    manual_instructions: Optional[str] = None
    requires_credentials: Sequence[str] = field(default_factory=tuple)
    existence_check: Optional[ExistenceCheck] = None
    extras: Dict[str, object] = field(default_factory=dict)

    def normalised_key(self) -> str:
        return self.key.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Dataset registry construction
# ---------------------------------------------------------------------------


PRESET_REGIONS: Dict[str, Region] = {
    "usa_lower48": Region(lon_min=-125.0, lat_min=24.5, lon_max=-66.95, lat_max=49.5),
    "usa": Region(lon_min=-170.0, lat_min=18.0, lon_max=-65.0, lat_max=72.0),
    "europe": Region(lon_min=-11.0, lat_min=35.0, lon_max=40.0, lat_max=71.0),
    "global": Region(lon_min=-180.0, lat_min=-90.0, lon_max=180.0, lat_max=90.0),
}


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class DataAgent:
    """Coordinates discovery and download of project datasets."""

    STATUS_PATH = Path("data") / "agent_status.json"

    def __init__(
        self,
        region: Region,
        force: bool = False,
        sentinel_limit: int = 2,
        sentinel_days: int = 30,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parent
        self.region = region
        self.force = force
        self.sentinel_limit = max(0, sentinel_limit)
        self.sentinel_days = max(1, sentinel_days)
        self.datasets: Dict[str, Dataset] = {}
        self._status_data = self._load_status()
        self._build_registry()

    # ------------------------------------------------------------------
    # Status handling
    # ------------------------------------------------------------------

    def _load_status(self) -> Dict[str, object]:
        path = self.repo_root / self.STATUS_PATH
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                LOG.warning("Status file %s is corrupted; starting fresh", path)
        return {"datasets": {}, "last_run": None, "region": vars(self.region)}

    def _save_status(self) -> None:
        path = self.repo_root / self.STATUS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._status_data["last_run"] = datetime.utcnow().isoformat()
        self._status_data["region"] = vars(self.region)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._status_data, fh, indent=2)

    def record_status(self, dataset: Dataset, status: str, details: Dict[str, object]) -> None:
        LOG.debug("Recording status %s for %s", status, dataset.key)
        entry = {
            "status": status,
            "details": details,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._status_data.setdefault("datasets", {})[dataset.key] = entry
        self._save_status()

    # ------------------------------------------------------------------
    # Dataset registry
    # ------------------------------------------------------------------

    def _build_registry(self) -> None:
        root = self.repo_root

        def register(dataset: Dataset) -> None:
            key = dataset.normalised_key()
            if key in self.datasets:
                raise ValueError(f"Dataset with key '{dataset.key}' already registered")
            self.datasets[key] = dataset

        def dir_has_files(path: Path, pattern: str = "*") -> bool:
            directory = path
            return directory.exists() and any(directory.glob(pattern))

        def file_exists(path: Path) -> bool:
            return path.exists()

        register(
            Dataset(
                key="copernicus_dem",
                title="Copernicus DEM (30 m)",
                phase=1,
                description="Global 30 m digital elevation model tiles covering the AOI.",
                targets=[root / "data" / "raw" / "elevation" / "copernicus_dem"],
                download=self._download_copernicus_dem,
                existence_check=lambda agent, ds: dir_has_files(ds.targets[0], "*.tif"),
            )
        )

        register(
            Dataset(
                key="egm2008",
                title="EGM2008 Free-Air Anomaly",
                phase=1,
                description="Baseline 1 arc-minute global gravity grid used by the pipeline.",
                targets=[root / "data" / "raw" / "gravity" / "EGM2008_Free_Air_Anomaly.tif"],
                download=lambda ds, agent=self: agent._download_baseline_dataset(
                    "EGM2008_Free_Air_Anomaly.tif"
                ),
                existence_check=lambda agent, ds: file_exists(ds.targets[0]),
            )
        )

        register(
            Dataset(
                key="emag2",
                title="EMAG2 v3 Magnetic Anomaly",
                phase=1,
                description="Global magnetic anomaly grid (2 arc-minute).",
                targets=[root / "data" / "raw" / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"],
                download=lambda ds, agent=self: agent._download_baseline_dataset(
                    "EMAG2_V3_Sea_Level.tif"
                ),
                existence_check=lambda agent, ds: file_exists(ds.targets[0]),
            )
        )

        register(
            Dataset(
                key="sentinel1",
                title="Sentinel-1 InSAR (recent archive)",
                phase=2,
                description=(
                    "Downloads the most recent Sentinel-1 SLC products intersecting the"
                    " AOI. Requires Copernicus Data Space credentials."
                ),
                targets=[root / "data" / "raw" / "insar" / "sentinel1"],
                download=self._download_sentinel1,
                existence_check=lambda agent, ds: dir_has_files(ds.targets[0], "*.zip"),
                extras={
                    "limit": self.sentinel_limit,
                    "days": self.sentinel_days,
                    "product_type": "SLC",
                },
            )
        )

        register(
            Dataset(
                key="xgm2019e",
                title="XGM2019e High-Resolution Gravity",
                phase=2,
                description=(
                    "Highest resolution global gravity model available for free."
                    " Requires manual download from the ICGEM service."
                ),
                targets=[root / "data" / "gravity" / "global" / "XGM2019e"],
                manual_instructions=(
                    "Download via ICGEM grid calculator: http://icgem.gfz-potsdam.de/calcgrid\n"
                    "Model: XGM2019e_2159, Grid step: 0.02°, Quantity: gravity disturbance.\n"
                    "Save the resulting GeoTIFF into data/gravity/global/XGM2019e/."
                ),
                existence_check=lambda agent, ds: dir_has_files(ds.targets[0], "*.tif"),
            )
        )

        register(
            Dataset(
                key="sentinel2",
                title="Sentinel-2 Optical Imagery",
                phase=3,
                description=(
                    "Multispectral optical imagery from Sentinel-2 (10 m resolution)."
                    " Use the Copernicus Data Space or AWS Open Data program."
                ),
                targets=[root / "data" / "raw" / "optical" / "sentinel2"],
                manual_instructions=(
                    "Use the dataspace.copernicus.eu catalogue or AWS Sentinel-2 COGs.\n"
                    "Recommended tools: sentinelsat or aws cli. Download tiles intersecting"
                    " your AOI and place the SAFE/COG products under data/raw/optical/sentinel2/."
                ),
                existence_check=lambda agent, ds: dir_has_files(ds.targets[0]),
            )
        )

        register(
            Dataset(
                key="usgs_3dep",
                title="USGS 3DEP Lidar",
                phase=3,
                description="High-resolution 1 m lidar for US coverage (manual retrieval).",
                targets=[root / "data" / "raw" / "lidar" / "usgs_3dep"],
                manual_instructions=(
                    "Download from https://apps.nationalmap.gov/downloader/. Select the"
                    " 3DEP Lidar products for your AOI and store the LAS/LAZ tiles in"
                    " data/raw/lidar/usgs_3dep/."
                ),
                existence_check=lambda agent, ds: dir_has_files(ds.targets[0]),
            )
        )

    # ------------------------------------------------------------------
    # Dataset existence checks
    # ------------------------------------------------------------------

    def dataset_exists(self, dataset: Dataset) -> bool:
        if dataset.existence_check:
            try:
                return bool(dataset.existence_check(self, dataset))
            except Exception as exc:  # pragma: no cover - defensive
                LOG.debug("Existence check failed for %s: %s", dataset.key, exc)
                return False
        for target in dataset.targets:
            if target.is_dir():
                if any(target.iterdir()):
                    return True
            elif target.exists():
                return True
        return False

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_baseline_dataset(self, dataset_key: str) -> bool:
        try:
            from download_missing_data import download_dataset
        except ImportError as exc:  # pragma: no cover - module always present
            LOG.error("Unable to import download_missing_data: %s", exc)
            return False

        LOG.info("Fetching %s via download_missing_data helper", dataset_key)
        return bool(download_dataset(dataset_key, force=self.force))

    def _download_copernicus_dem(self, dataset: Dataset) -> bool:
        try:
            from download_copernicus_dem import download_copernicus_dem_region
        except ImportError as exc:  # pragma: no cover - module always present
            LOG.error("Unable to import download_copernicus_dem: %s", exc)
            return False

        lon_min, lat_min, lon_max, lat_max = self.region.to_tuple()
        output_dir = dataset.targets[0]
        LOG.info(
            "Requesting Copernicus DEM tiles for %.2f/%.2f/%.2f/%.2f",
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        )
        result = download_copernicus_dem_region(
            lon_min=lon_min,
            lat_min=lat_min,
            lon_max=lon_max,
            lat_max=lat_max,
            output_dir=output_dir,
        )
        downloaded = result.get("downloaded_tiles", 0)
        LOG.info("Copernicus DEM downloader reports %s tiles", downloaded)
        return downloaded > 0 or self.dataset_exists(dataset)

    def _download_sentinel1(self, dataset: Dataset) -> bool:
        if self.sentinel_limit <= 0:
            LOG.info("Sentinel-1 limit set to 0; skipping download")
            return True

        try:
            import download_sentinel1 as s1
        except ImportError as exc:  # pragma: no cover - module always present
            LOG.error("Unable to import download_sentinel1: %s", exc)
            return False

        try:
            credentials = s1.load_credentials()
        except SystemExit:
            LOG.error(
                "Sentinel-1 credentials missing. Set COPERNICUS_USER and COPERNICUS_PASS"
            )
            return False

        username, password = credentials
        try:
            token = s1.get_access_token(username, password)
        except Exception as exc:  # pragma: no cover - network/auth failures
            LOG.error("Authentication failed: %s", exc)
            return False

        bounds = self.region.to_tuple()
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=int(dataset.extras.get("days", self.sentinel_days)))
        product_type = str(dataset.extras.get("product_type", "SLC"))
        limit = int(dataset.extras.get("limit", self.sentinel_limit))

        LOG.info(
            "Searching Sentinel-1 products from %s to %s (limit=%d)",
            start_date,
            end_date,
            limit,
        )
        products = s1.search_sentinel1(
            bounds=bounds,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            product_type=product_type,
            orbit=None,
        )

        if not products:
            LOG.warning("No Sentinel-1 products found for the requested window")
            return False

        output_dir = dataset.targets[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        success = 0
        for product in products[:limit]:
            product_id = product["Id"]
            product_name = product["Name"]
            LOG.info("Downloading Sentinel-1 product %s", product_name)
            if s1.download_product(product_id, product_name, token, output_dir):
                success += 1

        if success == 0:
            LOG.warning("Sentinel-1 download attempts failed")
            return False

        return True

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def list_datasets(self) -> List[Dataset]:
        return sorted(self.datasets.values(), key=lambda d: (d.phase, d.key))

    def select_datasets(
        self,
        keys: Optional[Iterable[str]] = None,
        phases: Optional[Iterable[int]] = None,
    ) -> List[Dataset]:
        selected: List[Dataset] = []
        phase_filter = {int(p) for p in phases} if phases else None
        key_filter = None
        if keys:
            normalised = {k.lower().replace("-", "_") for k in keys}
            key_filter = normalised
        for dataset in self.list_datasets():
            if phase_filter and dataset.phase not in phase_filter:
                continue
            if key_filter and dataset.normalised_key() not in key_filter:
                continue
            selected.append(dataset)
        return selected

    def download_datasets(
        self, keys: Optional[Iterable[str]] = None, phases: Optional[Iterable[int]] = None
    ) -> None:
        datasets = self.select_datasets(keys=keys, phases=phases)
        if not datasets:
            LOG.info("No datasets selected for download")
            return

        for dataset in datasets:
            LOG.info("\n=== %s (Phase %d) ===", dataset.title, dataset.phase)
            if dataset.requires_credentials:
                missing = [env for env in dataset.requires_credentials if not self._has_env(env)]
                if missing:
                    message = (
                        "Credentials required: " + ", ".join(dataset.requires_credentials) +
                        ". Missing: " + ", ".join(missing)
                    )
                    LOG.warning(message)
                    self.record_status(dataset, "blocked", {"reason": message})
                    continue

            if self.dataset_exists(dataset) and not self.force:
                LOG.info("Already present, skipping (use --force to re-download)")
                self.record_status(dataset, "complete", {"cached": True})
                continue

            if dataset.download:
                try:
                    success = dataset.download(dataset)
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.exception("Download failed for %s", dataset.key)
                    self.record_status(dataset, "failed", {"error": str(exc)})
                    continue

                if success and self.dataset_exists(dataset):
                    LOG.info("✓ %s download complete", dataset.key)
                    self.record_status(dataset, "complete", {"downloaded": True})
                else:
                    LOG.warning("✗ %s download unsuccessful", dataset.key)
                    self.record_status(dataset, "failed", {"downloaded": False})
                continue

            if dataset.manual_instructions:
                LOG.info("Manual action required for %s:\n%s", dataset.key, dataset.manual_instructions)
                self.record_status(dataset, "manual", {"instructions": dataset.manual_instructions})
                continue

            LOG.warning("No download method defined for %s", dataset.key)
            self.record_status(dataset, "skipped", {"reason": "no downloader"})

    def _has_env(self, name: str) -> bool:
        return name in os.environ

    def status_table(
        self, phases: Optional[Iterable[int]] = None
    ) -> List[Dict[str, object]]:
        table: List[Dict[str, object]] = []
        for dataset in self.select_datasets(keys=None, phases=phases):
            status_record = self._status_data.get("datasets", {}).get(dataset.key, {})
            status = status_record.get("status", "unknown")
            present = self.dataset_exists(dataset)
            table.append(
                {
                    "key": dataset.key,
                    "title": dataset.title,
                    "phase": dataset.phase,
                    "status": status,
                    "present": present,
                    "description": dataset.description,
                }
            )
        return table

    def print_status(self, phases: Optional[Iterable[int]] = None) -> None:
        table = self.status_table(phases)
        if not table:
            LOG.info("No datasets registered")
            return

        header = f"{'Phase':<5}  {'Key':<15}  {'Status':<12}  {'Present':<7}  Title"
        print(header)
        print("-" * len(header))
        for row in table:
            status = row["status"]
            present = "yes" if row["present"] else "no"
            print(
                f"{row['phase']:<5}  {row['key']:<15}  {status:<12}  {present:<7}  {row['title']}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeoAnomalyMapper data agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_REGIONS.keys()),
        default="usa_lower48",
        help="Region preset to use when lon/lat bounds are not provided",
    )
    parser.add_argument("--lon-min", type=float, help="Minimum longitude")
    parser.add_argument("--lat-min", type=float, help="Minimum latitude")
    parser.add_argument("--lon-max", type=float, help="Maximum longitude")
    parser.add_argument("--lat-max", type=float, help="Maximum latitude")
    parser.add_argument(
        "--download",
        nargs="+",
        help="Dataset keys to download (use 'all' for every dataset in scope)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        help="Restrict listing/downloading to the specified phase numbers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    parser.add_argument(
        "--sentinel-products",
        type=int,
        default=2,
        help="Maximum Sentinel-1 products to download per run",
    )
    parser.add_argument(
        "--sentinel-days",
        type=int,
        default=30,
        help="Look-back window for Sentinel-1 searches (days)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit status as JSON instead of a table",
    )
    return parser.parse_args(argv)


def resolve_region(args: argparse.Namespace) -> Region:
    if None not in (args.lon_min, args.lat_min, args.lon_max, args.lat_max):
        return Region(args.lon_min, args.lat_min, args.lon_max, args.lat_max)
    return PRESET_REGIONS[args.preset]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    region = resolve_region(args)
    agent = DataAgent(
        region=region,
        force=args.force,
        sentinel_limit=args.sentinel_products,
        sentinel_days=args.sentinel_days,
    )

    phases = args.phases
    if args.download:
        if "all" in {k.lower() for k in args.download}:
            keys = None
        else:
            keys = args.download
        agent.download_datasets(keys=keys, phases=phases)

    if args.json:
        json.dump(agent.status_table(phases), sys.stdout, indent=2)
        print()
    else:
        agent.print_status(phases)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
