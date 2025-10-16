"""Data acquisition agent for GeoAnomalyMapper."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
import yaml
from tqdm import tqdm

from ..utils.hashing import sha256_path
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)
STATUS_PATH = Path("data_status.json")


@dataclass
class Dataset:
    name: str
    url: str
    target: Path
    checksum: Optional[str]


def load_config(path: Path) -> Dict[str, Dataset]:
    config = yaml.safe_load(Path(path).read_text())
    datasets = {}
    for key, entry in config.get("rasters", {}).items():
        datasets[key] = Dataset(
            name=key,
            url=entry["url"],
            target=Path(entry["target"]),
            checksum=entry.get("checksum"),
        )
    return datasets


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    ensure_directory(dest)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    progress.update(len(chunk))
        progress.close()


def load_status() -> Dict[str, Dict[str, str]]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {}


def save_status(status: Dict[str, Dict[str, str]]) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def sync_datasets(config_path: Path, datasets: Optional[Iterable[str]] = None, force: bool = False) -> None:
    catalog = load_config(config_path)
    status = load_status()
    selected = datasets if datasets else catalog.keys()
    for key in selected:
        if key not in catalog:
            LOGGER.warning("Dataset %s not defined in config", key)
            continue
        dataset = catalog[key]
        target = dataset.target
        if target.exists() and not force:
            LOGGER.info("Dataset %s already present at %s", key, target)
            status[key] = {
                "path": str(target),
                "hash": sha256_path(target),
            }
            continue
        LOGGER.info("Downloading %s from %s", key, dataset.url)
        download_file(dataset.url, target)
        checksum = sha256_path(target)
        if dataset.checksum and dataset.checksum != checksum:
            raise ValueError(f"Checksum mismatch for {key}")
        status[key] = {"path": str(target), "hash": checksum}
    save_status(status)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download datasets for GeoAnomalyMapper")
    sub = parser.add_subparsers(dest="command", required=True)
    sync_parser = sub.add_parser("sync", help="Download datasets")
    sync_parser.add_argument("--config", type=Path, default=Path("config/data_sources.yaml"))
    sync_parser.add_argument("--dataset", action="append", help="Specific dataset names")
    sync_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "sync":
        sync_datasets(args.config, args.dataset, args.force)


if __name__ == "__main__":
    main()
