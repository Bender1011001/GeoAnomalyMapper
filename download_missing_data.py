#!/usr/bin/env python3
"""Utility to resolve missing baseline datasets referenced in reports.

The tool parses the final project report (or processing log) for dataset names,
checks whether the required GeoTIFFs exist locally, and, when possible, attempts
an automated download. When a fully automated download is not available, clear
manual instructions are written to the console so the operator can obtain the
files.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from requests import RequestException
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Known baseline datasets required by the processing pipeline.
DATASET_REGISTRY: Dict[str, Dict[str, Optional[str]]] = {
    "EMAG2_V3_Sea_Level.tif": {
        "download_url": "https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif",
        "checksum": None,  # Published checksum not provided in the upstream catalog.
        "description": "Global magnetic anomaly grid at sea level (2 arc-min resolution).",
        "destination": Path("data/raw/magnetic/EMAG2_V3_Sea_Level.tif"),
        "notes": (
            "Large (~1.7 GB) download. NOAA servers occasionally throttle long"
            " transfers—rerun the script if the connection drops."
        ),
    },
    "EGM2008_Free_Air_Anomaly.tif": {
        "download_url": "https://topex.ucsd.edu/gravity/EGM2008/EGM2008_Free_Air_Anomaly.tif",
        "checksum": None,
        "description": "EGM2008 global free-air gravity anomaly grid (1 arc-min).",
        "destination": Path("data/raw/gravity/EGM2008_Free_Air_Anomaly.tif"),
        "notes": (
            "If the UCSD mirror is unavailable, request the GeoTIFF from the ICGEM"
            " portal (https://icgem.gfz-potsdam.de/) using the grid download tool."
        ),
    },
}

# Regex patterns capture the ways missing datasets are referenced in the report/log.
REQUIRED_PATTERN = re.compile(r"Required:\s*(?P<names>.+)", re.IGNORECASE)
MISSING_PATTERN = re.compile(r"Missing\s+data:\s*(?P<names>.+)", re.IGNORECASE)
DATASET_TOKEN_SPLIT = re.compile(r"[,;]\s*")


def parse_report_for_datasets(report_text: str) -> List[str]:
    """Extract dataset names listed as missing/required."""

    candidates: List[str] = []
    for pattern in (REQUIRED_PATTERN, MISSING_PATTERN):
        for match in pattern.finditer(report_text):
            raw_names = match.group("names")
            if not raw_names:
                continue
            for token in DATASET_TOKEN_SPLIT.split(raw_names):
                cleaned = token.strip().strip(".")
                if cleaned:
                    candidates.append(cleaned)

    return candidates


def normalise_dataset_name(name: str) -> Optional[str]:
    """Map arbitrary tokens to canonical dataset keys."""

    normalised = name.strip()
    if normalised in DATASET_REGISTRY:
        return normalised

    # Fallback: tolerate paths or prefixes.
    for key in DATASET_REGISTRY:
        if key.lower() in normalised.lower():
            logger.debug("Normalised '%s' to '%s'", name, key)
            return key

    logger.warning("Unknown dataset reference: %s", name)
    return None


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_dataset(name: str, force: bool = False) -> bool:
    info = DATASET_REGISTRY[name]
    destination: Path = info["destination"]  # type: ignore[assignment]
    ensure_directory(destination)

    if destination.exists() and not force:
        logger.info("%s already present at %s", name, destination)
        return True

    url = info["download_url"]
    if not url:
        logger.error("No automated download URL available for %s", name)
        return False

    logger.info("Downloading %s", name)
    logger.info("Source: %s", url)
    logger.info("Destination: %s", destination)

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            temp_path = destination.with_suffix(destination.suffix + ".part")
            with temp_path.open("wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    progress.update(len(chunk))
            temp_path.replace(destination)
    except RequestException as exc:
        logger.error("Failed to download %s: %s", name, exc)
        return False

    checksum = info.get("checksum")
    if checksum:
        actual = compute_sha256(destination)
        if actual.lower() != checksum.lower():
            logger.error(
                "Checksum mismatch for %s (expected %s, got %s)",
                name,
                checksum,
                actual,
            )
            return False
        logger.info("Checksum verified for %s", name)

    notes = info.get("notes")
    if notes:
        logger.info("Notes: %s", notes)

    logger.info("Successfully ensured %s", name)
    return True


def resolve_datasets_from_sources(
    report_path: Optional[Path],
    explicit: Iterable[str],
) -> List[str]:
    requested: List[str] = []

    if report_path and report_path.exists():
        logger.info("Parsing report/log: %s", report_path)
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        requested.extend(parse_report_for_datasets(report_text))
    elif report_path:
        logger.warning("Report path %s does not exist", report_path)

    requested.extend(explicit)

    normalised: List[str] = []
    for token in requested:
        key = normalise_dataset_name(token)
        if key and key not in normalised:
            normalised.append(key)

    if not normalised:
        logger.warning("No recognised datasets found. Falling back to all baseline datasets.")
        normalised = list(DATASET_REGISTRY.keys())

    return normalised


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or verify GeoAnomalyMapper baseline datasets",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/outputs/processing.log"),
        help="Path to the final project report or processing log",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Explicit dataset name to ensure (can be repeated)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the dataset already exists",
    )

    args = parser.parse_args(argv)

    datasets = resolve_datasets_from_sources(args.report, args.dataset)
    logger.info("Datasets to ensure: %s", ", ".join(datasets))

    success = True
    for dataset in datasets:
        ensured = download_dataset(dataset, force=args.force)
        if not ensured:
            info = DATASET_REGISTRY[dataset]
            notes = info.get("notes")
            if notes:
                logger.info("Manual follow-up required for %s: %s", dataset, notes)
            success = False

    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
