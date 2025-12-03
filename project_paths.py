"""
Centralised filesystem helpers for GeoAnomalyMapper.

All processing scripts share the same convention: keep data, processed
artefacts and outputs under the repository's ``data/`` directory by default,
but allow users to override the location via the ``GEOANOMALYMAPPER_DATA_DIR``
environment variable.  Import this module instead of duplicating ad-hoc path
logic in every CLI tool.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_data_root() -> Path:
    """Return the configured data directory, expanding user overrides with validation."""
    custom = os.environ.get("GEOANOMALYMAPPER_DATA_DIR")
    if custom:
        data_path = Path(custom).expanduser().resolve()
        
        # Validate the path exists
        if not data_path.exists():
            logger.warning(f"Data directory does not exist: {data_path}. Creating it.")
            try:
                data_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PermissionError(f"Cannot create data directory {data_path}: {e}")
        
        # Check if writable
        if not os.access(data_path, os.W_OK):
            raise PermissionError(f"Data directory is not writable: {data_path}")
        
        logger.info(f"Using custom data directory: {data_path}")
        return data_path
    return REPO_ROOT / "data"


DATA_DIR: Path = _resolve_data_root()
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"


def ensure_directories(directories: Iterable[Path] | None = None) -> None:
    """
    Create the core data directories (or any supplied list) if they are missing.

    Parameters
    ----------
    directories:
        Optional iterable of paths to create. When omitted the function ensures
        raw/processed/output subdirectories exist beneath the configured root.
    """

    dirs = list(directories) if directories is not None else [
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUTS_DIR,
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "OUTPUTS_DIR",
    "REPO_ROOT",
    "ensure_directories",
]
