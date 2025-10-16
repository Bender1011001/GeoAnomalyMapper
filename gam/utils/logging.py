"""Central logging utilities for GeoAnomalyMapper."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logging handlers.

    Parameters
    ----------
    level:
        Logging level to apply to the root logger. Defaults to ``logging.INFO``.
    log_file:
        Optional path to a file where logs should additionally be written. The
        directory is created if required.
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger for the requested module.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the caller.
    level:
        Optional logging level that overrides the root configuration for this
        logger.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    if not logger.handlers and not logging.getLogger().handlers:
        configure_logging()
    return logger


__all__ = ["configure_logging", "get_logger"]
