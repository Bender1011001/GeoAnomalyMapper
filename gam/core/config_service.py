"""
Lightweight configuration service.

This module provides ConfigService with a single responsibility: load a YAML
configuration file and validate it against the AppConfig Pydantic model.

Import-time behavior is safe (no I/O); file reads happen only when load() is called.
"""
from __future__ import annotations

from typing import Any
import yaml
from pydantic import ValidationError

from gam.core.config_models import AppConfig


class ConfigService:
    """
    Configuration loader that validates YAML against AppConfig.

    Usage:
        svc = ConfigService()
        cfg = svc.load("path/to/config.yaml")
    """

    def load(self, path: str) -> AppConfig:
        """
        Load and validate configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            AppConfig validated instance.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if YAML is invalid or validation fails.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        try:
            cfg = AppConfig(**raw)
        except ValidationError as exc:
            # Provide a clearer error for callers while preserving original context
            raise ValueError(f"Configuration validation error: {exc}") from exc

        return cfg