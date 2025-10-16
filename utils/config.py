"""Modern configuration management for GeoAnomalyMapper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore


class ConfigManager:
    """Singleton loader for the project's JSON configuration file."""

    _instance: Optional["ConfigManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: Optional[Path] = None,
        env_prefix: str = "GAM",
        auto_create_paths: bool = True,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self._project_root = Path(__file__).resolve().parents[1]
        self._config_path = (
            Path(config_path)
            if config_path is not None
            else self._project_root / "config" / "config.json"
        )
        self._example_path = self._config_path.with_suffix(".json.example")
        self._env_prefix = env_prefix
        self._auto_create_paths = auto_create_paths

        # Load environment variables from .env when available
        env_file = self._project_root / ".env"
        if load_dotenv is not None and env_file.exists():
            load_dotenv(env_file)

        self._directory_keys = (
            "project.data_root",
            "project.outputs_dir",
            "project.processed_dir",
            "project.cache_dir",
            "paths.raw_data",
            "paths.interim",
            "paths.features",
            "paths.models",
            "paths.logs",
        )

        self._config_data = self._load_config_file()
        self._apply_environment_overrides()

        if self._auto_create_paths:
            self._ensure_directories()

        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_config_file(self) -> Dict[str, Any]:
        if self._config_path.exists():
            path = self._config_path
        elif self._example_path.exists():
            path = self._example_path
        else:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _apply_environment_overrides(self) -> None:
        prefix = f"{self._env_prefix}__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path_tokens = key[len(prefix) :].split("__")
            self._set_nested(path_tokens, self._coerce_value(value))

    def _set_nested(self, tokens: Iterable[str], value: Any) -> None:
        cursor: Dict[str, Any] = self._config_data
        tokens_list = list(tokens)
        for token in tokens_list[:-1]:
            if token not in cursor or not isinstance(cursor[token], dict):
                cursor[token] = {}
            cursor = cursor[token]
        cursor[tokens_list[-1]] = value

    def _ensure_directories(self) -> None:
        for key in self._directory_keys:
            path = self.get_path(key)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_value(value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def source_path(self) -> Path:
        return self._config_path

    def reload(self) -> None:
        self._config_data = self._load_config_file()
        self._apply_environment_overrides()
        if self._auto_create_paths:
            self._ensure_directories()

    def get(self, key: str, default: Any = None) -> Any:
        tokens = key.split(".") if key else []
        cursor: Any = self._config_data
        for token in tokens:
            if isinstance(cursor, dict) and token in cursor:
                cursor = cursor[token]
            else:
                return default
        return cursor

    def set(self, key: str, value: Any) -> None:
        tokens = key.split(".") if key else []
        if not tokens:
            raise ValueError("Configuration key must not be empty")
        self._set_nested(tokens, value)
        if self._auto_create_paths and (
            key.endswith("_dir") or key in self._directory_keys or key.startswith("paths.")
        ):
            path = self.get_path(key)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str, default: Optional[Any] = None) -> Optional[Path]:
        value = self.get(key, default)
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (self._project_root / path).resolve()
        return path

    def items(self) -> Iterator[tuple[str, Any]]:
        def _walk(prefix: str, node: Any) -> Iterator[tuple[str, Any]]:
            if isinstance(node, dict):
                for child_key, child_value in node.items():
                    child_prefix = f"{prefix}.{child_key}" if prefix else child_key
                    yield from _walk(child_prefix, child_value)
            else:
                yield prefix, node

        yield from _walk("", self._config_data)

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._config_data))


# Global singleton instance used across the project
config = ConfigManager()


if __name__ == "__main__":  # pragma: no cover - convenience execution
    cfg = ConfigManager()
    print(f"Loaded configuration from: {cfg.source_path}")
    for key, value in list(cfg.items())[:10]:
        print(f"{key}: {value}")