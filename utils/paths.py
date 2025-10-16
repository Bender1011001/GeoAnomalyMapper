"""Path helpers backed by :class:`ConfigManager`."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import ConfigManager


class PathManager:
    """Expose common project directories as :class:`pathlib.Path` objects."""

    _instance: Optional["PathManager"] = None

    def __new__(cls) -> "PathManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._config = ConfigManager()
        self._paths: Dict[str, Path] = {
            "data_dir": self._config.get_path("project.data_root", "data"),
            "output_dir": self._config.get_path("project.outputs_dir", "data/outputs"),
            "processed_dir": self._config.get_path(
                "project.processed_dir", "data/processed"
            ),
            "cache_dir": self._config.get_path("project.cache_dir", "data/cache"),
            "raw_data": self._config.get_path("paths.raw_data", "data/raw"),
            "interim": self._config.get_path("paths.interim", "data/interim"),
            "features": self._config.get_path("paths.features", "data/features"),
            "models": self._config.get_path("paths.models", "data/models"),
            "logs": self._config.get_path("paths.logs", "data/outputs/logs"),
        }

        self._initialized = True

    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        if key not in self._paths:
            if default is None:
                raise KeyError(f"Unknown path key '{key}'")
            return Path(default)
        return self._paths[key]

    @property
    def data_dir(self) -> Path:
        return self.get_path("data_dir")

    @property
    def output_dir(self) -> Path:
        return self.get_path("output_dir")

    @property
    def processed_dir(self) -> Path:
        return self.get_path("processed_dir")

    @property
    def cache_dir(self) -> Path:
        return self.get_path("cache_dir")

    def resolve(self, relative_path: str) -> Path:
        return (self._config.project_root / relative_path).resolve()

    def join(self, base_key: str, *subpaths: str) -> Path:
        base = self.get_path(base_key)
        return base.joinpath(*subpaths)

    def keys(self) -> list[str]:
        return list(self._paths.keys())

    def items(self) -> list[tuple[str, Path]]:
        return list(self._paths.items())


paths = PathManager()


if __name__ == "__main__":  # pragma: no cover - convenience execution
    mgr = PathManager()
    print(f"Data dir: {mgr.data_dir}")
    print(f"Output dir: {mgr.output_dir}")
    print("Known keys:", ", ".join(mgr.keys()))
