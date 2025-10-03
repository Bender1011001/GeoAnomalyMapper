"""
GeoAnomalyMapper.gam.visualization — lightweight package initializer.

This package exposes visualization utilities with lazy imports so optional heavy
dependencies (e.g., PyGMT which requires the GMT C library) are only imported
when those specific features are accessed. This prevents failures in runtime
environments that don't have GMT installed, while still allowing modules like
globe_viewer (CesiumJS HTML emitter) to be used.

Public API (lazy-loaded on attribute access):
- VisualizationManager (from .manager) — may import PyGMT via maps_2d.
- StaticMapGenerator, InteractiveMapGenerator (from .maps_2d) — requires PyGMT/GMT.
- volume_3d helpers if used (from .volume_3d) — optional.
- Submodules: globe_viewer, kml_export, tiles_builder.

Using lazy import ensures that:
- `from gam.visualization.globe_viewer import GlobeViewer` does NOT import PyGMT/GMT.
"""

from __future__ import annotations
from typing import Any

__all__ = [
    "VisualizationManager",
    "StaticMapGenerator",
    "InteractiveMapGenerator",
    "globe_viewer",
    "kml_export",
    "tiles_builder",
    # add more names here if needed
]


def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolver to delay importing optional heavy modules until accessed.

    Raises:
        ImportError: When an optional backend is requested but its dependency (e.g., GMT)
                     is not installed in the environment.
    """
    if name == "VisualizationManager":
        try:
            from .manager import VisualizationManager as _VisualizationManager
            return _VisualizationManager
        except Exception as exc:
            raise ImportError(
                "VisualizationManager requires optional backends (e.g., PyGMT/GMT). "
                "Install GMT and PyGMT or use modules that don't require them (e.g., globe_viewer)."
            ) from exc

    if name in ("StaticMapGenerator", "InteractiveMapGenerator"):
        try:
            from .maps_2d import (
                StaticMapGenerator as _StaticMapGenerator,
                InteractiveMapGenerator as _InteractiveMapGenerator,
            )
            return _StaticMapGenerator if name == "StaticMapGenerator" else _InteractiveMapGenerator
        except Exception as exc:
            raise ImportError(
                f"{name} requires PyGMT and the GMT shared library (libgmt). "
                "Please install GMT and PyGMT. See https://www.pygmt.org/ for instructions."
            ) from exc

    if name in ("globe_viewer", "kml_export", "tiles_builder"):
        # Expose submodules lazily; these imports are lightweight and do not pull in PyGMT.
        try:
            from importlib import import_module
            return import_module(f"{__name__}.{name}")
        except Exception as exc:
            raise ImportError(f"Failed to import submodule '{name}' from {__name__}") from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")