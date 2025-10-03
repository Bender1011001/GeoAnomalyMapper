"""KML/KMZ export utilities for anomaly heatmap overlays.

This module provides simple helpers to export a 2D anomaly grid as a translucent
PNG heatmap draped over the Earth via KML GroundOverlay, and an optional KMZ
packaging convenience.

Coordinate assumptions
- BBox order is (min_lon, min_lat, max_lon, max_lat).
- Coordinates are degrees (WGS84 / EPSG:4326) as typically expected by Google Earth.

Typical usage
- KML: Produces a .kml file that references a PNG placed adjacent to the KML.
- KMZ: Produces a single .kmz archive bundling the KML and PNG (via simplekml.savekmz).

Open the resulting KML/KMZ in Google Earth (desktop) or compliant viewers.

Notes
- This utility normalizes the anomaly array to [0, 1] in a NaN-safe manner.
- Rendering uses per-pixel alpha equal to the normalized value, making low values
  more transparent and high values more opaque.
- A deterministic filename "heatmap.png" is used for the overlay image next to
  the output path directory. For KMZ, the PNG is created, then packaged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import simplekml
import matplotlib.pyplot as plt
from matplotlib import cm
import zipfile
import io
import os

logger = logging.getLogger(__name__)


def _normalize_anomalies(anomalies: np.ndarray) -> np.ndarray:
    """Normalize a 2D anomaly array to [0, 1] with NaN-safe handling.

    Parameters
    ----------
    anomalies : np.ndarray
        2D array of anomaly values. NaNs are permitted.

    Returns
    -------
    np.ndarray
        2D float array in [0, 1], NaNs mapped to 0.

    Raises
    ------
    ValueError
        If anomalies is not 2D.
    """
    if anomalies.ndim != 2:
        raise ValueError("Expected a 2D array for 'anomalies'")

    arr = np.array(anomalies, dtype=float)
    # Handle case where all values are NaN
    with np.errstate(all="ignore"):
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.zeros_like(arr, dtype=float)

    eps = 1e-12
    rng = max(vmax - vmin, eps)
    norm = (arr - vmin) / rng
    # Map NaNs/infs to finite [0,1] range
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(norm, 0.0, 1.0)


def _render_heatmap_png(
    norm: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_path: Path,
    cmap: str = "hot",
) -> Path:
    """Render a normalized heatmap to a transparent PNG aligned to a geographic bbox.

    Parameters
    ----------
    norm : np.ndarray
        2D array already normalized to [0, 1].
    bbox : Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) in degrees.
    image_path : pathlib.Path
        Output PNG path (parent directories will be created).
    cmap : str, optional
        Matplotlib colormap name (default "hot").

    Returns
    -------
    pathlib.Path
        The path to the written PNG image.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Build RGBA image with per-pixel alpha = normalized value
    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(norm)  # shape (H, W, 4)
    rgba[..., 3] = norm   # set alpha to normalized intensity

    height, width = norm.shape
    dpi = 100.0
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        ax.imshow(
            rgba,
            extent=(min_lon, max_lon, min_lat, max_lat),  # (left, right, bottom, top)
            origin="lower",
            interpolation="nearest",
        )
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.axis("off")
        # Remove any margins so the raster fills the canvas exactly
        fig.subplots_adjust(0, 0, 1, 1)

        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Transparent background; no extra padding
        fig.savefig(str(image_path), transparent=True, bbox_inches="tight", pad_inches=0)
    finally:
        plt.close(fig)

    return image_path


def export_anomaly_kml(
    anomalies: np.ndarray,
    bbox: Tuple[float, float, float, float],
    out_path: str,
    cmap: str = "hot",
) -> str:
    """Export anomaly heatmap as KML with a GroundOverlay referencing a PNG.

    Steps
    - Validates anomalies is 2D and normalizes to [0, 1] (NaN-safe).
    - Renders a transparent PNG heatmap aligned to bbox and saves it next to the KML.
    - Creates a KML GroundOverlay with href to "heatmap.png" and bbox bounds.
    - Saves the KML to out_path and returns the path.

    Parameters
    ----------
    anomalies : np.ndarray
        2D anomaly array.
    bbox : Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) in degrees (EPSG:4326).
    out_path : str
        Destination .kml file path.
    cmap : str, optional
        Matplotlib colormap name. Default is "hot".

    Returns
    -------
    str
        The out_path provided.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.random.rand(100, 150)
    >>> bbox = (-123.5, 37.5, -121.0, 38.7)  # (min_lon, min_lat, max_lon, max_lat)
    >>> _ = export_anomaly_kml(arr, bbox, "outputs/anomaly_overlay.kml", cmap="hot")
    """
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    norm = _normalize_anomalies(anomalies)

    # Deterministic filename placed adjacent to KML
    image_path = out_path_p.parent / "heatmap.png"
    _render_heatmap_png(norm, bbox, image_path, cmap=cmap)

    min_lon, min_lat, max_lon, max_lat = bbox
    kml = simplekml.Kml()
    ground = kml.newgroundoverlay(name="Anomaly Heatmap")
    ground.icon.href = "heatmap.png"  # relative href so KML resolves adjacent file
    ground.latlonbox.north = max_lat
    ground.latlonbox.south = min_lat
    ground.latlonbox.east = max_lon
    ground.latlonbox.west = min_lon
    ground.latlonbox.rotation = 0

    kml.save(str(out_path_p))
    logger.info(f"Exported KML GroundOverlay to {out_path_p}")
    return str(out_path_p)


def export_anomaly_kmz(
    anomalies: np.ndarray,
    bbox: Tuple[float, float, float, float],
    out_path: str,
    cmap: str = "hot",
) -> str:
    """Export anomaly heatmap as a KMZ (KML + embedded PNG).

    Behavior
    - Same normalization and rendering as export_anomaly_kml.
    - Uses simplekml.Kml().savekmz(out_path) to package the KML and referenced PNG.
    - Returns out_path. If packaging is unsupported, attempts a minimal ZIP fallback.

    Parameters
    ----------
    anomalies : np.ndarray
        2D anomaly array.
    bbox : Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) in degrees (EPSG:4326).
    out_path : str
        Destination .kmz file path (should end with .kmz).
    cmap : str, optional
        Matplotlib colormap name. Default is "hot".

    Returns
    -------
    str
        The out_path provided.

    Notes
    -----
    - Preferred path uses simplekml.savekmz which bundles local file references.
    - If savekmz is unavailable/fails, a minimal fallback creates a KMZ via zipfile
      with entries:
        - doc.kml
        - files/heatmap.png
      and rewrites the href accordingly.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.random.rand(200, 300)
    >>> bbox = (-10.0, 45.0, 2.0, 55.0)
    >>> _ = export_anomaly_kmz(arr, bbox, "outputs/anomaly_overlay.kmz", cmap="hot")
    """
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    norm = _normalize_anomalies(anomalies)

    # Create overlay image next to output for savekmz to discover
    temp_image_path = out_path_p.parent / "heatmap.png"
    _render_heatmap_png(norm, bbox, temp_image_path, cmap=cmap)

    min_lon, min_lat, max_lon, max_lat = bbox
    kml = simplekml.Kml()
    ground = kml.newgroundoverlay(name="Anomaly Heatmap")
    # Use relative href for bundling
    ground.icon.href = "heatmap.png"
    ground.latlonbox.north = max_lat
    ground.latlonbox.south = min_lat
    ground.latlonbox.east = max_lon
    ground.latlonbox.west = min_lon
    ground.latlonbox.rotation = 0

    # Preferred: simplekml's kmz packer
    try:
        # savekmz should find and include 'heatmap.png' into KMZ
        kml.savekmz(str(out_path_p))
        logger.info(f"Exported KMZ GroundOverlay to {out_path_p} (simplekml.savekmz)")
        # Cleanup temporary PNG after packaging
        try:
            temp_image_path.unlink(missing_ok=True)  # Python 3.8+: missing_ok param exists in 3.8+? Actually it's 3.8+, else ignore.
        except TypeError:
            # Fallback for older Python: ignore if file missing
            if temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except Exception:
                    pass
        return str(out_path_p)
    except Exception as e:
        logger.warning(f"simplekml.savekmz failed ({e}); attempting minimal ZIP fallback")

    # Fallback: create KMZ manually (zip with doc.kml and files/heatmap.png)
    try:
        # Rewrite href to point into 'files/' inside KMZ
        ground.icon.href = "files/heatmap.png"

        # Obtain KML document as bytes
        # simplekml doesn't provide direct bytes method in all versions; use temp buffer
        kml_bytes: bytes
        try:
            # Some versions support .kml() returning XML string
            kml_str = kml.kml()  # type: ignore[attr-defined]
            if isinstance(kml_str, str):
                kml_bytes = kml_str.encode("utf-8")
            else:
                # Unexpected type; fallback to file write
                raise AttributeError
        except Exception:
            # Fallback: write to in-memory file-like via save(), then read back
            # save() needs a filesystem path; use temporary on disk
            tmp_doc = out_path_p.parent / "doc.kml"
            kml.save(str(tmp_doc))
            with open(tmp_doc, "rb") as f:
                kml_bytes = f.read()
            try:
                tmp_doc.unlink()
            except Exception:
                pass

        # Read image bytes
        with open(temp_image_path, "rb") as f:
            img_bytes = f.read()

        # Write KMZ structure
        with zipfile.ZipFile(str(out_path_p), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml_bytes)
            zf.writestr("files/heatmap.png", img_bytes)

        logger.info(f"Exported KMZ GroundOverlay to {out_path_p} (zip fallback)")
    finally:
        # Cleanup temporary PNG
        try:
            temp_image_path.unlink(missing_ok=True)
        except TypeError:
            if temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except Exception:
                    pass

    return str(out_path_p)