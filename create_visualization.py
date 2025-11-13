#!/usr/bin/env python3
"""
Utilities for turning GeoAnomalyMapper rasters into shareable graphics.

The module exposes small helper functions so other tooling (e.g. the workflow
runner) can generate consistent PNG overlays, Google Earth KMZ bundles and
annotated preview images without shelling out to the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # pragma: no mutate (non-interactive backend)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling as RIOResampling
import simplekml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_data_range(geotiff_path: Path) -> Tuple[float, float, float, float]:
    """Inspect the raster and determine sensible color scale limits."""

    with rasterio.open(geotiff_path) as src:
        data = src.read(1, masked=True)

    valid = data.compressed()
    p1, p5, p95, p99 = np.percentile(valid, [1, 5, 95, 99])
    mean = float(np.mean(valid))
    std = float(np.std(valid))

    logger.info("Data stats for %s — mean: %.3f, std: %.3f", geotiff_path.name, mean, std)
    logger.info("Percentiles -> 1%%: %.3f | 5%%: %.3f | 95%%: %.3f | 99%%: %.3f", p1, p5, p95, p99)

    vmin = p5 - 0.2 * abs(p5)
    vmax = p99 + 0.2 * abs(p99)
    abs_max = max(abs(vmin), abs(vmax))
    return -abs_max, abs_max, mean, std


def create_visualization_png(
    geotiff_path: Path,
    output_path: Path,
    vmin: float,
    vmax: float,
    max_dimension: int = 8000,
) -> None:
    """Render a transparent PNG overlay with adaptive downsampling."""

    with rasterio.open(geotiff_path) as src:
        height, width = src.height, src.width
        downsample_factor = max(1, max(height, width) // max_dimension)
        if downsample_factor > 1:
            logger.info(
                "Downsampling %s by %sx for PNG export",
                geotiff_path.name,
                downsample_factor,
            )
            data = src.read(
                1,
                out_shape=(height // downsample_factor, width // downsample_factor),
                resampling=RIOResampling.average,
                masked=True,
            )
        else:
            data = src.read(1, masked=True)

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    rgba = cmap(norm(data.filled(np.nan)))

    magnitude = np.abs(data.data)
    max_mag = max(abs(vmin), abs(vmax))
    alpha = 0.2 + 0.8 * (magnitude / max_mag)
    alpha[data.mask] = 0
    rgba[:, :, 3] = alpha

    height, width = data.shape
    dpi = min(100, int(8000 / max(height, width) * 100))
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgba, origin="upper", interpolation="bilinear")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    logger.info("Saved overlay PNG to %s", output_path)


def create_kmz(
    geotiff_path: Path,
    png_path: Path,
    output_path: Path,
    vmin: float,
    vmax: float,
    mean: float,
    std: float,
) -> None:
    """Create a KMZ bundle for Google Earth."""

    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds

    kml = simplekml.Kml()
    kml.document.name = f"Multi-Resolution Fusion: {geotiff_path.stem}"
    kml.document.description = (
        "Multi-resolution geophysical anomaly map\n\n"
        f"Mean: {mean:.3f} σ\nStd Dev: {std:.3f} σ\n"
        f"Display range: {vmin:.2f} σ to {vmax:.2f} σ\n"
    )

    overlay = kml.newgroundoverlay(name="Anomaly Map")
    overlay.icon.href = png_path.name
    overlay.latlonbox.north = bounds.top
    overlay.latlonbox.south = bounds.bottom
    overlay.latlonbox.east = bounds.right
    overlay.latlonbox.west = bounds.left
    overlay.color = simplekml.Color.changealphaint(180, simplekml.Color.white)

    legend = kml.newfolder(name="Legend")
    legend.description = (
        "COLOR LEGEND\n\n"
        f"Strong negative: {vmin:.2f} σ\n"
        f"Moderate negative: {vmin / 2:.2f} σ\n"
        "Zero anomaly: 0 σ\n"
        f"Moderate positive: {vmax / 2:.2f} σ\n"
        f"Strong positive: {vmax:.2f} σ\n"
    )

    kml.savekmz(str(output_path), [str(png_path)])
    logger.info("Saved KMZ to %s", output_path)


def create_preview_image(
    geotiff_path: Path,
    output_path: Path,
    vmin: float,
    vmax: float,
    max_dimension: int = 4000,
) -> None:
    """Create a downsampled preview PNG with axes/colour bar."""

    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        height, width = src.height, src.width
        downsample_factor = max(1, max(height, width) // max_dimension)
        if downsample_factor > 1:
            data = src.read(
                1,
                out_shape=(height // downsample_factor, width // downsample_factor),
                resampling=RIOResampling.average,
                masked=True,
            )
        else:
            data = src.read(1, masked=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Strength (σ units)", fontsize=12)
    ax.set_title(f"Multi-Resolution Fusion: {geotiff_path.stem}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved preview PNG to %s", output_path)


def generate_visualization_bundle(
    input_path: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate the overlay PNG, KMZ and preview images for ``input_path``."""

    if output_dir is None:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    vmin, vmax, mean, std = analyze_data_range(input_path)
    base_name = input_path.stem

    overlay_png = output_dir / f"{base_name}_overlay.png"
    kmz_path = output_dir / f"{base_name}.kmz"
    preview_png = output_dir / f"{base_name}_preview.png"

    create_visualization_png(input_path, overlay_png, vmin, vmax)
    create_kmz(input_path, overlay_png, kmz_path, vmin, vmax, mean, std)
    create_preview_image(input_path, preview_png, vmin, vmax)

    return {
        "overlay_png": overlay_png,
        "kmz": kmz_path,
        "preview_png": preview_png,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create PNG, KMZ and preview images for GeoTIFF outputs.",
    )
    parser.add_argument("input_tif", type=str, help="Input GeoTIFF file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for generated graphics (default: alongside the input).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> Dict[str, Path]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_tif)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    outputs = generate_visualization_bundle(input_path, output_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION FILES CREATED")
    print("=" * 70)
    print(f"\nInput: {input_path}")
    print("Outputs:")
    print(f"  - Google Earth KMZ : {outputs['kmz']}")
    print(f"  - Preview PNG      : {outputs['preview_png']}")
    print(f"  - Overlay PNG      : {outputs['overlay_png']}")
    print("=" * 70 + "\n")
    return outputs


if __name__ == "__main__":
    main()
