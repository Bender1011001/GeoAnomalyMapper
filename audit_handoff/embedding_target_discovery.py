#!/usr/bin/env python3
"""
Cap 7: Automated Target Discovery from AlphaEarth Embedding Analysis
=====================================================================

Chains Capabilities 1, 2, and 3 from satellite_embeddings.py into an
automated pipeline that:

1. Runs clustering anomaly detection (Cap 1)
2. Runs spatial neighbourhood anomaly detection (Cap 3)
3. Optionally runs temporal trajectory analysis (Cap 2) if multiple years given
4. Fuses the per-pixel scores with configurable weights
5. Identifies candidate target centroids by finding local maxima in the fused
   anomaly score raster
6. Outputs a ranked target list compatible with the PHASE_X_TARGETS format in
   run_biondi_exploration.py, ready to be fed into the PINN pipeline

Usage (CLI):
    python embedding_target_discovery.py \\
        --raster data/alphaearth_exports/vacaville_2023.tif \\
        --year-raster 2022:data/alphaearth_exports/vacaville_2022.tif \\
        --year-raster 2023:data/alphaearth_exports/vacaville_2023.tif \\
        --region Vacaville_CA \\
        --max-targets 20 \\
        --output-dir data/embedding_targets \\
        --targets-json data/embedding_targets/vacaville_targets.json
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine

from satellite_embeddings import (
    cluster_embedding_anomalies,
    compute_spatial_anomaly_score,
    compute_temporal_anomaly_trajectory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_single_band(path: Path) -> Tuple[np.ndarray, Affine, str]:
    """Load the first band of a single-band GeoTIFF into a float32 numpy array."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = str(src.crs)
    return data, transform, crs


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1], NaN-safe."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _find_local_maxima(
    score_map: np.ndarray,
    *,
    min_distance_px: int = 30,
    threshold: float = 0.5,
    max_peaks: int = 100,
) -> List[Tuple[int, int, float]]:
    """Find local maxima in a 2-D score map via non-maximum suppression.

    Returns list of (row, col, score) tuples sorted by score descending.
    Uses a sliding max-filter approach (no scipy.signal.find_peaks_2d dependency).
    """
    from scipy.ndimage import maximum_filter

    valid = np.where(np.isnan(score_map), -np.inf, score_map)
    # Local max: pixel must equal the maximum in its neighbourhood
    footprint_size = 2 * min_distance_px + 1
    local_max_map = maximum_filter(valid, size=footprint_size, mode="constant", cval=-np.inf)
    is_peak = (valid == local_max_map) & (valid >= threshold)

    rows, cols = np.where(is_peak)
    scores = score_map[rows, cols]
    order = np.argsort(scores)[::-1]
    rows, cols, scores = rows[order], cols[order], scores[order]
    return [(int(r), int(c), float(s)) for r, c, s in zip(rows, cols, scores)][:max_peaks]


def _pixel_to_latlon(row: int, col: int, transform: Affine, crs: str) -> Tuple[float, float]:
    """Convert pixel (row, col) to (lat, lon) geographic coordinates."""
    x, y = transform * (col + 0.5, row + 0.5)
    # If the raster CRS is already geographic (EPSG:4326), x=lon, y=lat
    if "4326" in crs or "WGS 84" in crs or "Geographic" in crs:
        return float(y), float(x)
    # Otherwise project via pyproj
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lat), float(lon)
    except ImportError:
        logger.warning("pyproj not available, returning projected coords as lat/lon approximation")
        return float(y), float(x)


# ---------------------------------------------------------------------------
# Core discovery function
# ---------------------------------------------------------------------------

def discover_targets(
    raster_path: Path,
    *,
    year_rasters: Optional[Dict[int, Path]] = None,
    region_name: str = "unknown_region",
    cluster_weight: float = 0.30,
    spatial_weight: float = 0.40,
    temporal_weight: float = 0.30,
    n_clusters: int = 32,
    spatial_radius_px: int = 30,
    min_peak_distance_px: int = 50,
    min_anomaly_score: float = 0.45,
    max_targets: int = 20,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Automated target discovery from embedding analysis (Cap 7).

    Parameters
    ----------
    raster_path : Path
        Primary (or most recent) AlphaEarth 64-band GeoTIFF export.
    year_rasters : dict, optional
        {year: Path} mapping for multi-year temporal analysis.
        If provided and len >= 2, temporal analysis (Cap 2) is run.
    region_name : str
        Human-readable name embedded in output target dicts.
    cluster_weight, spatial_weight, temporal_weight : float
        Weights for fusing the three anomaly score layers.
        Must sum to > 0; will be normalised internally.
    n_clusters : int
        Number of K-means clusters for Cap 1.
    spatial_radius_px : int
        Neighbourhood radius in pixels for Cap 3.
    min_peak_distance_px : int
        Minimum separation between discovered targets in pixels.
    min_anomaly_score : float
        Minimum fused score (0-1) for a pixel to be considered a target.
    max_targets : int
        Maximum number of targets to return.
    output_dir : Path, optional
        If provided, intermediate anomaly rasters are saved here.

    Returns
    -------
    list of dict
        Ranked target list. Each entry is compatible with PHASE_X_TARGETS:
        {
            "name": str,
            "lat": float, "lon": float,
            "anomaly_score": float,   # fused embedding score
            "cluster_score": float,
            "spatial_score": float,
            "temporal_score": float,  # NaN if temporal not run
            "source": "embedding_auto_discovery"
        }
    """
    raster_path = Path(raster_path)
    persist_outputs = output_dir is not None
    temp_dir_ctx = None
    if output_dir is None:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="embedding_target_discovery_")
        work_dir = Path(temp_dir_ctx.name)
    else:
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"[Cap 7] Starting target discovery for {region_name} from {raster_path.name}")

        # --- Cap 1: Clustering anomaly score ---
        logger.info("[Cap 7] Running clustering anomaly detection (Cap 1)...")
        cluster_out = work_dir / f"{region_name}_cluster_anomaly.tif"
        cluster_embedding_anomalies(
            raster_path,
            n_clusters=n_clusters,
            anomaly_out=cluster_out,
        )
        cluster_arr, transform, crs = _load_single_band(cluster_out)

        # --- Cap 3: Spatial neighbourhood anomaly score ---
        logger.info("[Cap 7] Running spatial neighbourhood anomaly detection (Cap 3)...")
        spatial_out = work_dir / f"{region_name}_spatial_anomaly.tif"
        compute_spatial_anomaly_score(
            raster_path,
            outer_radius_px=spatial_radius_px,
            anomaly_out=spatial_out,
        )
        spatial_arr, transform, crs = _load_single_band(spatial_out)

        # --- Cap 2: Temporal trajectory anomaly score (optional) ---
        temporal_arr: Optional[np.ndarray] = None
        run_temporal = year_rasters is not None and len(year_rasters) >= 2
        if run_temporal:
            assert year_rasters is not None
            logger.info(f"[Cap 7] Running temporal trajectory analysis (Cap 2) over {sorted(year_rasters.keys())}...")
            variance_out = work_dir / f"{region_name}_trajectory_variance.tif"
            compute_temporal_anomaly_trajectory(
                year_rasters,
                variance_out=variance_out,
            )
            temporal_arr, _, _ = _load_single_band(variance_out)
        else:
            logger.info("[Cap 7] Skipping temporal analysis (need >= 2 year rasters).")

        # --- Determine canvas shape ---
        with rasterio.open(raster_path) as src:
            H, W = src.height, src.width
            if transform is None:
                transform = src.transform
            if crs is None:
                crs = str(src.crs)

        # --- Fuse scores ---
        # Normalise weights
        available_weights = []
        available_layers = []
        if cluster_arr is not None:
            available_weights.append(cluster_weight)
            available_layers.append(_normalize(cluster_arr))
        if spatial_arr is not None:
            available_weights.append(spatial_weight)
            available_layers.append(_normalize(spatial_arr))
        if temporal_arr is not None:
            available_weights.append(temporal_weight)
            available_layers.append(_normalize(temporal_arr))

        if not available_layers:
            raise RuntimeError("[Cap 7] No anomaly layers could be computed. Check output_dir and input rasters.")

        total_w = sum(available_weights)
        fused = np.zeros((H, W), dtype=np.float32)
        for w, layer in zip(available_weights, available_layers):
            fused += (w / total_w) * layer.astype(np.float32)

        # Save fused map
        if persist_outputs:
            fused_out = work_dir / f"{region_name}_fused_anomaly.tif"
            with rasterio.open(raster_path) as src:
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "count": 1,
                    "height": H,
                    "width": W,
                    "crs": src.crs,
                    "transform": transform,
                    "compress": "lzw",
                }
            with rasterio.open(fused_out, "w", **profile) as dst:
                dst.set_band_description(1, "fused_embedding_anomaly_score")
                dst.write(fused, 1)
            logger.info(f"[Cap 7] Fused anomaly map saved: {fused_out}")

        # --- Find local maxima = candidate targets ---
        peaks = _find_local_maxima(
            fused,
            min_distance_px=min_peak_distance_px,
            threshold=min_anomaly_score,
            max_peaks=max_targets * 3,  # oversample then trim
        )
        logger.info(f"[Cap 7] Found {len(peaks)} candidate peaks above threshold {min_anomaly_score:.2f}")

        # --- Build target dicts ---
        targets: List[Dict] = []
        for rank, (row, col, score) in enumerate(peaks[:max_targets], start=1):
            lat, lon = _pixel_to_latlon(row, col, transform, crs)

            cl_score = float(cluster_arr[row, col]) if cluster_arr is not None else float("nan")
            sp_score = float(spatial_arr[row, col]) if spatial_arr is not None else float("nan")
            tm_score = float(temporal_arr[row, col]) if temporal_arr is not None else float("nan")

            targets.append({
                "name": f"{region_name}_auto_{rank:03d}",
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "anomaly_score": round(score, 5),
                "cluster_anomaly_score": round(cl_score, 5) if not np.isnan(cl_score) else None,
                "spatial_anomaly_score": round(sp_score, 5) if not np.isnan(sp_score) else None,
                "temporal_variance_score": round(tm_score, 5) if not np.isnan(tm_score) else None,
                "pixel_row": row,
                "pixel_col": col,
                "region": region_name,
                "source": "embedding_auto_discovery",
                "discovery_rank": rank,
            })

        logger.info(f"[Cap 7] Top {len(targets)} targets discovered for {region_name}")
        for t in targets[:5]:
            logger.info(
                f"  #{t['discovery_rank']} {t['name']}: lat={t['lat']:.5f}, lon={t['lon']:.5f}, "
                f"fused_score={t['anomaly_score']:.3f}"
            )

        return targets
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cap 7: Automated target discovery from AlphaEarth embedding analysis."
    )
    p.add_argument("--raster", required=True, type=Path, help="Primary 64-band embedding GeoTIFF.")
    p.add_argument(
        "--year-raster",
        action="append",
        metavar="YEAR:PATH",
        help="Year and raster for temporal analysis. Format: 2022:/path/to/file.tif. Repeat for each year.",
    )
    p.add_argument("--region", default="unknown_region", help="Region name for output files.")
    p.add_argument("--cluster-weight", type=float, default=0.30)
    p.add_argument("--spatial-weight", type=float, default=0.40)
    p.add_argument("--temporal-weight", type=float, default=0.30)
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--spatial-radius-px", type=int, default=30)
    p.add_argument("--min-distance-px", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.45)
    p.add_argument("--max-targets", type=int, default=20)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--targets-json", type=Path, help="Write ranked target list to this JSON file.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    year_rasters: Optional[Dict[int, Path]] = None
    if args.year_raster:
        year_rasters = {}
        for entry in args.year_raster:
            yr_str, path_str = entry.split(":", 1)
            year_rasters[int(yr_str)] = Path(path_str)

    targets = discover_targets(
        raster_path=args.raster,
        year_rasters=year_rasters,
        region_name=args.region,
        cluster_weight=args.cluster_weight,
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
        n_clusters=args.n_clusters,
        spatial_radius_px=args.spatial_radius_px,
        min_peak_distance_px=args.min_distance_px,
        min_anomaly_score=args.min_score,
        max_targets=args.max_targets,
        output_dir=args.output_dir,
    )

    if args.targets_json:
        args.targets_json.parent.mkdir(parents=True, exist_ok=True)
        args.targets_json.write_text(
            json.dumps({"region": args.region, "targets": targets}, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Target list written to {args.targets_json}")
    else:
        print(json.dumps({"region": args.region, "targets": targets}, indent=2))


if __name__ == "__main__":
    main()
