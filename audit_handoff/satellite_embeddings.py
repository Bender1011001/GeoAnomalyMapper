#!/usr/bin/env python3
"""
AlphaEarth / Google Satellite Embedding utilities for GeoAnomalyMapper.

The Google Satellite Embedding collection is useful here as annual surface-context
data. It is not raw SAR SLC data and cannot replace the phase history required by
the Doppler vibrometry pipeline. This module operates on real 64-band embedding
GeoTIFF exports and computes unit-vector dot-product change products.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import guard_transform
from rasterio.vrt import WarpedVRT


EARTH_ENGINE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS: Tuple[str, ...] = tuple(f"A{i:02d}" for i in range(64))
DEFAULT_CHANGE_THRESHOLDS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)


class EmbeddingRasterError(RuntimeError):
    """Raised when an input raster cannot be treated as a 64D embedding raster."""


@dataclass
class HistogramStats:
    """Streaming numeric summary with histogram-backed percentile estimates."""

    histogram_min: float
    histogram_max: float
    histogram_bins: int
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    histogram: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if self.histogram_max <= self.histogram_min:
            raise ValueError("histogram_max must be greater than histogram_min")
        self.histogram = np.zeros(self.histogram_bins, dtype=np.int64)
        self.edges = np.linspace(
            self.histogram_min, self.histogram_max, self.histogram_bins + 1
        )

    def update(self, values: np.ndarray) -> None:
        valid = np.asarray(values, dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return

        self.count += int(valid.size)
        self.total += float(valid.sum(dtype=np.float64))
        self.total_sq += float(np.square(valid, dtype=np.float64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(valid.min()))
        self.maximum = max(self.maximum, float(valid.max()))

        clipped = np.clip(valid, self.histogram_min, self.histogram_max)
        hist, _ = np.histogram(clipped, bins=self.edges)
        self.histogram += hist.astype(np.int64)

    @property
    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self) -> Optional[float]:
        if self.count == 0:
            return None
        mean = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean * mean), 0.0)
        return float(np.sqrt(variance))

    def percentile(self, q: float) -> Optional[float]:
        if self.count == 0:
            return None
        if not 0.0 <= q <= 100.0:
            raise ValueError("percentile must be in [0, 100]")
        cumulative = np.cumsum(self.histogram)
        target = max(int(np.ceil((q / 100.0) * self.count)), 1)
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(idx, self.histogram_bins - 1)
        return float((self.edges[idx] + self.edges[idx + 1]) / 2.0)

    def to_dict(self, percentiles: Sequence[float]) -> Dict[str, Optional[float]]:
        if self.count == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                **{f"approx_p{int(q)}": None for q in percentiles},
            }
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            **{f"approx_p{int(q)}": self.percentile(q) for q in percentiles},
        }


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object is not JSON serializable: {type(obj)!r}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _embedding_band_indexes(src: rasterio.io.DatasetReader) -> List[int]:
    descriptions = [desc or "" for desc in src.descriptions]
    description_map = {name: i + 1 for i, name in enumerate(descriptions) if name}
    if all(name in description_map for name in EMBEDDING_BANDS):
        return [description_map[name] for name in EMBEDDING_BANDS]

    if src.count == len(EMBEDDING_BANDS):
        return list(range(1, len(EMBEDDING_BANDS) + 1))

    raise EmbeddingRasterError(
        f"{src.name} has {src.count} bands and no complete A00..A63 band descriptions"
    )


def _grid_matches(
    left: rasterio.io.DatasetReader,
    right: rasterio.io.DatasetReader,
) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and left.transform.almost_equals(right.transform)
    )


def _single_band_profile(
    src: rasterio.io.DatasetReader,
    *,
    nodata: float = np.nan,
) -> Dict:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _read_embedding_block(
    src: rasterio.io.DatasetReader,
    indexes: Sequence[int],
    window,
) -> Tuple[np.ndarray, np.ndarray]:
    block = src.read(indexes=list(indexes), window=window, masked=True).astype("float32")
    mask = np.ma.getmaskarray(block).any(axis=0)
    data = np.asarray(block.filled(np.nan), dtype=np.float32)
    finite = np.isfinite(data).all(axis=0)
    return data, mask | ~finite


def compute_embedding_similarity(
    before_block: np.ndarray,
    after_block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AlphaEarth dot-product similarity and change score for matching blocks.

    Inputs must be shaped as (64, rows, cols). Because official embeddings are
    unit-length, the dot product is cosine similarity. The returned change score is
    1 - clipped_similarity, where 0 means unchanged and larger values mean less
    similar annual surface conditions.
    """

    if before_block.shape != after_block.shape:
        raise ValueError(
            f"Embedding block shapes must match, got {before_block.shape} and {after_block.shape}"
        )
    if before_block.ndim != 3 or before_block.shape[0] != len(EMBEDDING_BANDS):
        raise ValueError(
            f"Embedding blocks must have shape (64, rows, cols), got {before_block.shape}"
        )

    raw_similarity = np.einsum("brc,brc->rc", before_block, after_block, dtype=np.float64)
    similarity = np.clip(raw_similarity, -1.0, 1.0).astype(np.float32)
    change_score = (1.0 - similarity).astype(np.float32)
    return similarity, change_score


def inspect_embedding_raster(
    raster_path: Path,
    *,
    summary_out: Optional[Path] = None,
    histogram_bins: int = 1024,
) -> Dict:
    """Inspect a real 64-band embedding raster without loading the full image at once."""

    raster_path = Path(raster_path)
    norm_stats = HistogramStats(0.0, 2.0, histogram_bins)

    with rasterio.open(raster_path) as src:
        indexes = _embedding_band_indexes(src)
        for _, window in src.block_windows(1):
            block, invalid = _read_embedding_block(src, indexes, window)
            norms = np.sqrt(np.sum(np.square(block, dtype=np.float64), axis=0))
            norms[invalid] = np.nan
            norm_stats.update(norms)

        summary = {
            "input": str(raster_path),
            "collection": EARTH_ENGINE_COLLECTION,
            "band_count": src.count,
            "embedding_band_indexes": indexes,
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "pixel_size": [float(src.transform.a), float(src.transform.e)],
            "descriptions": [desc or "" for desc in src.descriptions],
            "unit_length_check": norm_stats.to_dict([1, 5, 50, 95, 99]),
        }

    if summary_out:
        summary_out = Path(summary_out)
        _ensure_parent(summary_out)
        summary_out.write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8")

    return summary


def compare_embedding_rasters(
    before_path: Path,
    after_path: Path,
    *,
    change_out: Path,
    similarity_out: Optional[Path] = None,
    summary_out: Optional[Path] = None,
    resample_to_before: bool = False,
    histogram_bins: int = 2048,
    change_thresholds: Iterable[float] = DEFAULT_CHANGE_THRESHOLDS,
) -> Dict:
    """
    Compare two annual 64-band embedding rasters and write a change-score GeoTIFF.

    The rasters are streamed block by block, so full AlphaEarth export tiles do not
    need to fit in memory. By default both rasters must share grid, CRS, transform,
    and dimensions. Use resample_to_before=True to bilinearly warp the second raster
    onto the first raster's grid during reading.
    """

    before_path = Path(before_path)
    after_path = Path(after_path)
    change_out = Path(change_out)
    similarity_out = Path(similarity_out) if similarity_out else None
    summary_out = Path(summary_out) if summary_out else None

    thresholds = tuple(float(t) for t in change_thresholds)
    threshold_counts = {str(t): 0 for t in thresholds}
    change_stats = HistogramStats(0.0, 2.0, histogram_bins)
    similarity_stats = HistogramStats(-1.0, 1.0, histogram_bins)
    clipped_similarity_pixels = 0

    _ensure_parent(change_out)
    if similarity_out:
        _ensure_parent(similarity_out)
    if summary_out:
        _ensure_parent(summary_out)

    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
        before_indexes = _embedding_band_indexes(before_src)
        after_indexes = _embedding_band_indexes(after_src)

        if not _grid_matches(before_src, after_src) and not resample_to_before:
            raise EmbeddingRasterError(
                "Input rasters are not on the same grid. Re-export with identical AOI/scale "
                "or rerun with --resample-to-before."
            )

        if resample_to_before and not _grid_matches(before_src, after_src):
            after_reader = WarpedVRT(
                after_src,
                crs=before_src.crs,
                transform=before_src.transform,
                width=before_src.width,
                height=before_src.height,
                resampling=Resampling.bilinear,
            )
        else:
            after_reader = after_src

        profile = _single_band_profile(before_src)
        with rasterio.open(change_out, "w", **profile) as change_dst:
            change_dst.set_band_description(1, "alphaearth_change_score")
            if similarity_out:
                similarity_dst_ctx = rasterio.open(similarity_out, "w", **profile)
            else:
                similarity_dst_ctx = None

            try:
                if similarity_dst_ctx:
                    similarity_dst_ctx.set_band_description(1, "alphaearth_dot_product_similarity")

                for _, window in before_src.block_windows(1):
                    before_block, before_invalid = _read_embedding_block(
                        before_src, before_indexes, window
                    )
                    after_block, after_invalid = _read_embedding_block(
                        after_reader, after_indexes, window
                    )

                    raw_similarity = np.einsum(
                        "brc,brc->rc", before_block, after_block, dtype=np.float64
                    )
                    invalid = before_invalid | after_invalid | ~np.isfinite(raw_similarity)
                    clipped_similarity_pixels += int(
                        np.count_nonzero(
                            (~invalid) & ((raw_similarity < -1.0) | (raw_similarity > 1.0))
                        )
                    )

                    similarity = np.clip(raw_similarity, -1.0, 1.0).astype(np.float32)
                    change = (1.0 - similarity).astype(np.float32)
                    similarity[invalid] = np.nan
                    change[invalid] = np.nan

                    valid_change = change[np.isfinite(change)]
                    change_stats.update(valid_change)
                    similarity_stats.update(similarity[np.isfinite(similarity)])
                    for threshold in thresholds:
                        threshold_counts[str(threshold)] += int(
                            np.count_nonzero(valid_change > threshold)
                        )

                    change_dst.write(change, 1, window=window)
                    if similarity_dst_ctx:
                        similarity_dst_ctx.write(similarity, 1, window=window)
            finally:
                if similarity_dst_ctx:
                    similarity_dst_ctx.close()
                if isinstance(after_reader, WarpedVRT):
                    after_reader.close()

        valid_pixels = change_stats.count
        threshold_fractions = {
            threshold: (count / valid_pixels if valid_pixels else None)
            for threshold, count in threshold_counts.items()
        }

        summary = {
            "collection": EARTH_ENGINE_COLLECTION,
            "before": str(before_path),
            "after": str(after_path),
            "change_out": str(change_out),
            "similarity_out": str(similarity_out) if similarity_out else None,
            "resample_to_before": resample_to_before,
            "definition": {
                "similarity": "dot product of the 64-dimensional annual embedding vectors",
                "change_score": "1 - clipped_similarity; 0 means unchanged, larger means less similar",
            },
            "grid": {
                "crs": str(before_src.crs),
                "width": before_src.width,
                "height": before_src.height,
                "transform": list(before_src.transform)[:6],
            },
            "valid_pixels": valid_pixels,
            "change_score": change_stats.to_dict([1, 5, 50, 90, 95, 99]),
            "similarity": similarity_stats.to_dict([1, 5, 50, 90, 95, 99]),
            "change_threshold_pixel_counts": threshold_counts,
            "change_threshold_pixel_fractions": threshold_fractions,
            "clipped_similarity_pixels": clipped_similarity_pixels,
        }

    if summary_out:
        summary_out.write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8")

    return summary


def write_earth_engine_export_script(
    *,
    lat: float,
    lon: float,
    buffer_deg: float,
    years: Sequence[int],
    output_js: Path,
    description_prefix: str = "geoanomaly_alphaearth",
    drive_folder: str = "GeoAnomalyMapper_AlphaEarth",
) -> Path:
    """
    Write an Earth Engine JavaScript export script for real AlphaEarth GeoTIFFs.

    The script exports one 64-band GeoTIFF per requested year over the provided AOI.
    It must be run in the Earth Engine Code Editor by an account with Earth Engine
    access; this function only writes the reproducible export script.
    """

    if buffer_deg <= 0:
        raise ValueError("buffer_deg must be positive")
    if not years:
        raise ValueError("at least one year is required")

    sorted_years = sorted({int(year) for year in years})
    for year in sorted_years:
        if year < 2017:
            raise ValueError("Satellite Embedding V1 annual layers start at 2017")

    output_js = Path(output_js)
    _ensure_parent(output_js)

    west = lon - buffer_deg
    east = lon + buffer_deg
    south = lat - buffer_deg
    north = lat + buffer_deg
    bands_js = ", ".join(f"'{band}'" for band in EMBEDDING_BANDS)

    lines = [
        "// Generated by GeoAnomalyMapper satellite_embeddings.py",
        f"var dataset = ee.ImageCollection('{EARTH_ENGINE_COLLECTION}');",
        f"var embeddingBands = [{bands_js}];",
        f"var aoi = ee.Geometry.Rectangle([{west:.10f}, {south:.10f}, {east:.10f}, {north:.10f}], null, false);",
        "Map.centerObject(aoi, 11);",
        "Map.addLayer(aoi, {color: 'yellow'}, 'Export AOI');",
        "",
    ]

    for year in sorted_years:
        next_year = year + 1
        safe_prefix = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in description_prefix
        )
        image_var = f"embedding_{year}"
        filename = f"{safe_prefix}_{year}_{lat:.5f}_{lon:.5f}".replace("-", "m").replace(".", "p")
        lines.extend(
            [
                f"var {image_var} = dataset",
                f"  .filterDate('{year}-01-01', '{next_year}-01-01')",
                "  .filterBounds(aoi)",
                "  .mosaic()",
                "  .select(embeddingBands)",
                "  .clip(aoi);",
                f"Map.addLayer({image_var}, {{min: -0.3, max: 0.3, bands: ['A01', 'A16', 'A09']}}, '{year} embeddings');",
                "Export.image.toDrive({",
                f"  image: {image_var},",
                f"  description: '{filename}',",
                f"  folder: '{drive_folder}',",
                f"  fileNamePrefix: '{filename}',",
                "  region: aoi,",
                "  scale: 10,",
                "  maxPixels: 10000000000000,",
                "  fileFormat: 'GeoTIFF'",
                "});",
                "",
            ]
        )

    output_js.write_text("\n".join(lines), encoding="utf-8")
    return output_js


def _parse_years(raw_years: Sequence[str]) -> List[int]:
    years: List[int] = []
    for item in raw_years:
        for part in item.split(","):
            part = part.strip()
            if part:
                years.append(int(part))
    return years


def cluster_embedding_anomalies(
    raster_path: Path,
    *,
    n_clusters: int = 32,
    min_cluster_fraction: float = 0.001,
    anomaly_out: Optional[Path] = None,
    summary_out: Optional[Path] = None,
    histogram_bins: int = 1024,
    max_pixels_for_fit: int = 500_000,
) -> Dict:
    """Cap 1: K-means cluster 64D embedding vectors and score outlier pixels.

    Pixels in small or distant-from-center clusters receive high anomaly scores.
    These are statistically unusual surface conditions — potential surface
    expressions of subsurface structures.

    Returns a summary dict with cluster stats and candidate anomaly pixel coords.
    """
    from sklearn.cluster import MiniBatchKMeans

    raster_path = Path(raster_path)
    anomaly_out = Path(anomaly_out) if anomaly_out else None
    summary_out = Path(summary_out) if summary_out else None

    # --- Pass 1: subsample pixels for KMeans fit ---
    sample_rows: List[np.ndarray] = []
    sample_count = 0
    with rasterio.open(raster_path) as src:
        indexes = _embedding_band_indexes(src)
        profile = _single_band_profile(src)
        width, height = src.width, src.height
        crs = str(src.crs)
        transform = list(src.transform)[:6]

        for _, window in src.block_windows(1):
            block, invalid = _read_embedding_block(src, indexes, window)
            # block: (64, rows, cols), C-order flatten to (rows*cols, 64)
            b_flat = block.reshape(len(indexes), -1).T  # (N, 64)
            v_flat = (~invalid).flatten()
            valid_vecs = b_flat[v_flat]
            if valid_vecs.size == 0:
                continue
            if sample_count < max_pixels_for_fit:
                sample_rows.append(valid_vecs)
                sample_count += valid_vecs.shape[0]

    if sample_count == 0:
        raise EmbeddingRasterError(f"No valid pixels found in {raster_path}")

    fit_data = np.vstack(sample_rows)[:max_pixels_for_fit].astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, fit_data.shape[0]),
        batch_size=min(10_000, fit_data.shape[0]),
        n_init=5,
        random_state=42,
    )
    kmeans.fit(fit_data)

    centers = kmeans.cluster_centers_.astype(np.float32)  # (K, 64)
    # Cluster sizes from fit
    labels_fit = kmeans.labels_
    cluster_counts = np.bincount(labels_fit, minlength=n_clusters)
    total_fit = labels_fit.size
    cluster_fractions = cluster_counts / max(total_fit, 1)

    # Anomaly score per cluster: small fraction = unusual
    # Score = 1 - sqrt(cluster_fraction / max_fraction), clamped [0,1]
    max_frac = float(cluster_fractions.max()) if cluster_fractions.max() > 0 else 1.0
    cluster_anomaly = np.sqrt(np.clip(cluster_fractions / max_frac, 0.0, 1.0))
    cluster_anomaly = (1.0 - cluster_anomaly).astype(np.float32)

    anomaly_stats = HistogramStats(0.0, 1.0, histogram_bins)

    # --- Pass 2: assign every pixel an anomaly score ---
    if anomaly_out:
        _ensure_parent(anomaly_out)
        with rasterio.open(raster_path) as src:
            indexes = _embedding_band_indexes(src)
            with rasterio.open(anomaly_out, "w", **profile) as dst:
                dst.set_band_description(1, "embedding_cluster_anomaly_score")
                for _, window in src.block_windows(1):
                    block, invalid = _read_embedding_block(src, indexes, window)
                    b_flat = block.reshape(len(indexes), -1).T  # (N, 64)
                    assigned = kmeans.predict(b_flat.astype(np.float32))
                    scores = cluster_anomaly[assigned].reshape(
                        block.shape[1], block.shape[2]
                    ).astype(np.float32)
                    scores[invalid] = np.nan
                    anomaly_stats.update(scores[~invalid])
                    dst.write(scores, 1, window=window)
    else:
        with rasterio.open(raster_path) as src:
            indexes = _embedding_band_indexes(src)
            for _, window in src.block_windows(1):
                block, invalid = _read_embedding_block(src, indexes, window)
                b_flat = block.reshape(len(indexes), -1).T
                assigned = kmeans.predict(b_flat.astype(np.float32))
                scores = cluster_anomaly[assigned]
                anomaly_stats.update(scores[~invalid.flatten()])

    summary = {
        "input": str(raster_path),
        "n_clusters": int(kmeans.n_clusters),
        "pixels_used_for_fit": int(total_fit),
        "cluster_fractions": cluster_fractions.tolist(),
        "cluster_anomaly_scores": cluster_anomaly.tolist(),
        "anomaly_score_distribution": anomaly_stats.to_dict([5, 25, 50, 75, 90, 95, 99]),
        "anomaly_out": str(anomaly_out) if anomaly_out else None,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    if summary_out:
        _ensure_parent(Path(summary_out))
        Path(summary_out).write_text(
            json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
        )
    return summary


def compute_spatial_anomaly_score(
    raster_path: Path,
    *,
    outer_radius_px: int = 30,
    anomaly_out: Optional[Path] = None,
    summary_out: Optional[Path] = None,
    histogram_bins: int = 1024,
) -> Dict:
    """Cap 3: Score each pixel by how unlike its local neighbourhood it is.

    For each pixel the anomaly score is 1 - dot_product(pixel_embedding,
    neighbourhood_mean_embedding). High scores mark pixels whose surface
    conditions differ from surroundings — typical of subsidence zones,
    thermal anomalies, or disturbed ground above voids.

    Uses scipy.ndimage.uniform_filter per band to compute neighbourhood means
    without loading the full raster into RAM (band-by-band streaming).
    """
    from scipy.ndimage import uniform_filter

    raster_path = Path(raster_path)
    anomaly_out = Path(anomaly_out) if anomaly_out else None
    summary_out = Path(summary_out) if summary_out else None

    size = 2 * outer_radius_px + 1  # filter size in pixels

    with rasterio.open(raster_path) as src:
        indexes = _embedding_band_indexes(src)
        profile = _single_band_profile(src)
        crs = str(src.crs)
        transform_vals = list(src.transform)[:6]
        W, H = src.width, src.height

        # Load all 64 bands at once — embedding tiles are ~163 840 m so a typical
        # exported AOI is small enough to fit in RAM as float32.
        all_bands = src.read(indexes=list(indexes), masked=True).astype(np.float32)

    # all_bands: (64, H, W)
    invalid_any = np.ma.getmaskarray(np.ma.masked_invalid(all_bands)).any(axis=0)  # (H, W)
    data = np.where(invalid_any[np.newaxis], 0.0, all_bands.filled(0.0))  # zero out invalid

    # Per-band neighbourhood mean
    neigh = np.stack(
        [uniform_filter(data[b], size=size, mode="reflect") for b in range(data.shape[0])],
        axis=0,
    ).astype(np.float32)  # (64, H, W)

    # Normalise neighbourhood vectors
    neigh_norm = np.linalg.norm(neigh, axis=0, keepdims=True).clip(min=1e-8)
    neigh_unit = neigh / neigh_norm

    # Pixel norms (already unit-length per AlphaEarth spec, but renorm for safety)
    pix_norm = np.linalg.norm(data, axis=0, keepdims=True).clip(min=1e-8)
    pix_unit = data / pix_norm

    # Dot product pixel vs neighbourhood mean
    dot = np.einsum("bhw,bhw->hw", pix_unit, neigh_unit).astype(np.float32)
    anomaly = np.clip(1.0 - dot, 0.0, 2.0).astype(np.float32)
    anomaly[invalid_any] = np.nan

    stats = HistogramStats(0.0, 2.0, histogram_bins)
    stats.update(anomaly[~invalid_any])

    if anomaly_out:
        _ensure_parent(anomaly_out)
        with rasterio.open(anomaly_out, "w", **profile) as dst:
            dst.set_band_description(1, "embedding_spatial_anomaly_score")
            dst.write(anomaly, 1)

    summary = {
        "input": str(raster_path),
        "outer_radius_px": outer_radius_px,
        "filter_size_px": size,
        "anomaly_score_distribution": stats.to_dict([5, 25, 50, 75, 90, 95, 99]),
        "anomaly_out": str(anomaly_out) if anomaly_out else None,
        "crs": crs,
        "transform": transform_vals,
        "width": W,
        "height": H,
    }
    if summary_out:
        _ensure_parent(Path(summary_out))
        Path(summary_out).write_text(
            json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
        )
    return summary


def compute_temporal_anomaly_trajectory(
    raster_paths: Dict[int, Path],
    *,
    variance_out: Optional[Path] = None,
    trend_out: Optional[Path] = None,
    breakpoint_out: Optional[Path] = None,
    summary_out: Optional[Path] = None,
    histogram_bins: int = 1024,
) -> Dict:
    """Cap 2: Multi-year embedding trajectory analysis.

    For every pixel, computes:
    - trajectory_variance: variance of pairwise dot products across all year pairs
      (0 = perfectly stable, higher = more change)
    - trend_magnitude: magnitude of the linear embedding drift vector
    - breakpoint_year: year of maximum year-over-year change

    Progressive or sudden surface changes over multiple years indicate
    active processes (subsidence, ground disturbance, fluid movement)
    that can reveal underground voids or objects.
    """
    if len(raster_paths) < 2:
        raise ValueError("At least two years required for temporal analysis")

    sorted_years = sorted(raster_paths.keys())
    n_years = len(sorted_years)

    # Load all years band data into a list
    # Each entry: (64, H, W) float32, invalid mask (H, W)
    year_data: List[np.ndarray] = []
    year_invalid: List[np.ndarray] = []
    ref_profile: Optional[Dict] = None
    ref_crs: str = ""
    ref_transform: List[float] = []
    W = H = 0

    for yr in sorted_years:
        path = Path(raster_paths[yr])
        with rasterio.open(path) as src:
            indexes = _embedding_band_indexes(src)
            if ref_profile is None:
                ref_profile = _single_band_profile(src)
                ref_crs = str(src.crs)
                ref_transform = list(src.transform)[:6]
                W, H = src.width, src.height
            else:
                mismatches = []
                if src.width != W or src.height != H:
                    mismatches.append(f"shape {src.height}x{src.width} != reference {H}x{W}")
                if str(src.crs) != ref_crs:
                    mismatches.append(f"CRS {src.crs} != reference {ref_crs}")
                if not np.allclose(
                    np.asarray(guard_transform(src.transform)[:6], dtype=np.float64),
                    np.asarray(ref_transform, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    mismatches.append(
                        f"transform {list(src.transform)[:6]} != reference {ref_transform}"
                    )
                if mismatches:
                    ref_year = sorted_years[0]
                    raise EmbeddingRasterError(
                        f"Temporal raster grid mismatch for year {yr} ({path}) relative to "
                        f"reference year {ref_year} ({raster_paths[ref_year]}): "
                        + "; ".join(mismatches)
                    )
            bands = src.read(indexes=list(indexes), masked=True).astype(np.float32)
        invalid = np.ma.getmaskarray(np.ma.masked_invalid(bands)).any(axis=0)
        data = bands.filled(0.0)
        # Re-normalise for safety
        nrm = np.linalg.norm(data, axis=0, keepdims=True).clip(min=1e-8)
        year_data.append(data / nrm)  # unit vectors
        year_invalid.append(invalid)

    combined_invalid = np.stack(year_invalid, axis=0).any(axis=0)  # (H, W)

    # Pairwise dot products for all year pairs -> variance
    pair_dots: List[np.ndarray] = []
    for i in range(n_years):
        for j in range(i + 1, n_years):
            dot = np.einsum("bhw,bhw->hw", year_data[i], year_data[j]).astype(np.float32)
            pair_dots.append(dot)

    dot_stack = np.stack(pair_dots, axis=0)  # (n_pairs, H, W)
    trajectory_variance = dot_stack.var(axis=0).astype(np.float32)
    trajectory_variance[combined_invalid] = np.nan

    # Year-over-year change magnitude and breakpoint year
    yoy_change: List[np.ndarray] = []
    for i in range(n_years - 1):
        diff = year_data[i + 1] - year_data[i]  # (64, H, W)
        mag = np.linalg.norm(diff, axis=0).astype(np.float32)  # (H, W)
        yoy_change.append(mag)

    yoy_stack = np.stack(yoy_change, axis=0)  # (n_years-1, H, W)
    breakpoint_idx = yoy_stack.argmax(axis=0).astype(np.float32)  # (H, W)
    # Map index to year
    breakpoint_year = np.array(
        [sorted_years[int(i)] for i in breakpoint_idx.flatten()],
        dtype=np.float32,
    ).reshape(H, W)
    breakpoint_year[combined_invalid] = np.nan

    # Linear trend: fit 64D embedding vector as function of year index
    # trend_magnitude = norm of the least-squares slope vector
    year_idx = np.arange(n_years, dtype=np.float32)  # (T,)
    year_idx_c = year_idx - year_idx.mean()
    denom = float((year_idx_c ** 2).sum()) or 1.0
    stacked = np.stack(year_data, axis=0)  # (T, 64, H, W)
    slope = np.einsum("t,tbhw->bhw", year_idx_c, stacked) / denom  # (64, H, W)
    trend_mag = np.linalg.norm(slope, axis=0).astype(np.float32)  # (H, W)
    trend_mag[combined_invalid] = np.nan

    var_stats = HistogramStats(0.0, 1.0, histogram_bins)
    trend_stats = HistogramStats(0.0, 2.0, histogram_bins)
    var_stats.update(trajectory_variance[~combined_invalid])
    trend_stats.update(trend_mag[~combined_invalid])

    assert ref_profile is not None
    for out_path, arr, desc in [
        (variance_out, trajectory_variance, "embedding_trajectory_variance"),
        (trend_out, trend_mag, "embedding_trend_magnitude"),
        (breakpoint_out, breakpoint_year, "embedding_breakpoint_year"),
    ]:
        if out_path:
            out_path = Path(out_path)
            _ensure_parent(out_path)
            with rasterio.open(out_path, "w", **ref_profile) as dst:
                dst.set_band_description(1, desc)
                dst.write(arr, 1)

    summary = {
        "years": sorted_years,
        "n_years": n_years,
        "n_pairs": len(pair_dots),
        "trajectory_variance": var_stats.to_dict([5, 25, 50, 75, 90, 95, 99]),
        "trend_magnitude": trend_stats.to_dict([5, 25, 50, 75, 90, 95, 99]),
        "variance_out": str(variance_out) if variance_out else None,
        "trend_out": str(trend_out) if trend_out else None,
        "breakpoint_out": str(breakpoint_out) if breakpoint_out else None,
        "crs": ref_crs,
        "transform": ref_transform,
        "width": W,
        "height": H,
    }
    if summary_out:
        _ensure_parent(Path(summary_out))
        Path(summary_out).write_text(
            json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
        )
    return summary


def find_similar_sites(
    reference_raster: Path,
    reference_bbox_px: Tuple[int, int, int, int],
    search_rasters: Sequence[Path],
    *,
    top_k: int = 20,
    min_similarity: float = 0.80,
    similarity_out: Optional[Path] = None,
    summary_out: Optional[Path] = None,
) -> List[Dict]:
    """Cap 5: Find pixels whose embeddings are most similar to a reference site.

    reference_bbox_px: (row_min, col_min, row_max, col_max) in pixel coords
    within reference_raster.

    Computes the mean unit embedding of the reference ROI, then scores every
    pixel in each search raster by dot product. Returns top_k candidates
    above min_similarity, with pixel and geographic coordinates.
    """
    reference_raster = Path(reference_raster)
    search_rasters = [Path(p) for p in search_rasters]
    if similarity_out and len(search_rasters) > 1:
        suffix = Path(similarity_out).suffix.lower()
        if suffix in {".tif", ".tiff"}:
            raise EmbeddingRasterError(
                "similarity_out must be a directory when searching multiple rasters; "
                "a single GeoTIFF path would be overwritten for each raster."
            )

    def _similarity_output_for(raster_path: Path) -> Optional[Path]:
        if not similarity_out:
            return None
        base = Path(similarity_out)
        if len(search_rasters) == 1:
            return base
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{raster_path.stem}_similarity.tif"

    # Build reference embedding mean
    with rasterio.open(reference_raster) as src:
        indexes = _embedding_band_indexes(src)
        r0, c0, r1, c1 = reference_bbox_px
        from rasterio.windows import Window
        win = Window(c0, r0, c1 - c0, r1 - r0)
        block, invalid = _read_embedding_block(src, indexes, win)
    valid_vecs = block.reshape(len(indexes), -1).T[~invalid.flatten()]  # (N, 64)
    if valid_vecs.shape[0] == 0:
        raise EmbeddingRasterError("Reference bbox contains no valid embedding pixels")
    ref_mean = valid_vecs.mean(axis=0).astype(np.float64)
    ref_mean /= max(float(np.linalg.norm(ref_mean)), 1e-10)
    ref_mean_f32 = ref_mean.astype(np.float32)

    candidates: List[Dict] = []

    output_rasters: List[str] = []
    for raster_path in search_rasters:
        raster_similarity_out = _similarity_output_for(raster_path)
        with rasterio.open(raster_path) as src:
            indexes = _embedding_band_indexes(src)
            profile = _single_band_profile(src)
            transform = src.transform

            # Score every pixel block-by-block
            if raster_similarity_out:
                _ensure_parent(raster_similarity_out)
                output_rasters.append(str(raster_similarity_out))
                sim_dst = rasterio.open(raster_similarity_out, "w", **profile)
                sim_dst.set_band_description(1, "site_similarity_score")
            else:
                sim_dst = None

            try:
                for _, window in src.block_windows(1):
                    block, invalid = _read_embedding_block(src, indexes, window)
                    # Dot product with reference mean
                    dot = np.einsum("brc,b->rc", block, ref_mean_f32).astype(np.float32)
                    dot[invalid] = np.nan

                    if sim_dst:
                        sim_dst.write(dot, 1, window=window)

                    # Collect top candidates
                    valid_mask = (~invalid) & (dot >= min_similarity)
                    if valid_mask.any():
                        rows_w, cols_w = np.where(valid_mask)
                        # Convert window-relative to absolute pixel coords
                        row_off = int(window.row_off)
                        col_off = int(window.col_off)
                        for rr, cc in zip(rows_w, cols_w):
                            abs_row = int(rr) + row_off
                            abs_col = int(cc) + col_off
                            lon, lat = transform * (abs_col + 0.5, abs_row + 0.5)
                            candidates.append({
                                "raster": str(raster_path),
                                "pixel_row": abs_row,
                                "pixel_col": abs_col,
                                "lat": float(lat),
                                "lon": float(lon),
                                "similarity": float(dot[rr, cc]),
                            })
            finally:
                if sim_dst:
                    sim_dst.close()

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    top = candidates[:top_k]

    summary: Dict = {
        "reference_raster": str(reference_raster),
        "reference_bbox_px": list(reference_bbox_px),
        "search_rasters": [str(p) for p in search_rasters],
        "min_similarity": min_similarity,
        "total_candidates": len(candidates),
        "top_k": top_k,
        "results": top,
    }
    if similarity_out:
        summary["similarity_out"] = str(similarity_out)
        summary["similarity_outputs"] = output_rasters
    if summary_out:
        _ensure_parent(Path(summary_out))
        Path(summary_out).write_text(
            json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
        )
    return top


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Google Satellite Embedding / AlphaEarth 64-band GeoTIFF exports."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a 64-band embedding GeoTIFF and check vector norms."
    )
    inspect_parser.add_argument("raster", type=Path)
    inspect_parser.add_argument("--summary-out", type=Path)
    inspect_parser.add_argument("--histogram-bins", type=int, default=1024)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two annual embedding GeoTIFFs and write a change raster."
    )
    compare_parser.add_argument("--before", required=True, type=Path)
    compare_parser.add_argument("--after", required=True, type=Path)
    compare_parser.add_argument("--change-out", required=True, type=Path)
    compare_parser.add_argument("--similarity-out", type=Path)
    compare_parser.add_argument("--summary-out", type=Path)
    compare_parser.add_argument("--resample-to-before", action="store_true")
    compare_parser.add_argument("--histogram-bins", type=int, default=2048)
    compare_parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Change-score threshold for exact pixel counts. May be repeated.",
    )

    export_parser = subparsers.add_parser(
        "export-ee-script",
        help="Write an Earth Engine Code Editor export script for AlphaEarth GeoTIFFs.",
    )
    export_parser.add_argument("--lat", required=True, type=float)
    export_parser.add_argument("--lon", required=True, type=float)
    export_parser.add_argument("--buffer-deg", required=True, type=float)
    export_parser.add_argument("--years", required=True, nargs="+")
    export_parser.add_argument("--out-js", required=True, type=Path)
    export_parser.add_argument("--description-prefix", default="geoanomaly_alphaearth")
    export_parser.add_argument("--drive-folder", default="GeoAnomalyMapper_AlphaEarth")

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster embedding vectors and score anomalous pixels (Cap 1).",
    )
    cluster_parser.add_argument("raster", type=Path)
    cluster_parser.add_argument("--n-clusters", type=int, default=32)
    cluster_parser.add_argument("--anomaly-out", type=Path)
    cluster_parser.add_argument("--summary-out", type=Path)
    cluster_parser.add_argument("--max-pixels", type=int, default=500_000)

    spatial_parser = subparsers.add_parser(
        "spatial-anomaly",
        help="Score pixels by dissimilarity to local neighbourhood (Cap 3).",
    )
    spatial_parser.add_argument("raster", type=Path)
    spatial_parser.add_argument("--outer-radius-px", type=int, default=30)
    spatial_parser.add_argument("--anomaly-out", type=Path)
    spatial_parser.add_argument("--summary-out", type=Path)

    temporal_parser = subparsers.add_parser(
        "temporal-anomaly",
        help="Multi-year embedding trajectory analysis (Cap 2).",
    )
    temporal_parser.add_argument(
        "--year-raster",
        action="append",
        metavar="YEAR:PATH",
        required=True,
        help="Year and raster path as YEAR:PATH. Repeat for each year.",
    )
    temporal_parser.add_argument("--variance-out", type=Path)
    temporal_parser.add_argument("--trend-out", type=Path)
    temporal_parser.add_argument("--breakpoint-out", type=Path)
    temporal_parser.add_argument("--summary-out", type=Path)

    similar_parser = subparsers.add_parser(
        "find-similar",
        help="Find locations with similar embedding to a reference site (Cap 5).",
    )
    similar_parser.add_argument("--reference", required=True, type=Path)
    similar_parser.add_argument(
        "--bbox-px",
        required=True,
        nargs=4,
        type=int,
        metavar=("ROW_MIN", "COL_MIN", "ROW_MAX", "COL_MAX"),
    )
    similar_parser.add_argument(
        "--search", required=True, nargs="+", type=Path,
        help="One or more rasters to search.",
    )
    similar_parser.add_argument("--top-k", type=int, default=20)
    similar_parser.add_argument("--min-similarity", type=float, default=0.80)
    similar_parser.add_argument("--similarity-out", type=Path)
    similar_parser.add_argument("--summary-out", type=Path)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        summary = inspect_embedding_raster(
            args.raster,
            summary_out=args.summary_out,
            histogram_bins=args.histogram_bins,
        )
    elif args.command == "compare":
        thresholds = (
            tuple(float(t) for t in args.threshold)
            if args.threshold
            else DEFAULT_CHANGE_THRESHOLDS
        )
        summary = compare_embedding_rasters(
            args.before,
            args.after,
            change_out=args.change_out,
            similarity_out=args.similarity_out,
            summary_out=args.summary_out,
            resample_to_before=args.resample_to_before,
            histogram_bins=args.histogram_bins,
            change_thresholds=thresholds,
        )
    elif args.command == "cluster":
        summary = cluster_embedding_anomalies(
            args.raster,
            n_clusters=args.n_clusters,
            anomaly_out=args.anomaly_out,
            summary_out=args.summary_out,
            max_pixels_for_fit=args.max_pixels,
        )
    elif args.command == "spatial-anomaly":
        summary = compute_spatial_anomaly_score(
            args.raster,
            outer_radius_px=args.outer_radius_px,
            anomaly_out=args.anomaly_out,
            summary_out=args.summary_out,
        )
    elif args.command == "temporal-anomaly":
        year_rasters: Dict[int, Path] = {}
        for entry in args.year_raster:
            yr_str, path_str = entry.split(":", 1)
            year_rasters[int(yr_str)] = Path(path_str)
        summary = compute_temporal_anomaly_trajectory(
            year_rasters,
            variance_out=args.variance_out,
            trend_out=args.trend_out,
            breakpoint_out=args.breakpoint_out,
            summary_out=args.summary_out,
        )
    elif args.command == "find-similar":
        results = find_similar_sites(
            reference_raster=args.reference,
            reference_bbox_px=tuple(args.bbox_px),
            search_rasters=args.search,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            similarity_out=args.similarity_out,
            summary_out=args.summary_out,
        )
        summary = {"results": results}
    elif args.command == "export-ee-script":
        output = write_earth_engine_export_script(
            lat=args.lat,
            lon=args.lon,
            buffer_deg=args.buffer_deg,
            years=_parse_years(args.years),
            output_js=args.out_js,
            description_prefix=args.description_prefix,
            drive_folder=args.drive_folder,
        )
        summary = {"earth_engine_script": str(output)}
    else:
        parser.error(f"Unhandled command: {args.command}")

    print(json.dumps(summary, indent=2, default=_json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
