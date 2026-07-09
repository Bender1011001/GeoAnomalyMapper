#!/usr/bin/env python3
"""
Biondi Doppler Tomography - Exploration Orchestrator
======================================================
This script orchestrates the large-scale mapping using the SAR Doppler
Tomography pipeline, following a phased approach:

    Phase 1: Reference/Exploratory Targets
    Phase 2: Regional Scan (California)
    Phase 3: Continental Scan (USA)

Execution requires NASA Earthdata authentication to download raw Sentinel-1
SLC data via the Alaska Satellite Facility (ASF) API.

Provide authentication either via environment variables or a .env file:
    EARTHDATA_TOKEN=your_earthdata_bearer_token
    # or
    EARTHDATA_USERNAME=your_username
    EARTHDATA_PASSWORD=your_password
"""

import os
import sys
import logging
import argparse
import traceback
import time
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple
from datetime import datetime

import numpy as np

# Set up paths and environment
from json_utils import dump_strict_json
from project_paths import DATA_DIR

import slc_data_fetcher
import sar_vibrometry
import pinn_vibro_inversion
import visualize_3d_subsurface

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    force=True
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

EXPLORE_DIR: Path = DATA_DIR / "biondi_exploration"
# Avoid creating output directories at import time. Pipeline/report writers ensure
# this directory exists immediately before writing.

DEFAULT_EMBEDDING_SURFACE_PRIOR_WEIGHT = 0.15
ANOMALY_COUNT_KEYS = ("anomaly_count", "anomalies_detected")


def _coerce_anomaly_count(value) -> Optional[int]:
    """Return a non-negative anomaly count, or None if the value is unusable."""
    if value is None:
        return None
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(count, 0)


def normalize_anomaly_count(record: Dict, default: int = 0) -> int:
    """Read anomaly counts from either current or legacy result-schema keys."""
    if not isinstance(record, dict):
        return default
    for key in ANOMALY_COUNT_KEYS:
        count = _coerce_anomaly_count(record.get(key))
        if count is not None:
            return count
    return default


def _set_anomaly_count_fields(record: Dict, anomaly_count) -> int:
    """Emit both anomaly-count keys for backward-compatible downstream consumers."""
    count = _coerce_anomaly_count(anomaly_count)
    if count is None:
        count = 0
    record["anomaly_count"] = count
    record["anomalies_detected"] = count
    return count


def _target_local_chip_shape(profile: Dict) -> Tuple[int, int]:
    """Resolve a bounded target-local SLC chip shape from an execution profile."""
    max_dim = int(profile.get("target_slc_chip_pixels", 4096))
    max_dim = max(64, min(max_dim, 4096))
    return (max_dim, max_dim)


def _target_safe_name(target: Dict) -> str:
    """Return the filesystem-safe target name used by pipeline outputs."""
    return target["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


def _resolve_embedding_raster_candidates(
    target_name: str,
    embedding_dir: Optional[Path] = None,
    embedding_rasters: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Resolve explicit embedding raster paths plus legacy directory/name fallbacks."""
    candidates = [Path(p) for p in (embedding_rasters or [])]
    if embedding_dir:
        ed = Path(embedding_dir)
        candidates.extend([
            ed / f"{target_name}_fused_anomaly.tif",
            ed / f"{target_name}_spatial_anomaly.tif",
            ed / f"{target_name}_cluster_anomaly.tif",
        ])
    return candidates


def load_cli_env(env_path: Optional[Path] = None) -> None:
    """Load .env credentials only for CLI execution, not import-time use."""
    print(">>> Orchestrator starting...", flush=True)
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print(f">>> Loading local environment from {env_path}", flush=True)
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k] = v
                    sensitive_markers = ("PASSWORD", "USERNAME", "TOKEN", "SECRET", "KEY")
                    if any(marker in k.upper() for marker in sensitive_markers):
                        safe_value = (
                            slc_data_fetcher.normalize_earthdata_token(v)
                            if "TOKEN" in k.upper()
                            else v
                        )
                        log_v = f"<{slc_data_fetcher.describe_secret_presence(safe_value)}>"
                    else:
                        log_v = v
                    print(f"    - Set {k}={log_v}", flush=True)
    print(f">>> DATA_DIR resolved to: {DATA_DIR}", flush=True)
    print(">>> Logger configured.", flush=True)


# ============================================================
# PIPELINE RESOLUTION PROFILES
# ============================================================
# Controls tradeoff between speed and detection quality.
#
# TRIVIAL COLLAPSE FIX (2025-03-10):
#   - deep_prior_weight reduced from 0.1/0.05/0.02 -> 0.01/0.005/0.001
#     (Old values penalized deep anomalies so heavily the PINN predicted solid rock everywhere)
#   - data_weight increased from 10.0 -> 50.0
#     (Forces the PINN to trust satellite surface observations over smoothing)
#   - domain_width_m and max_depth_m added per profile
#     (Old: 5000m domain / 32 grid = 156m voxels -> anomalies averaged out)
#   - excitation_frequency_hz added per profile
#     (Low freq penetrates deeper; high freq resolves shallow detail)
#
RESOLUTION_PROFILES = {
    "quick": {
        "epochs": 500,
        "grid_nx": 64,
        "grid_ny": 64,
        "grid_nz": 32,
        "domain_width_m": 800.0,
        "max_depth_m": 500.0,
        # REBALANCED WEIGHTS (2026-03-10 expert fix for trivial collapse)
        "physics_weight": 1.0,
        "data_weight": 20.0,
        "sparsity_weight": 1.0,
        "regularization_weight": 0.1,
        "deep_prior_weight": 1.0,
        "excitation_frequency_hz": 2.0,
        "synthetic_grid_size": 256,
        "num_sub_apertures": 3,
        "batch_size_collocation": 4096,
        "batch_size_boundary": 1024,
        "gradient_accumulation_steps": 1,
        "hidden_layers": 6,
        "hidden_neurons": 256,
    },
    "standard": {
        "epochs": 3000,
        "grid_nx": 128,
        "grid_ny": 128,
        "grid_nz": 64,
        "domain_width_m": 800.0,
        "max_depth_m": 1000.0,
        # REBALANCED WEIGHTS (2026-03-10 expert fix for trivial collapse)
        "physics_weight": 1.0,                # Was implicit 0.0001 - now primary driver
        "data_weight": 20.0,                  # Was 50.0 - lowered (Sentinel-1 is noisy)
        "sparsity_weight": 1.0,              # Fix for real data void collapse
        "regularization_weight": 0.1,         # Was 0.01 - promotes sharp boundaries
        "deep_prior_weight": 1.0,             # Anchors bottom of grid to solid rock
        "excitation_frequency_hz": 1.0,
        "synthetic_grid_size": 512,
        "num_sub_apertures": 5,
        "batch_size_collocation": 4096,
        "batch_size_boundary": 1024,
        "gradient_accumulation_steps": 1,
        "hidden_layers": 8,
        "hidden_neurons": 512,
    },
    "high": {
        "epochs": 5000,
        "grid_nx": 128,
        "grid_ny": 128,
        "grid_nz": 128,
        "domain_width_m": 800.0,
        "max_depth_m": 1000.0,
        # REBALANCED WEIGHTS (2026-03-10 expert fix for trivial collapse)
        "physics_weight": 1.0,                # Was implicit 0.0001 - now primary driver
        "data_weight": 20.0,                  # Was 50.0 - lowered (Sentinel-1 is noisy)
        "sparsity_weight": 1.0,              # Fix for real data void collapse
        "regularization_weight": 0.1,         # Was 0.01 - promotes sharp boundaries
        "deep_prior_weight": 1.0,             # Anchors bottom of grid to solid rock
        "excitation_frequency_hz": 0.5,       # Deepest penetration
        "synthetic_grid_size": 1024,
        "num_sub_apertures": 7,
        # H100 OPTIMIZATION: (80GB VRAM)
        "use_float64_physics": False,         # TF32 is sufficient, enables torch.compile
        "batch_size_collocation": 65536,      # Scaled down to prevent H100 OOM
        "batch_size_boundary": 16384,
        "gradient_accumulation_steps": 1,
        "hidden_layers": 8,
        "hidden_neurons": 768,                # float32 frees VRAM for bigger model
    },
    "deep": {
        "epochs": 6000,
        "grid_nx": 128,
        "grid_ny": 128,
        "grid_nz": 160,
        "domain_width_m": 2000.0,
        "max_depth_m": 5000.0,
        "physics_weight": 1.0,
        "data_weight": 20.0,
        "sparsity_weight": 1.0,
        "regularization_weight": 0.1,
        "deep_prior_weight": 1.0,
        "excitation_frequency_hz": 0.25,
        "synthetic_grid_size": 1024,
        "num_sub_apertures": 7,
        "batch_size_collocation": 32768,
        "batch_size_boundary": 8192,
        "gradient_accumulation_steps": 2,
        "hidden_layers": 8,
        "hidden_neurons": 768,
        "use_float64_physics": False,
    },
}


def select_depth_slices(max_depth_m: float) -> List[float]:
    """Choose report slices that keep deep runs visible without overcrowding plots."""
    standard_slices = [50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    selected = [float(depth) for depth in standard_slices if depth <= max_depth_m]
    if not selected:
        return [float(max_depth_m)]
    if selected[-1] < max_depth_m:
        selected.append(float(max_depth_m))
    if len(selected) <= 6:
        return selected
    positions = [round(i * (len(selected) - 1) / 5) for i in range(6)]
    return [selected[index] for index in positions]


# ============================================================
# TARGET REGIONS
# ============================================================
PHASE_1_TARGETS = [
    {
        "name": "Vacaville Target Area",
        "lat": 38.3512, 
        "lon": -121.986,
        "buffer_deg": 0.1,
        "description": "7626 Clement Rd Vacaville CA surrounds",
        "expected_depth_m": 120,
        "expected_void_type": "unknown",
    },
    {
        "name": "Great Pyramid of Giza (Khufu)",
        "lat": 29.9792,
        "lon": 31.1342,
        "buffer_deg": 0.15,
        "description": "Biondi's main target.",
        "expected_depth_m": 500,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Khafre Pyramid (Giza)",
        "lat": 29.9761,
        "lon": 31.1313,
        "buffer_deg": 0.1,
        "description": "Biondi claims strongest anomalies here.",
        "expected_depth_m": 888,
        "expected_void_type": "artificial_cavity",
    }
]

PHASE_2_TARGETS = [
    {
        "name": "Death Valley Region (CA)",
        "lat": 36.5323,
        "lon": -116.9325,
        "buffer_deg": 0.5,
        "description": "Geologically active zone, potential for undocumented structural anomalies.",
        "expected_depth_m": 500,
        "expected_void_type": "unknown",
    },
    {
        "name": "Mother Lode Gold Belt (CA)",
        "lat": 38.3333,
        "lon": -120.8333,
        "buffer_deg": 0.5,
        "description": "Historical mining region; locating abandoned interconnected tunnel systems.",
        "expected_depth_m": 200,
        "expected_void_type": "mine_workings",
    },
    {
        "name": "Salton Sea Basin (CA)",
        "lat": 33.3286,
        "lon": -115.8434,
        "buffer_deg": 0.5,
        "description": "High geothermal activity, tracking subsurface fluid conduits and crustal thinning.",
        "expected_depth_m": 1000,
        "expected_void_type": "geothermal",
    },
]

PHASE_3_TARGETS = [
    {
        "name": "Nevada Test Site (NV)",
        "lat": 37.1167,
        "lon": -116.0500,
        "buffer_deg": 0.2,
        "description": "Detecting artificial cavities resulting from historical underground testing.",
        "expected_depth_m": 600,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Yellowstone Caldera (WY)",
        "lat": 44.4280,
        "lon": -110.5885,
        "buffer_deg": 0.5,
        "description": "Mapping the deep magmatic plumbing system.",
        "expected_depth_m": 5000,
        "expected_void_type": "magma_chamber",
    },
    {
        "name": "Mammoth Cave System (KY)",
        "lat": 37.1862,
        "lon": -86.1005,
        "buffer_deg": 0.1,
        "description": "World's longest known cave system - 680+ km mapped passages.",
        "expected_depth_m": 120,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "New Madrid Seismic Zone (MO)",
        "lat": 36.5400,
        "lon": -89.5800,
        "buffer_deg": 0.3,
        "description": "Deep fault structure beneath sedimentary cover.",
        "expected_depth_m": 2000,
        "expected_void_type": "fault_zone",
    },
]


# ============================================================
# SYNTHETIC DATA GENERATION (Proper Biondi-simulated)
# ============================================================
def generate_proper_synthetic_slc(
    target: Dict,
    output_path: str,
    grid_size: int = 512,
    num_sub_apertures: int = 5,
    overlap_fraction: float = 0.3,
) -> str:
    """
    Generate a physically plausible synthetic SLC burst for a specific
    geological target using the sar_vibrometry module's generator.
    
    This uses the proper Rayleigh-distributed amplitude + phase modulation
    approach that embeds known anomalies, NOT random noise.
    """
    name = target["name"]
    expected_type = target.get("expected_void_type", "natural_cave")

    # Tune anomaly count and size based on target type
    if expected_type == "natural_cave":
        n_anomalies = 4
        noise_level = 0.05
    elif expected_type == "mine_workings":
        n_anomalies = 6
        noise_level = 0.08
    elif expected_type == "fault_zone":
        n_anomalies = 3
        noise_level = 0.12
    elif expected_type == "artificial_cavity":
        n_anomalies = 5
        noise_level = 0.04
    elif expected_type == "magma_chamber":
        n_anomalies = 2
        noise_level = 0.15
    elif expected_type == "geothermal":
        n_anomalies = 4
        noise_level = 0.10
    else:
        n_anomalies = 3
        noise_level = 0.10

    logger.info(
        f"Generating synthetic SLC: {name} "
        f"(type={expected_type}, anomalies={n_anomalies}, "
        f"noise={noise_level}, grid={grid_size})"
    )

    # Use the vibrometry module's proper synthetic generator
    result_path = sar_vibrometry.generate_synthetic_vibration_test(
        output_path=output_path,
        grid_size=grid_size,
        num_anomalies=n_anomalies,
        noise_level=noise_level,
        num_sub_apertures=num_sub_apertures,
        overlap_fraction=overlap_fraction,
    )
    return result_path


def _nearest_resize_2d(arr: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize a 2-D diagnostic mask with nearest-neighbour indexing."""
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if arr.shape == (target_h, target_w):
        return arr
    row_idx = np.rint(np.linspace(0, arr.shape[0] - 1, target_h)).astype(int)
    col_idx = np.rint(np.linspace(0, arr.shape[1] - 1, target_w)).astype(int)
    row_idx = np.clip(row_idx, 0, arr.shape[0] - 1)
    col_idx = np.clip(col_idx, 0, arr.shape[1] - 1)
    return arr[row_idx[:, None], col_idx[None, :]]


def write_synthetic_positive_control_diagnostics(
    *,
    slc_path: str,
    vib_amp_path: Optional[str],
    wave_speed_path: Optional[str],
    void_prob_path: Optional[str],
    output_dir: Path,
    anomaly_count: int,
    void_threshold: float,
    min_anomaly_voxels: int,
    background_wave_speed: float,
    void_speed_threshold_ratio: float = 0.7,
) -> Optional[str]:
    """Write positive-control diagnostics for synthetic fallback runs.

    The diagnostics are observational only: they summarize whether the planted
    synthetic 2-D mask survived vibrometry, whether the PINN emitted a low-speed
    volume, and whether visualization thresholds could see connected components.
    They do not create or modify detections.
    """
    gt_path = str(slc_path).replace(".npy", "_ground_truth.npy")
    if not Path(gt_path).exists():
        return None

    diagnostics: Dict[str, object] = {
        "schema_version": 1,
        "slc_path": str(slc_path),
        "ground_truth_path": gt_path,
        "anomaly_count": int(anomaly_count),
        "anomalies_detected": int(anomaly_count),
        "void_threshold": float(void_threshold),
        "min_anomaly_voxels": int(min_anomaly_voxels),
    }

    gt = np.load(gt_path).astype(np.float32)
    active_gt = gt > 0.05
    diagnostics["ground_truth"] = {
        "active_fraction": float(active_gt.mean()),
        "max": float(np.nanmax(gt)),
        "mean": float(np.nanmean(gt)),
    }

    vib_ratio = None
    if vib_amp_path and Path(vib_amp_path).exists():
        vib = np.load(vib_amp_path).astype(np.float32)
        gt_for_vib = _nearest_resize_2d(gt, vib.shape)
        active_for_vib = gt_for_vib > 0.05
        if active_for_vib.any() and (~active_for_vib).any():
            active_mean = float(np.nanmean(vib[active_for_vib]))
            inactive_mean = float(np.nanmean(vib[~active_for_vib]))
            vib_ratio = active_mean / max(inactive_mean, 1e-12)
            diagnostics["vibrometry"] = {
                "active_mean": active_mean,
                "inactive_mean": inactive_mean,
                "active_to_inactive_ratio": float(vib_ratio),
                "max": float(np.nanmax(vib)),
                "mean": float(np.nanmean(vib)),
                "std": float(np.nanstd(vib)),
            }

    void_max = None
    voxels_above_threshold = 0
    if void_prob_path and Path(void_prob_path).exists():
        void_prob = np.load(void_prob_path).astype(np.float32)
        void_max = float(np.nanmax(void_prob))
        voxels_above_threshold = int(np.sum(void_prob > void_threshold))
        diagnostics["void_probability"] = {
            "max": void_max,
            "mean": float(np.nanmean(void_prob)),
            "std": float(np.nanstd(void_prob)),
            "voxels_above_threshold": voxels_above_threshold,
        }

    wave_speed_min = None
    physical_void_speed = float(background_wave_speed * void_speed_threshold_ratio)
    if wave_speed_path and Path(wave_speed_path).exists():
        wave_speed = np.load(wave_speed_path).astype(np.float32)
        wave_speed_min = float(np.nanmin(wave_speed))
        diagnostics["wave_speed"] = {
            "min": wave_speed_min,
            "max": float(np.nanmax(wave_speed)),
            "mean": float(np.nanmean(wave_speed)),
            "std": float(np.nanstd(wave_speed)),
            "physical_void_speed_threshold": physical_void_speed,
        }

    if anomaly_count > 0:
        primary_blocker = "none_detected"
    elif vib_ratio is not None and np.isfinite(vib_ratio) and vib_ratio < 1.10:
        primary_blocker = "vibrometry_input_weakness"
    elif wave_speed_min is not None and np.isfinite(wave_speed_min) and wave_speed_min > physical_void_speed:
        primary_blocker = "pinn_collapse_no_low_velocity_volume"
    elif void_max is not None and np.isfinite(void_max) and void_max < void_threshold:
        primary_blocker = "thresholding_or_void_probability_calibration"
    elif voxels_above_threshold < min_anomaly_voxels:
        primary_blocker = "connected_component_size_filter"
    else:
        primary_blocker = "undetermined_observability_gap"
    diagnostics["primary_blocker"] = primary_blocker

    output_dir.mkdir(parents=True, exist_ok=True)
    diag_path = output_dir / "synthetic_positive_control_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        dump_strict_json(diagnostics, f, indent=2)
    logger.info(f"Synthetic positive-control diagnostics saved: {diag_path}")
    return str(diag_path)


# ============================================================
# PIPELINE EXECUTION
# ============================================================
def execute_biondi_pipeline_for_target(
    target: Dict,
    credentials: Dict[str, str],
    use_synthetic_fallback: bool = False,
    resolution: str = "standard",
    use_embeddings: bool = False,
    embedding_dir: Optional[Path] = None,
    embedding_rasters: Optional[Sequence[Path]] = None,
    surface_prior_weight: Optional[float] = None,
    excitation_frequency_hz: Optional[float] = None,
    locked_sentinel1_products: Optional[Sequence[Dict]] = None,
    require_locked_sentinel1: bool = False,
) -> Dict:
    """
    Executes the end-to-end Biondi SAR Doppler pipeline for a specific
    geographical target.

    Returns a result dict with status, paths, anomaly count, and timing.
    """
    profile = RESOLUTION_PROFILES.get(resolution, RESOLUTION_PROFILES["standard"])
    name = _target_safe_name(target)
    start_time = time.time()

    result = {
        "target_name": target["name"],
        "coordinates": {"lat": target["lat"], "lon": target["lon"]},
        "resolution_profile": resolution,
        "status": "failed",
        "anomaly_count": 0,
        "anomalies_detected": 0,
        "top_deep_targets": [],
        "data_source": "none",
        "outputs": {},
        "elapsed_seconds": 0,
        "error": None,
        "embedding_prior_used": False,
        "locked_product_enforced": False,
        "locked_product_ids": [],
    }

    logger.info(f"\n{'='*70}\nSTARTING PIPELINE FOR TARGET: {target['name']}\n{'='*70}")
    logger.info(f"Description:  {target['description']}")
    logger.info(f"Coordinates:  Lat {target['lat']}, Lon {target['lon']} (Buffer: {target['buffer_deg']} deg)")
    logger.info(f"Resolution:   {resolution} (epochs={profile['epochs']}, grid={profile['grid_nx']}x{profile['grid_ny']}x{profile['grid_nz']})")

    target_out_dir = EXPLORE_DIR / name
    target_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # STEP 1: SLC Data Acquisition
    # --------------------------------------------------------
    # Priority order: 1) Umbra X-band (free, 16cm res, Spotlight mode)
    #                 2) Sentinel-1 C-band (free, 5x20m res, TOPSAR mode)
    # Umbra is preferred because Spotlight mode provides the 60s dwell time
    # needed for Doppler vibrometry. Sentinel-1's 0.1s TOPSAR sweep cannot
    # track sub-second micro-vibrations.
    logger.info("\n--- STEP 1: SLC Data Acquisition (Umbra -> Sentinel-1 -> Synthetic) ---")
    earthdata_auth = slc_data_fetcher.resolve_earthdata_auth(
        earthdata_username=credentials.get("EARTHDATA_USERNAME"),
        earthdata_password=credentials.get("EARTHDATA_PASSWORD"),
        earthdata_token=(
            credentials.get("EARTHDATA_TOKEN")
            or credentials.get("EARTHDATA_BEARER_TOKEN")
        ),
    )
    has_earthdata_auth = earthdata_auth.get("mode") in {"token", "credentials"}

    bbox = slc_data_fetcher._build_search_bbox(
        lat=target['lat'],
        lon=target['lon'],
        buffer_deg=target['buffer_deg']
    )
    locked_sentinel1_products = list(locked_sentinel1_products or [])
    locked_product_ids = [
        str(product.get("product_id"))
        for product in locked_sentinel1_products
        if isinstance(product, dict) and product.get("product_id")
    ]
    result["locked_product_ids"] = locked_product_ids
    if require_locked_sentinel1:
        if use_synthetic_fallback:
            msg = "Locked Sentinel-1 execution cannot use synthetic fallback."
            logger.error(msg)
            result["error"] = msg
            result["elapsed_seconds"] = time.time() - start_time
            return result
        if len(locked_sentinel1_products) != 1 or len(locked_product_ids) != 1:
            msg = (
                "Locked Sentinel-1 execution currently requires exactly one locked product "
                f"with product_id metadata; received {len(locked_sentinel1_products)} product(s) "
                f"and {len(locked_product_ids)} product_id value(s)."
            )
            logger.error(msg)
            result["error"] = msg
            result["elapsed_seconds"] = time.time() - start_time
            return result
        result["locked_product_enforced"] = True

    slc_filepath = None
    if require_locked_sentinel1:
        locked_product = dict(locked_sentinel1_products[0])
        locked_product_id = locked_product_ids[0]
        locked_granule_name = (
            locked_product.get("granule_name")
            or locked_product.get("product_name")
            or locked_product.get("file_id")
            or locked_product.get("product_id")
        )
        if not locked_granule_name:
            msg = f"Locked product {locked_product_id!r} lacks a granule/product name for ASF download."
            logger.error(msg)
            result["error"] = msg
            result["elapsed_seconds"] = time.time() - start_time
            return result
        locked_product["granule_name"] = str(locked_granule_name)
        logger.info(
            "Locked Sentinel-1 product required; skipping Umbra/search fallback and using %s",
            locked_product_id,
        )
        try:
            download_dir = target_out_dir / "raw_slc"
            downloaded_path = slc_data_fetcher.download_slc_product(
                product=locked_product,
                output_dir=download_dir,
                earthdata_username=earthdata_auth.get("username"),
                earthdata_password=earthdata_auth.get("password"),
                earthdata_token=earthdata_auth.get("token"),
                expected_product_id=locked_product_id,
                require_product_identity=True,
            )
            if downloaded_path:
                logger.info("Extracting target-local chip from locked Sentinel-1 SAFE file...")
                extracted = slc_data_fetcher.extract_slc_burst(
                    Path(downloaded_path),
                    output_dir=target_out_dir / "bursts",
                    target_lat=target["lat"],
                    target_lon=target["lon"],
                    target_bbox=bbox,
                    target_chip_shape=_target_local_chip_shape(profile),
                )
                if extracted:
                    slc_filepath = str(extracted[0])
                    result["outputs"]["target_local_slc"] = slc_filepath
                    target_local_metadata = Path(slc_filepath).with_name(
                        f"{Path(slc_filepath).stem}_metadata.json"
                    )
                    if target_local_metadata.exists():
                        result["outputs"]["target_local_slc_metadata"] = str(target_local_metadata)
                    result["data_source"] = "sentinel1_cband_locked"
            try:
                zip_path = Path(downloaded_path)
                if zip_path.exists() and zip_path.suffix.lower() == ".zip":
                    zip_path.unlink()
                    logger.info(f"Deleted ZIP to save disk: {zip_path.name}")
            except Exception as e:
                logger.warning(f"Could not delete ZIP: {e}")
        except Exception as e:
            msg = f"Locked Sentinel-1 acquisition failed: {slc_data_fetcher.safe_exception_summary(e)}"
            logger.error(msg)
            result["error"] = msg
            result["elapsed_seconds"] = time.time() - start_time
            return result
    elif use_synthetic_fallback:
        logger.info("Forcing synthetic data generation (skipping real SLC download for speed).")
    else:
        # === TRY UMBRA FIRST (best for Doppler tomography) ===
        try:
            logger.info("Searching Umbra Space Open Data (X-band, 16cm-1m, Spotlight)...")
            umbra_products = slc_data_fetcher.search_umbra_open_data(
                bbox=bbox,
                max_results=3
            )
            if umbra_products:
                logger.info(f"Found {len(umbra_products)} Umbra X-band SLC products!")
                for up in umbra_products:
                    downloaded_path = slc_data_fetcher.download_umbra_slc(
                        product=up,
                        output_dir=target_out_dir / "raw_slc" / "umbra"
                    )
                    if downloaded_path:
                        logger.info("Extracting SLC from Umbra GeoTIFF...")
                        extracted = slc_data_fetcher.extract_umbra_slc_burst(
                            Path(downloaded_path),
                            output_dir=target_out_dir / "bursts"
                        )
                        if extracted:
                            slc_filepath = str(extracted[0])
                            result["data_source"] = f"umbra_xband ({up.get('resolution_m', '?')}m)"
                            logger.info(f"Using Umbra X-band SLC: {up.get('granule_name')} "
                                        f"(resolution: {up.get('resolution_m', '?')}m)")
                            break
            else:
                logger.info("No Umbra data available for this area - trying Sentinel-1...")
        except Exception as e:
            logger.warning(f"Umbra data acquisition failed (non-fatal): {e}")

        # === FALLBACK TO SENTINEL-1 (lower resolution but broader coverage) ===
        if not slc_filepath:
            try:
                slc_products = slc_data_fetcher.search_sentinel1_slc(
                    bbox=bbox,
                    max_results=1
                )

                if slc_products and has_earthdata_auth:
                    logger.info(f"Found {len(slc_products)} Sentinel-1 products. Downloading first...")
                    logger.warning("NOTE: Sentinel-1 C-band (5x20m, TOPSAR) has limited Doppler sensitivity. "
                                   "Results will be lower resolution than X-band Spotlight data.")
                    download_dir = target_out_dir / "raw_slc"
                    downloaded_path = slc_data_fetcher.download_slc_product(
                        product=slc_products[0],
                        output_dir=download_dir,
                        earthdata_username=earthdata_auth.get("username"),
                        earthdata_password=earthdata_auth.get("password"),
                        earthdata_token=earthdata_auth.get("token"),
                    )
                    if downloaded_path:
                        logger.info("Extracting target-local chip from downloaded SAFE file...")
                        extracted = slc_data_fetcher.extract_slc_burst(
                            Path(downloaded_path),
                            output_dir=target_out_dir / "bursts",
                            target_lat=target["lat"],
                            target_lon=target["lon"],
                            target_bbox=bbox,
                            target_chip_shape=_target_local_chip_shape(profile),
                        )
                        if extracted:
                            slc_filepath = str(extracted[0])
                            result["outputs"]["target_local_slc"] = slc_filepath
                            target_local_metadata = Path(slc_filepath).with_name(
                                f"{Path(slc_filepath).stem}_metadata.json"
                            )
                            if target_local_metadata.exists():
                                result["outputs"]["target_local_slc_metadata"] = str(target_local_metadata)
                            result["data_source"] = "sentinel1_cband"
                        # Delete ZIP to save disk (5.5GB+ per file)
                        try:
                            zip_path = Path(downloaded_path)
                            if zip_path.exists():
                                zip_path.unlink()
                                logger.info(f"Deleted ZIP to save disk: {zip_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete ZIP: {e}")
                elif not has_earthdata_auth:
                    logger.warning(
                        "EARTHDATA authentication not found - cannot download Sentinel-1 data. "
                        "Set EARTHDATA_TOKEN/EARTHDATA_BEARER_TOKEN or "
                        "EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
                    )
                    logger.warning(
                        "Earthdata auth status: %s",
                        slc_data_fetcher.describe_earthdata_auth(earthdata_auth),
                    )
                else:
                    logger.warning(f"No Sentinel-1 SLC products found over {target['name']}.")
            except Exception as e:
                logger.warning(f"SLC acquisition failed (non-fatal): {e}")

    # Fallback: generate proper synthetic data
    if not slc_filepath and use_synthetic_fallback:
        logger.warning("Falling back to Biondi-simulated synthetic data.")
        slc_filepath = str(target_out_dir / f"{name}_synthetic_burst.npy")
        generate_proper_synthetic_slc(
            target=target,
            output_path=slc_filepath,
            grid_size=profile["synthetic_grid_size"],
            num_sub_apertures=profile["num_sub_apertures"],
            overlap_fraction=0.3,
        )
        result["data_source"] = "synthetic"

    if not slc_filepath:
        msg = f"Failed to acquire SLC data for {target['name']}. Skipping."
        logger.error(msg)
        result["error"] = msg
        result["elapsed_seconds"] = time.time() - start_time
        return result

    # --------------------------------------------------------
    # STEP 2: Doppler Vibrometry (Sub-aperture Processing)
    # --------------------------------------------------------
    logger.info("\n--- STEP 2: Sub-aperture Doppler Processing ---")
    vibro_out_dir = target_out_dir / "vibrometry"
    vibro_results = sar_vibrometry.run_vibrometry_pipeline(
        slc_path=slc_filepath,
        output_dir=str(vibro_out_dir),
        config={
            "num_sub_apertures": profile["num_sub_apertures"],
            "overlap_fraction": 0.3,
        }
    )

    vib_amp_path = vibro_results.get("vibration_amplitude_npy")
    vib_freq_path = vibro_results.get("vibration_frequency_npy")

    # Delete burst .npy files to save disk (~1GB)
    burst_dir = target_out_dir / "bursts"
    if burst_dir.exists():
        import shutil
        burst_size_mb = sum(f.stat().st_size for f in burst_dir.rglob('*') if f.is_file()) / (1024**2)
        shutil.rmtree(burst_dir, ignore_errors=True)
        logger.info(f"Deleted burst dir to save disk: {burst_size_mb:.0f} MB freed")

    if not vib_amp_path:
        msg = "Vibrometry processing failed."
        logger.error(msg)
        result["error"] = msg
        result["elapsed_seconds"] = time.time() - start_time
        return result

    result["outputs"]["vibration_amplitude"] = vib_amp_path

    # --------------------------------------------------------
    # STEP 3: 3D PINN Inversion (Helmholtz Wave Equation)
    # --------------------------------------------------------
    logger.info("\n--- STEP 3: 3D PINN Vibro-Elastic Inversion ---")

    # Free GPU memory from vibrometry before loading PINN model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU cleanup: {torch.cuda.memory_allocated(0)/(1024**2):.0f} MB allocated after vibrometry cleanup")
    except Exception:
        pass

    pinn_out_dir = target_out_dir / "pinn_inversion"
    vib_map = np.load(vib_amp_path)
    freq_map = np.load(vib_freq_path) if vib_freq_path else None

    # Adapt domain to target: use target's expected depth to set max_depth
    # and use buffer_deg to estimate a sensible domain_width
    target_depth = target.get("expected_depth_m", 500)
    # Domain depth should be ~2x expected depth to capture full anomaly extent
    adaptive_max_depth = min(profile.get("max_depth_m", 1000.0), max(target_depth * 2.0, 200.0))
    # Use profile's domain width (already shrunk to 800m) or adapt from buffer_deg
    adaptive_domain_width = profile.get("domain_width_m", 800.0)

    # Prefer an explicit caller override; otherwise honor the selected resolution
    # profile so deep profiles use lower-frequency excitation for penetration.
    if excitation_frequency_hz is not None:
        adaptive_freq_hz = float(excitation_frequency_hz)
        excitation_source = "explicit override"
    elif "excitation_frequency_hz" in target:
        adaptive_freq_hz = float(target["excitation_frequency_hz"])
        excitation_source = "target override"
    else:
        adaptive_freq_hz = float(profile.get("excitation_frequency_hz", 35.0))
        excitation_source = f"{resolution} profile"

    pinn_config = {
        "epochs": profile["epochs"],
        "grid_nx": profile["grid_nx"],
        "grid_ny": profile["grid_ny"],
        "grid_nz": profile["grid_nz"],
        "domain_width_m": adaptive_domain_width,
        "max_depth_m": adaptive_max_depth,
        # Loss weights (all configurable now - no more hardcoded sparsity)
        "physics_weight": profile.get("physics_weight", 1.0),
        "data_weight": profile.get("data_weight", 20.0),
        "sparsity_weight": profile.get("sparsity_weight", 0.01),
        "regularization_weight": profile.get("regularization_weight", 0.1),
        "deep_prior_weight": profile.get("deep_prior_weight", 0.1),
        "excitation_frequency_hz": adaptive_freq_hz,
        "batch_size_collocation": profile.get("batch_size_collocation", 4096),
        "batch_size_boundary": profile.get("batch_size_boundary", 1024),
        "gradient_accumulation_steps": profile.get("gradient_accumulation_steps", 1),
        "hidden_layers": profile.get("hidden_layers", 8),
        "hidden_neurons": profile.get("hidden_neurons", 512),
        "use_float64_physics": profile.get("use_float64_physics", True),
    }
    resolved_surface_prior_weight = (
        DEFAULT_EMBEDDING_SURFACE_PRIOR_WEIGHT if use_embeddings else 0.0
    ) if surface_prior_weight is None else float(surface_prior_weight)
    pinn_config["surface_prior_weight"] = resolved_surface_prior_weight

    logger.info(f"  PINN config: domain={adaptive_domain_width:.0f}m, depth={adaptive_max_depth:.0f}m, "
                f"freq={adaptive_freq_hz:.1f}Hz ({excitation_source}), "
                f"grid={profile['grid_nx']}x{profile['grid_ny']}x{profile['grid_nz']}, "
                f"physics_w={pinn_config['physics_weight']}, data_w={pinn_config['data_weight']}, "
                f"sparsity_w={pinn_config['sparsity_weight']}, deep_w={pinn_config['deep_prior_weight']}")

    # Cap 4+6: Load surface embedding anomaly map for this target if available
    surface_anomaly_map: Optional[np.ndarray] = None
    if use_embeddings:
        candidates = _resolve_embedding_raster_candidates(name, embedding_dir, embedding_rasters)
        for cand in candidates:
            if cand.exists():
                try:
                    import rasterio
                    with rasterio.open(cand) as _src:
                        surface_anomaly_map = _src.read(1).astype(np.float32)
                    logger.info(f"Cap 4+6: Loaded surface anomaly map from {cand.name} "
                                f"(shape={surface_anomaly_map.shape})")
                    result["embedding_prior_used"] = resolved_surface_prior_weight > 0.0
                    result["outputs"]["surface_anomaly_map"] = str(cand)
                    break
                except Exception as _e:
                    logger.warning(f"Failed to load embedding anomaly map {cand}: {_e}")
        if surface_anomaly_map is None:
            explicit_rasters = [Path(p) for p in (embedding_rasters or [])]
            ed = Path(embedding_dir) if embedding_dir else None
            search_hint = f"explicit paths: {explicit_rasters}" if explicit_rasters else f"in {ed}"
            logger.info(
                f"Cap 4+6: --use-embeddings set but no anomaly raster found for '{name}' "
                f"{search_hint}. Run satellite_embeddings.py spatial-anomaly first."
            )

    pinn_results = pinn_vibro_inversion.train_vibro_pinn(
        vibration_map=vib_map,
        output_dir=str(pinn_out_dir),
        config=pinn_config,
        frequency_map=freq_map,
        surface_anomaly_map=surface_anomaly_map,
    )

    wave_speed_path = pinn_results.get("wave_speed_volume")
    void_prob_path = pinn_results.get("void_probability_volume")

    if not wave_speed_path:
        msg = "PINN inversion failed."
        logger.error(msg)
        result["error"] = msg
        result["elapsed_seconds"] = time.time() - start_time
        return result

    result["outputs"]["wave_speed_volume"] = wave_speed_path
    result["outputs"]["void_probability_volume"] = void_prob_path

    # --------------------------------------------------------
    # STEP 4: 3D Visualization & Anomaly Extraction
    # --------------------------------------------------------
    logger.info("\n--- STEP 4: 3D Visualization & Void Extraction ---")
    viz_out_dir = target_out_dir / "visualization"

    viz_results = visualize_3d_subsurface.run_visualization_pipeline(
        wave_speed_path=wave_speed_path,
        output_dir=str(viz_out_dir),
        void_probability_path=void_prob_path,
        config={
            "void_threshold": 0.35,
            "min_anomaly_voxels": 3,
            "domain_width_m": adaptive_domain_width,
            "max_depth_m": adaptive_max_depth,
            "depth_slices": select_depth_slices(adaptive_max_depth),
        },
        interactive=False,
        embedding_anomaly_map=surface_anomaly_map,
    )

    result["outputs"]["visualization_dir"] = str(viz_out_dir)
    if "anomaly_report" in viz_results:
        result["outputs"]["anomaly_report"] = viz_results["anomaly_report"]
    if "anomalies_csv" in viz_results:
        result["outputs"]["anomaly_catalog"] = viz_results["anomalies_csv"]
    if "audit_manifest" in viz_results:
        result["outputs"]["audit_manifest"] = viz_results["audit_manifest"]
    if "anomaly_list" in viz_results:
        result["top_deep_targets"] = viz_results["anomaly_list"][:5]

    # --------------------------------------------------------
    # STEP 5: Result Quantification
    # --------------------------------------------------------
    anomaly_threshold = 0.35
    anomaly_count = 0
    if "anomaly_count" in viz_results:
        anomaly_count = int(viz_results["anomaly_count"])
    elif "anomaly_list" in viz_results:
        anomaly_count = len(viz_results["anomaly_list"])
    elif void_prob_path and Path(void_prob_path).exists():
        void_vol = np.load(void_prob_path)
        anomaly_count = int(np.sum(void_vol > anomaly_threshold))
        # Count connected components for a better estimate
        try:
            from scipy import ndimage
            labeled, n_features = ndimage.label(void_vol > anomaly_threshold)
            anomaly_count = n_features
        except ImportError:
            pass

    anomaly_count = _set_anomaly_count_fields(result, anomaly_count)

    if result["data_source"] == "synthetic":
        diag_path = write_synthetic_positive_control_diagnostics(
            slc_path=slc_filepath,
            vib_amp_path=vib_amp_path,
            wave_speed_path=wave_speed_path,
            void_prob_path=void_prob_path,
            output_dir=target_out_dir,
            anomaly_count=anomaly_count,
            void_threshold=0.35,
            min_anomaly_voxels=3,
            background_wave_speed=pinn_config.get("background_wave_speed", 3500.0),
            void_speed_threshold_ratio=pinn_config.get("void_speed_threshold_ratio", 0.7),
        )
        if diag_path:
            result["outputs"]["synthetic_positive_control_diagnostics"] = diag_path

    result["status"] = "success"
    result["elapsed_seconds"] = time.time() - start_time

    logger.info(f"\nPipeline completed for {target['name']}!")
    logger.info(f"  Anomalies detected: {anomaly_count}")
    logger.info(f"  Elapsed time: {result['elapsed_seconds']:.1f}s")
    logger.info(f"  Results: {target_out_dir}")

    return result


# ============================================================
# PHASE SUMMARY REPORT
# ============================================================
def generate_phase_report(
    phase_num: int,
    results: List[Dict],
    output_dir: Path,
):
    """Generate a consolidated report for an exploration phase."""
    report_path = output_dir / f"phase_{phase_num}_report.txt"
    json_path = output_dir / f"phase_{phase_num}_results.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    total_anomalies = sum(normalize_anomaly_count(r) for r in results)
    total_time = sum(float(r.get("elapsed_seconds", 0) or 0) for r in results)

    lines = [
        "=" * 70,
        f"BIONDI EXPLORATION PHASE {phase_num} - CONSOLIDATED REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Targets processed:   {success}/{total}",
        f"Total anomalies:     {total_anomalies}",
        f"Total compute time:  {total_time:.1f}s ({total_time/60:.1f} min)",
        "",
        "-" * 70,
        "PER-TARGET RESULTS",
        "-" * 70,
    ]

    for r in results:
        status = r.get("status", "unknown")
        status_marker = "OK" if status == "success" else "FAIL"
        lines.append(f"\n  {status_marker} {r.get('target_name', r.get('target', '?'))}")
        lines.append(f"    Status:      {status}")
        lines.append(f"    Data source: {r.get('data_source', '?')}")
        lines.append(f"    Anomalies:   {normalize_anomaly_count(r)}")
        lines.append(f"    Time:        {float(r.get('elapsed_seconds', 0) or 0):.1f}s")
        top_targets = r.get("top_deep_targets") or []
        if top_targets:
            best = top_targets[0]
            best_depth = float(best.get("depth_m", 0) or 0)
            best_score = float(best.get("deep_target_score", best.get("fused_confidence_score", 0)) or 0)
            lines.append(
                "    Top deep target: "
                f"rank {best.get('deep_target_rank', '?')}, "
                f"depth {best_depth:.0f}m, "
                f"score {best_score:.3f}, "
                f"shape {best.get('shape_classification', '?')}"
            )
        if r.get("error"):
            lines.append(f"    Error:       {r['error']}")
        coords = r.get("coordinates", {})
        lines.append(f"    Location:    ({coords.get('lat', '?')}, {coords.get('lon', '?')})")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(json_path, "w", encoding="utf-8") as f:
        dump_strict_json(results, f, indent=2)

    logger.info(f"\nPhase {phase_num} report saved to:\n  {report_path}\n  {json_path}")
    print(report_text)

    return report_path


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    load_cli_env()
    parser = argparse.ArgumentParser(
        description="Biondi USA Exploration Orchestrator - SAR Doppler Tomography"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=1,
        help="Which phase to execute (1=reference/exploratory targets, 2=California, 3=USA)."
    )
    parser.add_argument(
        "--resolution", type=str, choices=["quick", "standard", "high", "deep"], default="standard",
        help="Resolution profile controlling grid size and training epochs."
    )
    parser.add_argument(
        "--synthetic-fallback", action="store_true",
        help="Use synthetic Biondi-simulated data if real SLC download fails."
    )
    parser.add_argument(
        "--auto-advance", action="store_true",
        help="Auto-advance to next phase after completing current one."
    )
    parser.add_argument(
        "--clean-rerun", action="store_true",
        help="Delete existing results and re-run from scratch."
    )
    parser.add_argument(
        "--use-embeddings", action="store_true",
        help="Cap 4+6: Load AlphaEarth surface anomaly maps from --embedding-dir and use them "
             "as PINN spatial priors and visualizer fusion weights."
    )
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=None,
        help="Directory containing pre-computed embedding anomaly rasters "
             "(output of satellite_embeddings.py or embedding_target_discovery.py). "
             "Expected filenames: <target_name>_{spatial,cluster,fused}_anomaly.tif"
    )
    parser.add_argument(
        "--embedding-raster", type=Path, action="append", default=[],
        help="Explicit anomaly raster path to use as a surface prior. May be repeated; "
             "checked before --embedding-dir filename fallbacks."
    )
    parser.add_argument(
        "--surface-prior-weight", type=float, default=None,
        help=f"PINN surface-prior loss weight when --use-embeddings loads a raster "
             f"(default: {DEFAULT_EMBEDDING_SURFACE_PRIOR_WEIGHT} with --use-embeddings, otherwise 0)."
    )
    parser.add_argument(
        "--excitation-frequency-hz", type=float, default=None,
        help="Explicit PINN excitation frequency override. By default the selected "
             "resolution profile's excitation_frequency_hz is used."
    )
    parser.add_argument(
        "--target-index", type=int, default=None,
        help="Run only one zero-based target index from the selected phase for narrow audits."
    )
    parser.add_argument(
        "--target-name", type=str, default=None,
        help="Run only targets whose name contains this case-insensitive text."
    )
    args = parser.parse_args(argv)

    # Load credentials
    credentials = {
        "EARTHDATA_USERNAME": os.environ.get("EARTHDATA_USERNAME", ""),
        "EARTHDATA_PASSWORD": os.environ.get("EARTHDATA_PASSWORD", ""),
        "EARTHDATA_TOKEN": os.environ.get("EARTHDATA_TOKEN", ""),
        "EARTHDATA_BEARER_TOKEN": os.environ.get("EARTHDATA_BEARER_TOKEN", ""),
    }
    auth_info = slc_data_fetcher.resolve_earthdata_auth(
        earthdata_username=credentials["EARTHDATA_USERNAME"],
        earthdata_password=credentials["EARTHDATA_PASSWORD"],
        earthdata_token=(credentials["EARTHDATA_TOKEN"] or credentials["EARTHDATA_BEARER_TOKEN"]),
    )

    if auth_info["mode"] not in {"token", "credentials"}:
        logger.warning("\n!!! NO EARTHDATA AUTHENTICATION FOUND !!!")
        logger.warning("To process real satellite data, provide NASA Earthdata authentication.")
        logger.warning(
            "Set ENV: EARTHDATA_TOKEN or EARTHDATA_BEARER_TOKEN; "
            "username/password remains supported via EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )
        logger.warning("Earthdata auth status: %s", slc_data_fetcher.describe_earthdata_auth(auth_info))
        if not args.synthetic_fallback:
            logger.warning("Run with --synthetic-fallback to simulate the pipeline anyway.\n")
    else:
        logger.info("Earthdata auth configured: %s", slc_data_fetcher.describe_earthdata_auth(auth_info))

    # Select phase targets
    phase_map = {1: PHASE_1_TARGETS, 2: PHASE_2_TARGETS, 3: PHASE_3_TARGETS}
    current_phase = args.phase

    while current_phase <= 3:
        targets = phase_map[current_phase]
        if args.target_index is not None:
            if args.target_index < 0 or args.target_index >= len(targets):
                raise ValueError(
                    f"--target-index {args.target_index} is outside phase {current_phase} "
                    f"target range 0..{len(targets) - 1}"
                )
            targets = [targets[args.target_index]]
        if args.target_name:
            target_name_filter = args.target_name.lower()
            targets = [t for t in targets if target_name_filter in t["name"].lower()]
            if not targets:
                raise ValueError(
                    f"--target-name {args.target_name!r} matched no phase {current_phase} targets"
                )
        # Phase 1 uses the configured reference/exploratory targets in PHASE_1_TARGETS
        # (Vacaville, Khufu, Khafre). It is not a formal validation set.
        logger.info(f"\n{'#'*70}")
        logger.info(f"# STARTING EXPLORATION PHASE {current_phase}")
        logger.info(f"# Targets: {len(targets)}")
        logger.info(f"# Resolution: {args.resolution}")
        logger.info(f"{'#'*70}\n")

        phase_results = []

        for i, t in enumerate(targets):
            logger.info(f"\n[Target {i+1}/{len(targets)}]")
            try:
                # Optionally clean previous results
                if args.clean_rerun:
                    clean_name = t["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                    old_dir = EXPLORE_DIR / clean_name
                    if old_dir.exists():
                        import shutil
                        logger.info(f"Cleaning previous results: {old_dir}")
                        shutil.rmtree(old_dir, ignore_errors=True)

                res = execute_biondi_pipeline_for_target(
                    target=t,
                    credentials=credentials,
                    use_synthetic_fallback=args.synthetic_fallback,
                    resolution=args.resolution,
                    use_embeddings=args.use_embeddings,
                    embedding_dir=args.embedding_dir,
                    embedding_rasters=args.embedding_raster,
                    surface_prior_weight=args.surface_prior_weight,
                    excitation_frequency_hz=args.excitation_frequency_hz,
                )
                phase_results.append(res)
            except Exception as e:
                logger.error(f"Critical failure processing {t['name']}: {e}")
                logger.error(traceback.format_exc())
                phase_results.append({
                    "target_name": t["name"],
                    "coordinates": {"lat": t["lat"], "lon": t["lon"]},
                    "resolution_profile": args.resolution,
                    "status": "crashed",
                    "anomaly_count": 0,
                    "anomalies_detected": 0,
                    "top_deep_targets": [],
                    "data_source": "none",
                    "outputs": {},
                    "elapsed_seconds": 0,
                    "error": str(e),
                })

        # Generate consolidated report
        generate_phase_report(current_phase, phase_results, EXPLORE_DIR)

        success_count = sum(1 for r in phase_results if r["status"] == "success")
        logger.info(
            f"\nExploration Phase {current_phase} Finished. "
            f"Processed {success_count}/{len(targets)} targets successfully."
        )

        # Auto-advance logic
        if args.auto_advance and current_phase < 3:
            current_phase += 1
            logger.info(f"\n>>> Auto-advancing to Phase {current_phase}...")
        else:
            break

    logger.info("\n>>> Orchestrator finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
