#!/usr/bin/env python3
"""
Biondi Doppler Tomography - Exploration Orchestrator
======================================================
This script orchestrates the large-scale mapping using the SAR Doppler
Tomography pipeline, following a phased approach:

    Phase 1: Known Voids & Structures (Calibration & Validation)
    Phase 2: Regional Scan (California)
    Phase 3: Continental Scan (USA)

Execution requires NASA Earthdata credentials to download raw Sentinel-1
SLC data via the Alaska Satellite Facility (ASF) API.

Provide credentials either via environment variables or a .env file:
    EARTHDATA_USERNAME=your_username
    EARTHDATA_PASSWORD=your_password
"""

import os
import sys
import json
import logging
import argparse
import traceback
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

print(">>> Orchestrator starting...", flush=True)

import numpy as np

# Load .env variables before importing project_paths
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print(f">>> Loading credentials from {env_path}", flush=True)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
                log_v = "********" if "PASSWORD" in k.upper() else v
                print(f"    - Set {k}={log_v}", flush=True)

# Set up paths and environment
print(">>> Importing project_paths...", flush=True)
from project_paths import DATA_DIR, OUTPUTS_DIR

print(f">>> DATA_DIR resolved to: {DATA_DIR}", flush=True)

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

print(">>> Logger configured.", flush=True)

EXPLORE_DIR = DATA_DIR / "biondi_exploration"
EXPLORE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PIPELINE RESOLUTION PROFILES
# ============================================================
# Controls tradeoff between speed and detection quality.
# 
# TRIVIAL COLLAPSE FIX (2025-03-10):
#   - deep_prior_weight reduced from 0.1/0.05/0.02 → 0.01/0.005/0.001
#     (Old values penalized deep anomalies so heavily the PINN predicted solid rock everywhere)
#   - data_weight increased from 10.0 → 50.0
#     (Forces the PINN to trust satellite surface observations over smoothing)
#   - domain_width_m and max_depth_m added per profile
#     (Old: 5000m domain / 32 grid = 156m voxels → anomalies averaged out)
#   - excitation_frequency_hz added per profile
#     (Low freq penetrates deeper; high freq resolves shallow detail)
#
RESOLUTION_PROFILES = {
    "quick": {
        "epochs": 500,                        # Quick sanity check
        "grid_nx": 64,                        # Was 32 → 64 for basic visibility
        "grid_ny": 64,
        "grid_nz": 32,                        # Was 16
        "domain_width_m": 800.0,              # Focused domain (was implicitly 5000)
        "max_depth_m": 500.0,                 # Shallow scan (was implicitly 5000)
        "deep_prior_weight": 0.01,            # Was 0.1 — caused trivial collapse
        "data_weight": 50.0,                  # Was 10.0
        "excitation_frequency_hz": 2.0,       # Medium-depth penetration
        "synthetic_grid_size": 256,
        "num_sub_apertures": 3,
        "batch_size_collocation": 4096,
        "batch_size_boundary": 1024,
        "gradient_accumulation_steps": 1,
        "hidden_layers": 6,                   # Was 4
        "hidden_neurons": 256,                # Was 128
    },
    "standard": {
        "epochs": 3000,                       # Was 2000 — PINNs need more iterations
        "grid_nx": 128,                       # Was 64 → 128 for ~6m resolution
        "grid_ny": 128,
        "grid_nz": 64,                        # Was 32
        "domain_width_m": 800.0,              # Focused domain
        "max_depth_m": 1000.0,                # Was 5000m (too deep for 500m targets)
        "deep_prior_weight": 0.005,           # Was 0.05 — caused trivial collapse
        "data_weight": 50.0,                  # Was 10.0
        "excitation_frequency_hz": 1.0,       # Deep microseismic peak
        "synthetic_grid_size": 512,
        "num_sub_apertures": 5,
        "batch_size_collocation": 4096,       # Fits easily in 24GB 4090 VRAM
        "batch_size_boundary": 1024,          # 1024 surface points
        "gradient_accumulation_steps": 1,     # No accumulation needed on 4090
        "hidden_layers": 8,                   # Full model depth
        "hidden_neurons": 512,                # Full model width
    },
    "high": {
        "epochs": 5000,                       # Full convergence
        "grid_nx": 128,                       # Fine spatial resolution
        "grid_ny": 128,
        "grid_nz": 64,                        # Fine depth resolution
        "domain_width_m": 800.0,              # Focused domain
        "max_depth_m": 1000.0,                # Max depth for detailed scans
        "deep_prior_weight": 0.001,           # Was 0.02 — minimal deep regularization
        "data_weight": 50.0,                  # Was 10.0
        "excitation_frequency_hz": 0.5,       # Deepest penetration
        "synthetic_grid_size": 512,
        "num_sub_apertures": 7,               # More frequency sampling
        "batch_size_collocation": 8192,       # Larger batches → better gradient estimates
        "batch_size_boundary": 2048,
        "gradient_accumulation_steps": 1,
        "hidden_layers": 10,                  # Deeper network
        "hidden_neurons": 768,                # Wider network → more capacity
    },
}


# ============================================================
# TARGET REGIONS
# ============================================================
PHASE_1_TARGETS = [
    {
        "name": "Great Pyramid of Giza (Khufu)",
        "lat": 29.9792,
        "lon": 31.1342,
        "buffer_deg": 0.15,
        "description": "Biondi's main target — claimed spiral columns & 80m chambers. JRE #2443 discussion.",
        "expected_depth_m": 500,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Khafre Pyramid (Giza)",
        "lat": 29.9761,
        "lon": 31.1313,
        "buffer_deg": 0.1,
        "description": "Biondi claims strongest anomalies here (vertical spirals + deep corridors).",
        "expected_depth_m": 1000,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Carlsbad Caverns (NM)",
        "lat": 32.1742,
        "lon": -104.4459,
        "buffer_deg": 0.05,
        "description": "Massive natural limestone cave system. Ideal for validating void detection deep underground.",
        "expected_depth_m": 300,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Bingham Canyon Mine (UT)",
        "lat": 40.5225,
        "lon": -112.1522,
        "buffer_deg": 0.05,
        "description": "Deepest open-pit mine in the world with extensive underground workings.",
        "expected_depth_m": 1200,
        "expected_void_type": "mine_workings",
    },
    {
        "name": "San Andreas Fault Observatory at Depth - SAFOD (CA)",
        "lat": 35.9750,
        "lon": -120.5520,
        "buffer_deg": 0.05,
        "description": "Instrumented borehole intersecting the fault. Good for testing varying density fields.",
        "expected_depth_m": 3000,
        "expected_void_type": "fault_zone",
    },
    {
        "name": "Menkaure Pyramid (Giza)",
        "lat": 29.9725,
        "lon": 31.1283,
        "buffer_deg": 0.08,
        "description": "2025 air-filled voids + Biondi subsurface extensions.",
        "expected_depth_m": 100,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Great Sphinx (Giza)",
        "lat": 29.9710,
        "lon": 31.1378,
        "buffer_deg": 0.1,
        "description": "Biondi mentions possible chambers beneath.",
        "expected_depth_m": 300,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Synthetic Biondi Chambers (Giza-style)",
        "lat": 29.9792,
        "lon": 31.1342,
        "buffer_deg": 0.15,
        "description": "TEST: 80m chambers + spiral columns exactly as described on JRE",
        "expected_depth_m": 800,
        "expected_void_type": "artificial_cavity",
    },
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
        "description": "World's longest known cave system — 680+ km mapped passages.",
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
) -> str:
    """
    Generate a physically plausible synthetic SLC burst for a specific
    geological target using the sar_vibrometry module's generator.
    
    This uses the proper Rayleigh-distributed amplitude + phase modulation
    approach that embeds known anomalies, NOT random noise.
    """
    name = target["name"]
    expected_type = target.get("expected_void_type", "natural_cave")
    expected_depth = target.get("expected_depth_m", 500)

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
    )
    return result_path


# ============================================================
# PIPELINE EXECUTION
# ============================================================
def execute_biondi_pipeline_for_target(
    target: Dict,
    credentials: Dict[str, str],
    use_synthetic_fallback: bool = False,
    resolution: str = "standard",
) -> Dict:
    """
    Executes the end-to-end Biondi SAR Doppler pipeline for a specific
    geographical target.

    Returns a result dict with status, paths, anomaly count, and timing.
    """
    profile = RESOLUTION_PROFILES.get(resolution, RESOLUTION_PROFILES["standard"])
    name = target["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    start_time = time.time()

    result = {
        "target_name": target["name"],
        "coordinates": {"lat": target["lat"], "lon": target["lon"]},
        "resolution_profile": resolution,
        "status": "failed",
        "anomalies_detected": 0,
        "data_source": "none",
        "outputs": {},
        "elapsed_seconds": 0,
        "error": None,
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
    logger.info("\n--- STEP 1: SLC Data Acquisition (Umbra → Sentinel-1 → Synthetic) ---")
    earthdata_user = credentials.get("EARTHDATA_USERNAME")
    earthdata_pass = credentials.get("EARTHDATA_PASSWORD")

    bbox = slc_data_fetcher._build_search_bbox(
        lat=target['lat'],
        lon=target['lon'],
        buffer_deg=target['buffer_deg']
    )

    slc_filepath = None
    if use_synthetic_fallback:
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
                logger.info("No Umbra data available for this area — trying Sentinel-1...")
        except Exception as e:
            logger.warning(f"Umbra data acquisition failed (non-fatal): {e}")

        # === FALLBACK TO SENTINEL-1 (lower resolution but broader coverage) ===
        if not slc_filepath:
            try:
                slc_products = slc_data_fetcher.search_sentinel1_slc(
                    bbox=bbox,
                    max_results=1
                )

                if slc_products and earthdata_user and earthdata_pass:
                    logger.info(f"Found {len(slc_products)} Sentinel-1 products. Downloading first...")
                    logger.warning("NOTE: Sentinel-1 C-band (5x20m, TOPSAR) has limited Doppler sensitivity. "
                                   "Results will be lower resolution than X-band Spotlight data.")
                    download_dir = target_out_dir / "raw_slc"
                    downloaded_path = slc_data_fetcher.download_slc_product(
                        product=slc_products[0],
                        output_dir=download_dir,
                        earthdata_username=earthdata_user,
                        earthdata_password=earthdata_pass,
                    )
                    if downloaded_path:
                        logger.info("Extracting first burst from downloaded SAFE file...")
                        extracted = slc_data_fetcher.extract_slc_burst(
                            Path(downloaded_path),
                            output_dir=target_out_dir / "bursts"
                        )
                        if extracted:
                            slc_filepath = str(extracted[0])
                            result["data_source"] = "sentinel1_cband"
                        # Delete ZIP to save disk (5.5GB+ per file)
                        try:
                            zip_path = Path(downloaded_path)
                            if zip_path.exists():
                                zip_path.unlink()
                                logger.info(f"Deleted ZIP to save disk: {zip_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete ZIP: {e}")
                elif not earthdata_user or not earthdata_pass:
                    logger.warning("EARTHDATA credentials not found — cannot download Sentinel-1 data.")
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

    # Select excitation frequency based on target depth:
    #   Shallow (0-100m) → 5-15 Hz, Medium (100-500m) → 1-5 Hz, Deep (500m+) → 0.5-1 Hz
    adaptive_freq_hz = profile.get("excitation_frequency_hz", 1.0)
    if target_depth <= 100:
        adaptive_freq_hz = max(adaptive_freq_hz, 5.0)   # Shallow: higher freq
    elif target_depth <= 500:
        adaptive_freq_hz = max(adaptive_freq_hz, 1.0)   # Medium depth
    # else: use profile's default (0.5-1.0 Hz for deep)

    pinn_config = {
        "epochs": profile["epochs"],
        "grid_nx": profile["grid_nx"],
        "grid_ny": profile["grid_ny"],
        "grid_nz": profile["grid_nz"],
        "domain_width_m": adaptive_domain_width,
        "max_depth_m": adaptive_max_depth,
        "deep_prior_weight": profile["deep_prior_weight"],
        "data_weight": profile.get("data_weight", 50.0),
        "excitation_frequency_hz": adaptive_freq_hz,
        "batch_size_collocation": profile.get("batch_size_collocation", 4096),
        "batch_size_boundary": profile.get("batch_size_boundary", 1024),
        "gradient_accumulation_steps": profile.get("gradient_accumulation_steps", 1),
        "hidden_layers": profile.get("hidden_layers", 8),
        "hidden_neurons": profile.get("hidden_neurons", 512),
    }

    logger.info(f"  PINN config: domain={adaptive_domain_width:.0f}m, depth={adaptive_max_depth:.0f}m, "
                f"freq={adaptive_freq_hz:.1f}Hz, grid={profile['grid_nx']}x{profile['grid_ny']}x{profile['grid_nz']}, "
                f"deep_prior={profile['deep_prior_weight']}, data_weight={profile.get('data_weight', 50.0)}")

    pinn_results = pinn_vibro_inversion.train_vibro_pinn(
        vibration_map=vib_map,
        output_dir=str(pinn_out_dir),
        config=pinn_config,
        frequency_map=freq_map
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
            "void_threshold": 0.5,
            "min_anomaly_voxels": 3,
        },
        interactive=False
    )

    result["outputs"]["visualization_dir"] = str(viz_out_dir)
    if "anomaly_report" in viz_results:
        result["outputs"]["anomaly_report"] = viz_results["anomaly_report"]
    if "detected_anomalies" in viz_results:
        result["outputs"]["anomaly_catalog"] = viz_results["detected_anomalies"]

    # --------------------------------------------------------
    # STEP 5: Result Quantification
    # --------------------------------------------------------
    anomaly_count = 0
    if "anomaly_list" in viz_results:
        anomaly_count = len(viz_results["anomaly_list"])
    elif void_prob_path and Path(void_prob_path).exists():
        void_vol = np.load(void_prob_path)
        anomaly_count = int(np.sum(void_vol > 0.5))
        # Count connected components for a better estimate
        try:
            from scipy import ndimage
            labeled, n_features = ndimage.label(void_vol > 0.5)
            anomaly_count = n_features
        except ImportError:
            pass

    result["anomalies_detected"] = anomaly_count
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

    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    total_anomalies = sum(r["anomalies_detected"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)

    lines = [
        "=" * 70,
        f"BIONDI EXPLORATION PHASE {phase_num} — CONSOLIDATED REPORT",
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
        status_marker = "OK" if r["status"] == "success" else "FAIL"
        lines.append(f"\n  {status_marker} {r['target_name']}")
        lines.append(f"    Status:      {r['status']}")
        lines.append(f"    Data source: {r['data_source']}")
        lines.append(f"    Anomalies:   {r['anomalies_detected']}")
        lines.append(f"    Time:        {r['elapsed_seconds']:.1f}s")
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
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nPhase {phase_num} report saved to:\n  {report_path}\n  {json_path}")
    print(report_text)

    return report_path


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Biondi USA Exploration Orchestrator — SAR Doppler Tomography"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=1,
        help="Which phase to execute (1=calibration, 2=California, 3=USA)."
    )
    parser.add_argument(
        "--resolution", type=str, choices=["quick", "standard", "high"], default="standard",
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
    args = parser.parse_args()

    # Load credentials
    credentials = {
        "EARTHDATA_USERNAME": os.environ.get("EARTHDATA_USERNAME", ""),
        "EARTHDATA_PASSWORD": os.environ.get("EARTHDATA_PASSWORD", ""),
    }

    if not credentials["EARTHDATA_USERNAME"]:
        logger.warning("\n!!! NO EARTHDATA CREDENTIALS FOUND !!!")
        logger.warning("To process real satellite data, provide NASA Earthdata credentials.")
        logger.warning("Set ENV: EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
        if not args.synthetic_fallback:
            logger.warning("Run with --synthetic-fallback to simulate the pipeline anyway.\n")

    # Select phase targets
    phase_map = {1: PHASE_1_TARGETS, 2: PHASE_2_TARGETS, 3: PHASE_3_TARGETS}
    current_phase = args.phase

    while current_phase <= 3:
        targets = phase_map[current_phase]
        if current_phase == 1:
            targets = [t for t in targets if "Giza" in t["name"]]
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
                    "anomalies_detected": 0,
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
