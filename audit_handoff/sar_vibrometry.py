#!/usr/bin/env python3
"""
SAR Vibrometry: Doppler Sub-Aperture Decomposition
====================================================

Implements the core of the Biondi SAR Doppler Tomography methodology:
extracting sub-millimeter surface vibration maps from single-pass SAR
acquisitions.

The Science
-----------
Standard InSAR compares phase between TWO different dates to measure
slow subsidence over weeks/months. The Biondi method instead tracks
phase/frequency shifts WITHIN A SINGLE satellite pass (fractions of a
second) to measure instantaneous vibrations of the Earth's surface.

These vibrations are caused by ambient seismic background noise
(microseisms from ocean waves, wind, internal planetary heat) interacting
with deep underground structures. Voids, tunnels, dense ore bodies, and
other anomalies alter the acoustic resonance patterns, creating detectable
signatures on the surface directly above them.

Algorithm Overview
------------------
1. **Sub-Aperture Decomposition**: Split the SLC image's azimuth
   (Doppler) spectrum into overlapping frequency bands. Each band
   corresponds to a different "look angle" during the satellite pass.

2. **Phase Tracking**: Track how each pixel's phase shifts across
   sub-apertures. The rate of phase change encodes the Doppler
   centroid shift, revealing instantaneous vibration velocity.

3. **Vibration Map Generation**: Combine the sub-aperture phase
   differences into a 2D map of surface micro-vibration amplitude
   and frequency, ready for PINN inversion.

Usage:
    python sar_vibrometry.py --input data/slc/bursts/burst_00.npy --num-sub 5
    python sar_vibrometry.py --input data/slc/bursts/burst_00.npy --output vibration_map.tif
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import windows

from project_paths import DATA_DIR, PROCESSED_DIR, ensure_directories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_gaussian_filter(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Fast GPU-accelerated separable Gaussian filtering for Complex/Real tensors (O(N) instead of O(N^2))"""
    if sigma <= 0:
        return tensor

    device = tensor.device
    filter_size = int(4 * float(sigma) + 0.5) * 2 + 1
    x = torch.arange(-filter_size // 2 + 1., filter_size // 2 + 1., device=device)

    kernel_1d = torch.exp(-(x**2) / (2 * float(sigma)**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    padding = filter_size // 2

    if tensor.is_complex():
        kx = kernel_1d.view(1, 1, 1, -1).to(tensor.real.dtype)
        ky = kernel_1d.view(1, 1, -1, 1).to(tensor.real.dtype)

        def _filter(ch):
            ch = F.conv2d(ch.unsqueeze(0).unsqueeze(0), kx, padding=(0, padding))
            ch = F.conv2d(ch, ky, padding=(padding, 0))
            return ch.squeeze(0).squeeze(0)

        return torch.complex(_filter(tensor.real), _filter(tensor.imag))
    else:
        kx = kernel_1d.view(1, 1, 1, -1).to(tensor.dtype)
        ky = kernel_1d.view(1, 1, -1, 1).to(tensor.dtype)

        res = F.conv2d(tensor.unsqueeze(0).unsqueeze(0), kx, padding=(0, padding))
        res = F.conv2d(res, ky, padding=(padding, 0))
        return res.squeeze(0).squeeze(0)


# ============================================================
# Configuration
# ============================================================
VIBROMETRY_DIR = DATA_DIR / "vibrometry"
VIBROMETRY_OUTPUTS = VIBROMETRY_DIR / "outputs"

DEFAULT_VIBROMETRY_CONFIG = {
    "num_sub_apertures": 5,           # Number of azimuth sub-bands
    "overlap_fraction": 0.3,          # Fractional overlap between adjacent sub-apertures
    "hamming_taper": True,            # Apply Hamming window to reduce spectral leakage
    "vibration_frequency_hz": 1.0,    # Expected dominant vibration frequency
    "radar_wavelength_m": 0.0555,     # Sentinel-1 C-band wavelength (5.55 cm)
    "prf_hz": 486.486,               # Sentinel-1 IW mode pulse repetition frequency
    "velocity_sat_ms": 7500.0,        # Satellite ground velocity (m/s)
    "coherence_threshold": 0.3,       # Minimum coherence for valid vibration estimate
    "spatial_smoothing_sigma": 1.5,   # Gaussian smoothing of output (pixels)
}


# ============================================================
# Core: Sub-Aperture Decomposition
# ============================================================
def extract_doppler_sub_apertures(
    slc_complex_data: np.ndarray,
    num_sub_apertures: int = 5,
    overlap_fraction: float = 0.3,
    apply_taper: bool = True,
) -> List[torch.Tensor]:
    """
    Split a single SLC SAR image into multiple time-domain sub-apertures
    using GPU-accelerated PyTorch tensors.
    """
    device = get_device()
    slc_tensor = torch.from_numpy(slc_complex_data).to(device)
    azimuth_len, range_len = slc_tensor.shape
    
    logger.info(
        f"Sub-aperture decomposition (GPU): {azimuth_len}x{range_len}, "
        f"{num_sub_apertures} sub-apertures, overlap={overlap_fraction:.1%}"
    )

    # 1D FFT along azimuth (rows)
    doppler_spectrum = torch.fft.fftshift(torch.fft.fft(slc_tensor, dim=0), dim=0)

    # Calculate sub-band parameters
    step_fraction = 1.0 - overlap_fraction
    band_size = int(azimuth_len / (1 + step_fraction * (num_sub_apertures - 1)))
    step_size = int(band_size * step_fraction)

    # Ensure minimum band size
    band_size = max(band_size, 16)
    step_size = max(step_size, 8)

    logger.info(f"  Band size: {band_size} bins, step: {step_size} bins")

    # Build taper window for spectral filtering
    # Kaiser(beta=5) provides superior radar sidelobe suppression vs Hamming
    if apply_taper:
        taper_np = windows.kaiser(band_size, beta=5.0).astype(np.float32)
        taper = torch.from_numpy(taper_np).to(device)
    else:
        taper = torch.ones(band_size, dtype=slc_tensor.dtype, device=device)

    sub_apertures = []

    for i in range(num_sub_apertures):
        center = int(azimuth_len / 2 - band_size / 2 +
                     (i - (num_sub_apertures - 1) / 2) * step_size)

        start = max(0, center)
        end = min(azimuth_len, start + band_size)
        actual_size = end - start

        # Apply windowed filter to isolate this Doppler band
        window = torch.zeros(azimuth_len, dtype=slc_tensor.dtype, device=device)
        if actual_size == band_size:
            window[start:end] = taper
        else:
            partial_taper = taper[:actual_size] if start == 0 else taper[-actual_size:]
            window[start:end] = partial_taper

        filtered_spectrum = doppler_spectrum * window.unsqueeze(1)

        # IFFT back to spatial domain
        sub_img = torch.fft.ifft(torch.fft.ifftshift(filtered_spectrum, dim=0), dim=0)

        # Free intermediate tensors to reduce VRAM
        del window, filtered_spectrum
        sub_apertures.append(sub_img)

        logger.debug(
            f"  Sub-aperture {i}: Doppler bins [{start}:{end}], "
            f"amplitude range [{torch.abs(sub_img).min().item():.4f}, {torch.abs(sub_img).max().item():.4f}]"
        )

    return sub_apertures


# ============================================================
# Phase & Vibration Extraction
# ============================================================
def compute_sub_aperture_interferograms(
    sub_apertures: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GPU-accelerated interferograms between consecutive sub-apertures.
    """
    num_pairs = len(sub_apertures) - 1
    h, w = sub_apertures[0].shape
    device = sub_apertures[0].device

    phase_diffs = torch.zeros((num_pairs, h, w), dtype=torch.float32, device=device)
    coherence_maps = torch.zeros((num_pairs, h, w), dtype=torch.float32, device=device)

    # Neighborhood size for coherence estimation
    coh_window = 5
    half_w = float(coh_window // 2)

    half_w = 2.5  # Gaussian sigma for multi-look smoothing

    for i in range(num_pairs):
        # Complex interferogram = S_i * conj(S_{i+1})
        interferogram = sub_apertures[i] * torch.conj(sub_apertures[i + 1])

        # CRITICAL FIX: Multi-looking — smooth complex interferogram BEFORE angle extraction.
        # Taking angle of raw interferogram yields pure white noise due to radar speckle.
        # Complex smoothing acts as amplitude-weighted phase filtering.
        interferogram_smooth = apply_gaussian_filter(interferogram, half_w)
        phase_diffs[i] = torch.angle(interferogram_smooth).float()

        numerator = torch.abs(interferogram_smooth)
        power_1 = apply_gaussian_filter(torch.abs(sub_apertures[i]) ** 2, half_w)
        power_2 = apply_gaussian_filter(torch.abs(sub_apertures[i + 1]) ** 2, half_w)
        coherence_maps[i] = (numerator / torch.sqrt(power_1 * power_2 + 1e-12)).float()

    return phase_diffs, coherence_maps


def phase_to_vibration_velocity(
    phase_diffs: torch.Tensor,
    coherence_maps: torch.Tensor,
    radar_wavelength_m: float = 0.0555,
    prf_hz: float = 486.486,
    num_sub_apertures: int = 5,
    coherence_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sub-aperture phase differences directly to vibration velocity on GPU.
    """
    conv_factor = (radar_wavelength_m * prf_hz) / (4 * np.pi * num_sub_apertures)
    velocity_pairs = phase_diffs * conv_factor

    valid_mask = coherence_maps >= coherence_threshold

    weights = coherence_maps ** 2 * valid_mask.float()
    weight_sum = torch.sum(weights, dim=0) + 1e-12

    weighted_v2 = torch.sum(weights * velocity_pairs ** 2, dim=0)
    rms_velocity = torch.sqrt(weighted_v2 / weight_sum)
    vibration_velocity = rms_velocity * 1000.0  # m/s -> mm/s

    vibration_quality = torch.sum(weights, dim=0) / (torch.sum(valid_mask.float(), dim=0) + 1e-12)
    vibration_quality = torch.clamp(vibration_quality, 0.0, 1.0)

    return vibration_velocity.float(), vibration_quality.float()


def compute_vibration_frequency_map(
    sub_apertures: List[torch.Tensor],
    prf_hz: float = 486.486,
    num_sub_apertures: int = 5,
) -> torch.Tensor:
    """
    Estimate dominant vibration frequency efficiently on the GPU.
    """
    h, w = sub_apertures[0].shape
    n_sub = len(sub_apertures)
    device = sub_apertures[0].device
    
    phase_stack = torch.stack([torch.angle(sa).float() for sa in sub_apertures])
    freq_map = torch.zeros((h, w), dtype=torch.float32, device=device)

    pad_len = max(64, n_sub * 16)
    temporal_sampling_rate = prf_hz / num_sub_apertures
    freqs = torch.fft.fftfreq(pad_len, d=1.0 / temporal_sampling_rate).to(device)
    freqs_valid = torch.abs(freqs[:pad_len // 2 + 1]).float()

    t = torch.arange(n_sub, dtype=torch.float32, device=device)
    t_centered = t - t.mean()
    t_centered_var = torch.sum(t_centered ** 2)
    t_centered_view = t_centered.view(-1, 1, 1)

    chunk_size = 64  # Smaller chunks to avoid OOM on large bursts
    for start_r in range(0, h, chunk_size):
        end_r = min(start_r + chunk_size, h)
        
        chunk_phases = phase_stack[:, start_r:end_r, :]
        
        diff = torch.diff(chunk_phases, dim=0)
        diff_wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        unwrapped_diff = torch.cat((chunk_phases[0:1], diff_wrapped), dim=0)
        phase_unwrapped = torch.cumsum(unwrapped_diff, dim=0)
        
        slope = torch.sum(t_centered_view * phase_unwrapped, dim=0) / t_centered_var
        detrended = phase_unwrapped - slope.unsqueeze(0) * t_centered_view
        
        spec = torch.abs(torch.fft.fft(detrended, n=pad_len, dim=0))
        spec[0, :, :] = 0  # Kill DC component
        
        peak_idx = torch.argmax(spec[:pad_len // 2 + 1], dim=0)
        freq_map[start_r:end_r, :] = freqs_valid[peak_idx]

    return freq_map


# ============================================================
# Full Pipeline
# ============================================================
def run_vibrometry_pipeline(
    slc_path: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None,
    georef_tif: Optional[str] = None,
) -> Dict[str, str]:
    """
    Execute the full SAR vibrometry pipeline on a single SLC burst.

    Steps:
    1. Load SLC complex data
    2. Sub-aperture decomposition
    3. Interferometric phase tracking
    4. Velocity & frequency map generation
    5. Save outputs

    Parameters
    ----------
    slc_path : str
        Path to SLC complex data (.npy file from extract_slc_burst()).
    output_dir : str, optional
        Directory for output files.
    config : dict, optional
        Override default vibrometry configuration.
    georef_tif : str, optional
        Reference GeoTIFF for georeferencing the output.

    Returns
    -------
    dict
        Paths to output files: 'vibration_amplitude', 'vibration_frequency',
        'vibration_quality', 'phase_diffs'.
    """
    cfg = DEFAULT_VIBROMETRY_CONFIG.copy()
    if config:
        cfg.update(config)

    if output_dir is None:
        output_dir = str(VIBROMETRY_OUTPUTS)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load SLC data
    logger.info(f"Loading SLC data from {slc_path}")
    slc_data = np.load(slc_path)
    if not np.iscomplexobj(slc_data):
        raise ValueError(
            f"SLC data must be complex. Got dtype={slc_data.dtype}. "
            f"Ensure the file contains IQ data, not amplitude."
        )
    logger.info(f"SLC shape: {slc_data.shape}, dtype: {slc_data.dtype}")

    # Crop large bursts to avoid OOM — 4096x4096 is still >>800m domain
    max_dim = 4096
    if slc_data.shape[0] > max_dim or slc_data.shape[1] > max_dim:
        h_orig, w_orig = slc_data.shape
        r0 = max(0, h_orig // 2 - max_dim // 2)
        c0 = max(0, w_orig // 2 - max_dim // 2)
        slc_data = slc_data[r0:r0+max_dim, c0:c0+max_dim]
        logger.info(f"Cropped SLC from {h_orig}x{w_orig} to {slc_data.shape[0]}x{slc_data.shape[1]} (center crop)")

    # 2. Sub-aperture decomposition
    logger.info("Step 2: Doppler sub-aperture decomposition...")
    sub_apertures = extract_doppler_sub_apertures(
        slc_data,
        num_sub_apertures=cfg["num_sub_apertures"],
        overlap_fraction=cfg["overlap_fraction"],
        apply_taper=cfg["hamming_taper"],
    )
    logger.info(f"  Generated {len(sub_apertures)} sub-apertures")
    torch.cuda.empty_cache()

    # 3. Interferometric phase tracking
    logger.info("Step 3: Computing sub-aperture interferograms...")
    phase_diffs, coherence_maps = compute_sub_aperture_interferograms(sub_apertures)
    logger.info(
        f"  Phase diff range: [{phase_diffs.min().item():.4f}, {phase_diffs.max().item():.4f}] rad"
    )
    logger.info(
        f"  Mean coherence: {coherence_maps.mean().item():.4f}"
    )
    torch.cuda.empty_cache()

    # 4. Vibration velocity map
    logger.info("Step 4: Converting phase to vibration velocity...")
    vibration_velocity, vibration_quality = phase_to_vibration_velocity(
        phase_diffs,
        coherence_maps,
        radar_wavelength_m=cfg["radar_wavelength_m"],
        prf_hz=cfg["prf_hz"],
        num_sub_apertures=cfg["num_sub_apertures"],
        coherence_threshold=cfg["coherence_threshold"],
    )

    # 5. Vibration frequency map
    logger.info("Step 5: Estimating vibration frequencies...")
    freq_map = compute_vibration_frequency_map(
        sub_apertures,
        prf_hz=cfg["prf_hz"],
        num_sub_apertures=cfg["num_sub_apertures"],
    )

    # 6. Spatial smoothing
    sigma = float(cfg["spatial_smoothing_sigma"])
    if sigma > 0:
        vibration_velocity = apply_gaussian_filter(vibration_velocity, sigma)
        freq_map = apply_gaussian_filter(freq_map, sigma)

    # Pull tensors back to CPU for I/O and evaluation
    vibration_velocity = vibration_velocity.cpu().numpy()
    freq_map = freq_map.cpu().numpy()
    vibration_quality = vibration_quality.cpu().numpy()
    phase_diffs = phase_diffs.cpu().numpy()

    # 7. Save outputs
    base_name = Path(slc_path).stem
    outputs = {}

    # Save as numpy arrays (always works)
    amp_path = Path(output_dir) / f"{base_name}_vibration_amplitude.npy"
    freq_path = Path(output_dir) / f"{base_name}_vibration_frequency.npy"
    qual_path = Path(output_dir) / f"{base_name}_vibration_quality.npy"
    phase_path = Path(output_dir) / f"{base_name}_phase_diffs.npy"

    np.save(amp_path, vibration_velocity)
    np.save(freq_path, freq_map)
    np.save(qual_path, vibration_quality)
    np.save(phase_path, phase_diffs)

    outputs["vibration_amplitude_npy"] = str(amp_path)
    outputs["vibration_frequency_npy"] = str(freq_path)
    outputs["vibration_quality_npy"] = str(qual_path)
    outputs["phase_diffs_npy"] = str(phase_path)

    # Save as GeoTIFF if geo-reference is available
    if georef_tif and os.path.exists(georef_tif):
        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(georef_tif) as ref:
                profile = ref.profile.copy()

            profile.update(
                dtype='float32',
                count=1,
                compress='deflate',
                nodata=np.nan,
            )

            # Resize vibration maps to match reference if needed
            h_ref, w_ref = profile['height'], profile['width']

            for arr, name in [
                (vibration_velocity, "vibration_amplitude"),
                (freq_map, "vibration_frequency"),
                (vibration_quality, "vibration_quality"),
            ]:
                # Simple resize if dimensions don't match
                if arr.shape != (h_ref, w_ref):
                    from scipy.ndimage import zoom
                    zoom_y = h_ref / arr.shape[0]
                    zoom_x = w_ref / arr.shape[1]
                    arr = zoom(arr, (zoom_y, zoom_x), order=1)

                tif_path = Path(output_dir) / f"{base_name}_{name}.tif"
                with rasterio.open(str(tif_path), 'w', **profile) as dst:
                    dst.write(arr.astype(np.float32), 1)
                    dst.set_band_description(1, name.replace("_", " ").title())
                outputs[name] = str(tif_path)

            logger.info("  Saved GeoTIFF outputs with geo-referencing")

        except Exception as e:
            logger.warning(f"  Failed to save GeoTIFFs: {e}. NumPy outputs still available.")

    # Log statistics
    logger.info("\n" + "=" * 60)
    logger.info("VIBROMETRY RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Vibration amplitude (mm/s): "
                f"min={vibration_velocity.min():.6f}, "
                f"max={vibration_velocity.max():.6f}, "
                f"mean={vibration_velocity.mean():.6f}")
    logger.info(f"  Dominant frequency (Hz): "
                f"min={freq_map.min():.2f}, "
                f"max={freq_map.max():.2f}, "
                f"mean={freq_map.mean():.2f}")
    logger.info(f"  Quality (coherence): "
                f"mean={vibration_quality.mean():.4f}, "
                f"pixels>{cfg['coherence_threshold']}: "
                f"{(vibration_quality > cfg['coherence_threshold']).sum()}/{vibration_quality.size}")
    logger.info("=" * 60)

    return outputs


def generate_synthetic_vibration_test(
    output_path: str,
    grid_size: int = 512,
    num_anomalies: int = 3,
    noise_level: float = 0.1,
) -> str:
    """
    Generate a synthetic SLC-like dataset with embedded vibration anomalies
    for testing the pipeline without real satellite data.

    Creates a complex radar image with known sub-surface void signatures
    implanted at specific locations. Useful for:
    - Validating the sub-aperture decomposition algorithm
    - Verifying the phase-to-velocity conversion
    - Testing the full pipeline end-to-end

    Parameters
    ----------
    output_path : str
        Path to save the synthetic SLC (.npy).
    grid_size : int
        Size of the synthetic image (grid_size x grid_size).
    num_anomalies : int
        Number of vibration anomalies to embed.
    noise_level : float
        Amplitude of background noise relative to signal.

    Returns
    -------
    str
        Path to the saved synthetic SLC file and ground truth file.
    """
    np.random.seed(42)

    # Generate base complex SAR backscatter (speckle)
    # Follows Rayleigh-distributed amplitude with uniform phase
    amplitude = np.random.rayleigh(scale=1.0, size=(grid_size, grid_size)).astype(np.float32)
    base_phase = np.random.uniform(-np.pi, np.pi, size=(grid_size, grid_size)).astype(np.float32)

    # Create known anomaly locations with vibration signatures
    anomaly_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    ground_truth = []

    for i in range(num_anomalies):
        # Random position (avoid edges)
        cy = np.random.randint(grid_size // 4, 3 * grid_size // 4)
        cx = np.random.randint(grid_size // 4, 3 * grid_size // 4)

        # Random size (10-50 pixels radius)
        radius = np.random.randint(10, 50)

        # Random vibration amplitude (0.1 to 1.0 mm/s)
        vib_amplitude = np.random.uniform(0.1, 1.0)

        # Random vibration frequency (1-20 Hz)
        vib_freq = np.random.uniform(1.0, 20.0)

        # Create circular anomaly
        yy, xx = np.ogrid[:grid_size, :grid_size]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = dist <= radius

        # Smooth edges with Gaussian falloff
        falloff = np.exp(-((dist - radius * 0.8) ** 2) / (2 * (radius * 0.2) ** 2))
        falloff[dist <= radius * 0.8] = 1.0
        falloff[dist > radius * 1.5] = 0.0

        anomaly_mask += falloff * vib_amplitude

        ground_truth.append({
            "id": i,
            "center_y": cy,
            "center_x": cx,
            "radius": radius,
            "vibration_amplitude_mm_s": vib_amplitude,
            "vibration_frequency_hz": vib_freq,
        })

        logger.info(
            f"  Anomaly {i}: center=({cy},{cx}), r={radius}, "
            f"amp={vib_amplitude:.2f}mm/s, freq={vib_freq:.1f}Hz"
        )

    # Convert vibration amplitude to phase modulation
    # phase_modulation ∝ vibration_velocity / (λ * PRF / N_sub)
    wavelength = 0.0555
    prf = 486.486
    n_sub = 5
    phase_modulation = anomaly_mask * (4 * np.pi * n_sub) / (wavelength * prf) * 1e-3

    # Apply vibration as modulated phase on top of base phase
    total_phase = base_phase + phase_modulation

    # Add noise
    noise_phase = noise_level * np.random.randn(grid_size, grid_size).astype(np.float32)
    total_phase += noise_phase

    # Reconstruct complex SLC
    slc_synthetic = (amplitude * np.exp(1j * total_phase)).astype(np.complex64)

    # Save SLC
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, slc_synthetic)
    logger.info(f"Saved synthetic SLC: {output_path} ({slc_synthetic.shape})")

    # Save ground truth
    gt_path = output_path.replace(".npy", "_ground_truth.npy")
    np.save(gt_path, anomaly_mask)

    gt_meta_path = output_path.replace(".npy", "_ground_truth.txt")
    with open(gt_meta_path, 'w') as f:
        f.write("# Synthetic Vibration Ground Truth\n")
        f.write(f"# Grid size: {grid_size}x{grid_size}\n")
        f.write(f"# Noise level: {noise_level}\n\n")
        for gt in ground_truth:
            f.write(f"Anomaly {gt['id']}:\n")
            for k, v in gt.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    logger.info(f"Saved ground truth: {gt_path}")
    return output_path


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAR Vibrometry: Doppler Sub-Aperture Processing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Process command
    proc_parser = subparsers.add_parser("process", help="Run vibrometry on SLC data")
    proc_parser.add_argument("--input", required=True, help="Path to SLC .npy file")
    proc_parser.add_argument("--output-dir", default=None, help="Output directory")
    proc_parser.add_argument("--num-sub", type=int, default=5, help="Number of sub-apertures")
    proc_parser.add_argument("--overlap", type=float, default=0.3, help="Sub-aperture overlap")
    proc_parser.add_argument("--georef", default=None, help="Reference GeoTIFF for geo-referencing")
    proc_parser.add_argument("--wavelength", type=float, default=0.0555, help="Radar wavelength (m)")
    proc_parser.add_argument("--prf", type=float, default=486.486, help="PRF (Hz)")

    # Synthetic test command
    synth_parser = subparsers.add_parser("synthetic", help="Generate synthetic test data")
    synth_parser.add_argument("--output", default="data/vibrometry/synthetic_slc.npy")
    synth_parser.add_argument("--size", type=int, default=512, help="Grid size")
    synth_parser.add_argument("--anomalies", type=int, default=3, help="Number of anomalies")
    synth_parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    synth_parser.add_argument("--run-pipeline", action="store_true", help="Also run pipeline on synthetic")

    args = parser.parse_args()

    if args.command == "process":
        config = {
            "num_sub_apertures": args.num_sub,
            "overlap_fraction": args.overlap,
            "radar_wavelength_m": args.wavelength,
            "prf_hz": args.prf,
        }
        outputs = run_vibrometry_pipeline(
            args.input,
            output_dir=args.output_dir,
            config=config,
            georef_tif=args.georef,
        )
        print("\nOutputs:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")

    elif args.command == "synthetic":
        slc_path = generate_synthetic_vibration_test(
            args.output,
            grid_size=args.size,
            num_anomalies=args.anomalies,
            noise_level=args.noise,
        )
        if args.run_pipeline:
            print("\nRunning pipeline on synthetic data...")
            outputs = run_vibrometry_pipeline(slc_path)
            print("\nOutputs:")
            for key, path in outputs.items():
                print(f"  {key}: {path}")

    else:
        parser.print_help()
