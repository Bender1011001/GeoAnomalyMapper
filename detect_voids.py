#!/usr/bin/env python3
"""
Dempster-Shafer (D-S) Decision Fusion for Void Detection.

Combines evidence from multiple sources (Gravity, InSAR, Poisson Analysis) using
Dempster-Shafer theory to estimate belief in 'Void', 'Solid', and 'Reinforced' states.

Frame of Discernment (Theta): {Void, Solid, Reinforced}

Outputs:
- fused_belief_reinforced.tif: Belief in 'Reinforced' state (0-1).
- void_probability.tif: Legacy compatibility (thresholded belief).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, FrozenSet
from itertools import chain, combinations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

# Try importing project paths, fallback to local if missing
try:
    from project_paths import OUTPUTS_DIR, PROCESSED_DIR
except ImportError:
    OUTPUTS_DIR = Path("outputs")
    PROCESSED_DIR = Path("processed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = OUTPUTS_DIR / "void_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Embedded Dempster-Shafer Logic (Removes dependency on pyds) ---

class LiteMassFunction:
    """
    A lightweight implementation of Dempster-Shafer Mass Functions 
    to avoid external dependencies like 'pyds'.
    """
    def __init__(self, masses: Dict[FrozenSet[str], float]):
        self.masses = masses
        # Normalize just in case
        total = sum(masses.values())
        if abs(total - 1.0) > 1e-6:
            self.masses = {k: v / total for k, v in masses.items()}

    def combine_dempster(self, other: 'LiteMassFunction') -> 'LiteMassFunction':
        """Combine this mass function with another using Dempster's Rule."""
        combined = {}
        conflict = 0.0

        for s1, m1 in self.masses.items():
            for s2, m2 in other.masses.items():
                intersection = s1.intersection(s2)
                product = m1 * m2
                
                if not intersection:
                    conflict += product
                else:
                    combined[intersection] = combined.get(intersection, 0.0) + product

        # Normalize by 1 - K (conflict)
        if conflict >= 1.0:
            raise ValueError("Total conflict between evidence sources.")
        
        scale = 1.0 / (1.0 - conflict)
        normalized_combined = {k: v * scale for k, v in combined.items()}
        return LiteMassFunction(normalized_combined)

    def bel(self, hypothesis: FrozenSet[str]) -> float:
        """Calculate Belief for a specific hypothesis (sum of masses of subsets)."""
        belief = 0.0
        for s, m in self.masses.items():
            if s.issubset(hypothesis):
                belief += m
        return belief

# --- 2. Raster Utilities ---

def load_and_match_grid(
    src_path: Path,
    match_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load a raster and reproject/resample it to match the grid of another raster.
    """
    if not src_path.exists():
        logger.warning(f"Source file not found: {src_path}")
        return None, None

    try:
        with rasterio.open(match_path) as match_ds:
            match_profile = match_ds.profile.copy()
            match_transform = match_ds.transform
            match_crs = match_ds.crs
            match_height, match_width = match_ds.height, match_ds.width

        with rasterio.open(src_path) as src_ds:
            # Initialize with NaN to represent missing data in the new grid
            dst_array = np.full((match_height, match_width), np.nan, dtype=np.float32)
            
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=dst_array,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=match_transform,
                dst_crs=match_crs,
                resampling=resampling,
                dst_nodata=np.nan,
            )

            # Update profile for consistency
            match_profile.update({
                'dtype': 'float32',
                'nodata': np.nan,
                'count': 1,
                'driver': 'GTiff'
            })

        return dst_array, match_profile

    except Exception as e:
        logger.error(f"Error reprojecting {src_path}: {e}")
        return None, None


# --- 3. Fusion Logic ---

def dempster_shafer_fusion(
    tdr_path: Path,
    artificiality_path: Path,
    poisson_path: Path,
    output_belief_path: Path,
    tdr_threshold: float = 0.5,
    artif_threshold: float = 0.7,
    poisson_threshold: float = -0.3,
    mass_tdr_vr: float = 0.6,
    mass_tdr_theta: float = 0.4,
    mass_artif_rs: float = 0.7,
    mass_artif_theta: float = 0.3,
    mass_poisson_r: float = 0.8,
    mass_poisson_theta: float = 0.2,
    target_mode: str = 'void',
) -> None:
    
    logger.info(f"Starting Dempster-Shafer Fusion (Mode: {target_mode.upper()})...")

    # --- Load Data ---
    if not tdr_path.exists():
        logger.error(f"Primary input TDR not found: {tdr_path}")
        return

    # Load TDR (Master Grid)
    with rasterio.open(tdr_path) as src:
        tdr = src.read(1)
        profile = src.profile.copy()
        # Clean profile for single band output
        profile.update({
            'dtype': 'float32', 
            'nodata': np.nan, 
            'count': 1,
            'compress': 'lzw'
        })
        nodata = src.nodata
        # Create validity mask
        if nodata is not None:
            mask = (tdr != nodata) & ~np.isnan(tdr)
        else:
            mask = ~np.isnan(tdr)
            
        tdr_abs = np.abs(tdr)

    # Load and align secondary rasters
    artif, _ = load_and_match_grid(artificiality_path, tdr_path)
    if artif is None:
        logger.warning("Artificiality raster missing, treating as neutral (0).")
        artif = np.zeros_like(tdr)

    poisson, _ = load_and_match_grid(poisson_path, tdr_path)
    if poisson is None:
        logger.warning("Poisson raster missing, treating as neutral (0).")
        poisson = np.zeros_like(tdr)

    # --- Precompute Lookup Table ---
    # Frame of discernment
    frame = frozenset(['Void', 'Solid', 'Reinforced'])
    theta = frame

    # Create masks
    # Determine the required sign for the gravity anomaly (TDR/Poisson)
    # Void: Mass Deficit (Negative anomaly/correlation)
    # Mineral: Mass Excess (Positive anomaly/correlation)
    
    # Use absolute value of poisson_threshold for comparison magnitude
    abs_poisson_thresh = abs(poisson_threshold)
    
    with np.errstate(invalid='ignore'):
        if target_mode == 'mineral':
            # Mineral Mode: Look for positive mass/correlation
            # TDR: We assume positive TDR indicates mass excess
            tdr_signal = (tdr > tdr_threshold) & mask
            
            # Poisson: Look for positive correlation above magnitude threshold
            poisson_signal = (poisson > abs_poisson_thresh) & mask & ~np.isnan(poisson)
            
        else:
            # Void Mode: Look for mass deficit/negative correlation
            # TDR: We use magnitude for edge detection, assuming TDR is an edge filter
            tdr_signal = (tdr_abs > tdr_threshold) & mask
            
            # Poisson: Look for negative correlation below negative threshold
            # Note: poisson_threshold is typically negative (e.g., -0.3)
            poisson_signal = (poisson < poisson_threshold) & mask & ~np.isnan(poisson)

        # Artificiality is same for now
        artif_signal = (artif > artif_threshold) & mask & ~np.isnan(artif)

    bel_lookup = np.zeros(8, dtype=np.float32)

    # Iterate through all 8 truth combinations of the 3 evidence layers
    for idx in range(8):
        t = (idx >> 2) & 1  # TDR signal?
        a = (idx >> 1) & 1  # Artificiality signal?
        p = idx & 1         # Poisson signal?

        # 1. Gravity TDR
        if t:
            if target_mode == 'mineral':
                # Supports {Mineral}
                m1 = LiteMassFunction({
                    frozenset(['Reinforced']): mass_tdr_vr, # Re-using 'Reinforced' label for 'Mineral' to save refactoring
                    theta: mass_tdr_theta,
                })
            else:
                # Supports {Void, Reinforced}
                m1 = LiteMassFunction({
                    frozenset(['Void', 'Reinforced']): mass_tdr_vr,
                    theta: mass_tdr_theta,
                })
        else:
            m1 = LiteMassFunction({theta: 1.0})

        # 2. InSAR: Supports {Reinforced, Solid} or Theta
        # MINERAL-PRO MODE: Neutralize artificiality bias.
        # We force this belief to be neutral (Theta) regardless of the input 'a'.
        # This allows natural voids (which lack artificial surface features) to be detected.
        # Original logic:
        # if a:
        #     m2 = LiteMassFunction({
        #         frozenset(['Reinforced', 'Solid']): mass_artif_rs,
        #         theta: mass_artif_theta,
        #     })
        # else:
        #     m2 = LiteMassFunction({theta: 1.0})
        
        m2 = LiteMassFunction({theta: 1.0})

        # 3. Poisson: Supports {Reinforced/Mineral} or Theta
        if p:
            m3 = LiteMassFunction({
                frozenset(['Reinforced']): mass_poisson_r,
                theta: mass_poisson_theta,
            })
        else:
            m3 = LiteMassFunction({theta: 1.0})

        try:
            m12 = m1.combine_dempster(m2)
            m_final = m12.combine_dempster(m3)
            # We specifically want Belief(Reinforced)
            bel_lookup[idx] = m_final.bel(frozenset(['Reinforced']))
        except ValueError:
            # Handle total conflict
            bel_lookup[idx] = 0.0

    # --- Apply Lookup to Raster ---
    logger.info("Applying fusion logic...")
    
    # Create index raster: (Bit 2: TDR) + (Bit 1: Artif) + (Bit 0: Poisson)
    # We cast to uint8 to save memory
    idx_raster = (tdr_signal.astype(np.uint8) << 2) | \
                 (artif_signal.astype(np.uint8) << 1) | \
                 (poisson_signal.astype(np.uint8))

    bel_reinforced = bel_lookup[idx_raster]
    
    # Apply global mask (where TDR was valid)
    bel_reinforced[~mask] = np.nan

    # --- Save Outputs ---
    logger.info(f"Saving fused belief to {output_belief_path}")
    output_belief_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_belief_path, 'w', **profile) as dst:
        dst.write(bel_reinforced.astype(np.float32), 1)
        dst.set_band_description(1, "Belief(Reinforced)")

    # Legacy Output: Thresholded Probability
    #void_prob_path = OUTPUT_DIR / "void_probability.tif"
    # Logic: If belief > 0.5, keep value, else 0
    #void_prob = np.where(bel_reinforced > 0.5, bel_reinforced, 0.0)
    
    #with rasterio.open(void_prob_path, 'w', **profile) as dst:
        #dst.write(void_prob.astype(np.float32), 1)

    logger.info("D-S Fusion complete.")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Dempster-Shafer Fusion")
    parser.add_argument("--tdr-thresh", type=float, default=0.5, help="TDR absolute threshold")
    parser.add_argument("--artif-thresh", type=float, default=0.7, help="Artificiality threshold")
    parser.add_argument("--poisson-thresh", type=float, default=-0.3, help="Poisson threshold (low)")
    parser.add_argument("--mode", type=str, default="void", help="Target mode: 'void' or 'mineral'")
    args = parser.parse_args()

    # Input paths
    tdr_path = PROCESSED_DIR / "gravity" / "gravity_tdr.tif"
    artif_path = PROCESSED_DIR / "insar" / "structural_artificiality.tif"
    poisson_path = PROCESSED_DIR / "poisson_correlation.tif"

    output_path = OUTPUT_DIR / "fused_belief_reinforced.tif"

    dempster_shafer_fusion(
        tdr_path=tdr_path,
        artificiality_path=artif_path,
        poisson_path=poisson_path,
        output_belief_path=output_path,
        tdr_threshold=args.tdr_thresh,
        artif_threshold=args.artif_thresh,
        poisson_threshold=args.poisson_thresh,
        target_mode=args.mode,
    )

if __name__ == "__main__":
    main()