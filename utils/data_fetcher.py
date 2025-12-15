#!/usr/bin/env python3
"""
Data Fetcher for Training Data in Supervised Learning

This module provides functions to fetch training data for supervised mineral exploration,
including integration with USGS Mineral Resources Data System (MRDS) and fallback
to curated known deposits.

Key Features:
- Fetch USGS MRDS data for specified regions
- Parse standard CSV format (lat, lon, deposit_type)
- Include hardcoded baseline of 17 known California deposits
- Filter deposits by type and region bounds
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import requests
import io
import os

logger = logging.getLogger(__name__)

# Hardcoded baseline: 17 known California mineral deposits
# Coordinates sourced from USGS MRDS, Mindat, and NASA records
CALIFORNIA_BASELINE_DEPOSITS = [
    {
        'name': 'Mountain Pass Mine',
        'lat': 35.4769,
        'lon': -115.5333,
        'type': 'Rare Earths',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Rio Tinto Boron Mine',
        'lat': 35.0429,
        'lon': -117.6793,
        'type': 'Borates',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Mesquite Mine',
        'lat': 33.0603,
        'lon': -114.9944,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Castle Mountain Mine',
        'lat': 35.2811,
        'lon': -115.1025,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Soledad Mountain Mine',
        'lat': 34.9978,
        'lon': -118.1806,
        'type': 'Gold/Silver',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Briggs Mine',
        'lat': 35.9375,
        'lon': -117.1850,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Golden Queen Mine',
        'lat': 34.9869,
        'lon': -118.1889,
        'type': 'Gold/Silver',
        'source': 'USGS MRDS'
    },
    {
        'name': 'McLaughlin Mine',
        'lat': 38.8381,
        'lon': -122.3639,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Gold Run Mining District',
        'lat': 39.1808,
        'lon': -120.8558,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Rand Mining District',
        'lat': 35.3500,
        'lon': -117.6500,
        'type': 'Gold/Silver/Tungsten',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Salton Sea Geothermal Field',
        'lat': 33.1863,
        'lon': -115.5844,
        'type': 'Lithium/Geothermal',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Eagle Mountain Mine',
        'lat': 33.8647,
        'lon': -115.5203,
        'type': 'Iron',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Iron Mountain Mine',
        'lat': 40.6722,
        'lon': -122.5278,
        'type': 'Iron/Copper/Zinc',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Leviathan Mine',
        'lat': 38.7081,
        'lon': -119.6572,
        'type': 'Sulfur/Copper',
        'source': 'USGS MRDS'
    },
    {
        'name': 'New Idria Mercury Mine',
        'lat': 36.4144,
        'lon': -120.6736,
        'type': 'Mercury',
        'source': 'USGS MRDS'
    },
    {
        'name': 'Dale Mining District',
        'lat': 34.05,
        'lon': -115.75,
        'type': 'Gold',
        'source': 'USGS MRDS'
    },
    {
        'name': 'New Target Candidate 1',
        'lat': 33.9515,
        'lon': -116.0035,
        'type': 'Unknown',
        'source': 'PINN Model'
    }
]


def parse_deposit_csv(csv_path: str, lat_col: str = 'lat', lon_col: str = 'lon',
                      type_col: str = 'deposit_type') -> List[Dict]:
    """
    Parse a CSV file containing deposit data.

    Args:
        csv_path: Path to CSV file
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        type_col: Name of deposit type column

    Returns:
        List of deposit dictionaries with keys: 'name', 'lat', 'lon', 'type', 'source'
    """
    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = [lat_col, lon_col, type_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        deposits = []
        for idx, row in df.iterrows():
            try:
                deposit = {
                    'name': f"Deposit_{idx}",
                    'lat': float(row[lat_col]),
                    'lon': float(row[lon_col]),
                    'type': str(row[type_col]),
                    'source': f"CSV:{Path(csv_path).name}"
                }
                deposits.append(deposit)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row {idx}: {e}")
                continue

        logger.info(f"Parsed {len(deposits)} deposits from {csv_path}")
        return deposits

    except Exception as e:
        logger.error(f"Failed to parse CSV {csv_path}: {e}")
        return []


def fetch_usgs_mrds(region_bounds: Tuple[float, float, float, float],
                    deposit_types: Optional[List[str]] = None,
                    csv_path: Optional[str] = None) -> List[Dict]:
    """
    Fetch mineral deposit data for a given region.

    This function provides a flexible interface for obtaining training data:
    - If csv_path is provided, loads data from local CSV file
    - Otherwise, falls back to hardcoded California baseline deposits
    - Filters by region bounds and deposit types

    Args:
        region_bounds: (lon_min, lat_min, lon_max, lat_max)
        deposit_types: List of deposit types to include (None for all)
        csv_path: Path to local CSV file (optional)

    Returns:
        List of deposit dictionaries
    """
    lon_min, lat_min, lon_max, lat_max = region_bounds

    # Load deposits from CSV if provided
    if csv_path and Path(csv_path).exists():
        deposits = parse_deposit_csv(csv_path)
        logger.info(f"Loaded {len(deposits)} deposits from {csv_path}")
    else:
        # Fallback to hardcoded California deposits
        deposits = CALIFORNIA_BASELINE_DEPOSITS.copy()
        logger.info(f"Using {len(deposits)} hardcoded California baseline deposits")

    # Filter by region bounds
    filtered_deposits = []
    for deposit in deposits:
        lat, lon = deposit['lat'], deposit['lon']
        if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
            filtered_deposits.append(deposit)

    logger.info(f"Filtered to {len(filtered_deposits)} deposits within region bounds")

    # Filter by deposit types if specified
    if deposit_types:
        type_filtered = []
        for deposit in filtered_deposits:
            # Check if deposit type matches any of the requested types
            deposit_type_lower = deposit['type'].lower()
            if any(req_type.lower() in deposit_type_lower or
                   deposit_type_lower in req_type.lower()
                   for req_type in deposit_types):
                type_filtered.append(deposit)

        logger.info(f"Filtered to {len(type_filtered)} deposits matching types: {deposit_types}")
        return type_filtered

    return filtered_deposits


def get_training_coordinates(deposits: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract coordinates and labels from deposit list for training.

    Args:
        deposits: List of deposit dictionaries

    Returns:
        Tuple of (coordinates_array, labels_list)
        coordinates_array: shape (n_samples, 2) with [lat, lon]
        labels_list: corresponding deposit names
    """
    if not deposits:
        return np.array([]).reshape(0, 2), []

    coords = []
    labels = []

    for deposit in deposits:
        coords.append([deposit['lat'], deposit['lon']])
        labels.append(deposit['name'])

    return np.array(coords), labels


def validate_deposit_data(deposits: List[Dict]) -> bool:
    """
    Validate deposit data structure and coordinate ranges.

    Args:
        deposits: List of deposit dictionaries to validate

    Returns:
        True if all deposits are valid
    """
    if not deposits:
        logger.warning("No deposits to validate")
        return False

    valid_count = 0
    for i, deposit in enumerate(deposits):
        try:
            # Check required keys
            required_keys = ['name', 'lat', 'lon', 'type']
            missing_keys = [key for key in required_keys if key not in deposit]
            if missing_keys:
                logger.warning(f"Deposit {i} missing keys: {missing_keys}")
                continue

            # Validate coordinate ranges
            lat, lon = deposit['lat'], deposit['lon']
            if not (-90 <= lat <= 90):
                logger.warning(f"Deposit {i} ({deposit['name']}) has invalid latitude: {lat}")
                continue
            if not (-180 <= lon <= 180):
                logger.warning(f"Deposit {i} ({deposit['name']}) has invalid longitude: {lon}")
                continue

            valid_count += 1

        except Exception as e:
            logger.warning(f"Error validating deposit {i}: {e}")
            continue

    logger.info(f"Validated {valid_count}/{len(deposits)} deposits")
    return valid_count == len(deposits)


def fetch_usgs_training_data(local_cache_path: str = 'data/usgs_mrds_full.csv') -> pd.DataFrame:
    """
    Fetches and filters the USGS MRDS database for high-quality training sites ("Goldilocks" dataset).
    
    Filters:
    1. Development Status: Producer, Past Producer (removes Occurrences)
    2. Primary Commodity: Copper, Lithium, Rare Earths, Iron, Gold, Silver, Nickel, Zinc
    3. Deposit Type: Excludes surficial/placer deposits that lack deep gravity signatures
    
    Args:
        local_cache_path: Path to cache the downloaded full dataset
        
    Returns:
        DataFrame of filtered high-quality training sites
    """
    # Create data directory if it doesn't exist
    Path(local_cache_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data (Download or Cache)
    if Path(local_cache_path).exists():
        logger.info(f"Loading cached USGS database from {local_cache_path}...")
        try:
            df = pd.read_csv(local_cache_path, low_memory=False)
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}. Re-downloading...")
            df = None
            
    if not Path(local_cache_path).exists() or df is None:
        url = "https://mrdata.usgs.gov/mrds/mrds.csv"
        logger.info(f"Downloading USGS Database from {url}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8', errors='replace')), low_memory=False)
            
            # Save to cache
            logger.info(f"Caching database to {local_cache_path}")
            df.to_csv(local_cache_path, index=False)
        except Exception as e:
            logger.error(f"Failed to download USGS data: {e}")
            raise RuntimeError("Could not fetch USGS MRDS database")

    logger.info(f"Total entries in MRDS: {len(df)}")

    # 2. FILTER: Status = Producer / Past Producer
    # "Occurrence" and "Prospect" are too noisy for gravity inversion training
    # Standardizing status strings
    if 'dev_stat' not in df.columns:
        logger.warning("Column 'dev_stat' not found. Available columns: " + str(df.columns[:10]))
        return pd.DataFrame()
        
    # Filter for producers
    df['dev_stat'] = df['dev_stat'].astype(str).str.title()
    high_confidence = df[df['dev_stat'].isin(['Producer', 'Past Producer'])]
    logger.info(f"Filtered by Status (Producer/Past Producer): {len(high_confidence)}")

    # 2b. FILTER: Country = United States
    if 'country' in df.columns:
        high_confidence = high_confidence[high_confidence['country'] == 'United States']
        logger.info(f"Filtered by Country (United States): {len(high_confidence)}")
    else:
        logger.warning("Column 'country' not found, skipping country filter")

    # 3. FILTER: Target Commodities
    # Focusing on commodities that form massive deposits detectable by gravity
    targets = ['Copper', 'Lithium', 'Rare Earths', 'Iron', 'Gold', 'Silver', 'Zinc', 'Nickel']
    
    if 'commod1' in high_confidence.columns:
        # Check if primary commodity contains any of our targets
        # case insensitive match
        pattern = '|'.join(targets)
        relevant_minerals = high_confidence[high_confidence['commod1'].astype(str).str.contains(pattern, case=False, na=False)]
        logger.info(f"Filtered by Commodity ({', '.join(targets)}): {len(relevant_minerals)}")
    else:
        logger.warning("Column 'commod1' not found, skipping commodity filter")
        relevant_minerals = high_confidence

    # 4. FILTER: Massive Deposit Types (Tier 1)
    # Exclude "Vein", "Shear zone", "Pegmatite" (unless huge), "Placer"
    # Include only massive types that create density contrast
    
    if 'dep_type' in relevant_minerals.columns:
        # Positive inclusion list for massive deposits
        massive_types = [
            'Porphyry', 'VMS', 'Volcanogenic massive sulfide', 
            'SEDEX', 'Sedimentary exhalative', 
            'Skarn', 'IOCG', 'Iron oxide copper gold',
            'Mississipi Valley', 'MVT',
            'Kimberlite', 'Carbonatite',
            'Epithermal', 
            'Massive sulfide',
            'Stratabound',
            'Stockwork',
            'Lode',           # Structurally controlled but significant
            'Replacement',    # e.g., Carbonate Replacement (CRD)
            'Disseminated',   # Large volume low grade
            'Breccia',        # Breccia pipes
            'Manto',          # Manto type
            'Polymetallic'    # Poly-metallic replacement/vein systems
        ]
        
        # Strict Inclusion Logic
        # We only keep deposits that match our "Massive" whitelist.
        allowed_pattern = '|'.join(massive_types)
        
        # Convert to string and handle NaNs
        dep_types = relevant_minerals['dep_type'].astype(str)
        
        # Filter for allowed types
        strict_included = relevant_minerals[dep_types.str.contains(allowed_pattern, case=False, na=False)]
        logger.info(f"Filtered by Strict Allowed Types: {len(strict_included)}")
        
        # We strictly trust this list for the "Goldilocks" dataset.
        hard_rock = strict_included
    else:
        hard_rock = relevant_minerals

    # Select and rename columns for consistency
    # Looking for: name, lat, lon, type
    # MRDS columns: site_name, latitude, longitude, dep_type, commod1
    
    final_df = hard_rock[['site_name', 'latitude', 'longitude', 'commod1', 'dev_stat', 'dep_type']].copy()
    final_df = final_df.rename(columns={
        'site_name': 'name',
        'latitude': 'lat',
        'longitude': 'lon',
        'dep_type': 'type'
    })
    
    # Return formatted DataFrame
    logger.info(f"Final 'Goldilocks' training set size: {len(final_df)} sites")
    return final_df
