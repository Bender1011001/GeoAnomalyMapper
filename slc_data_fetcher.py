#!/usr/bin/env python3
"""
SLC (Single Look Complex) Data Fetcher for SAR Vibrometry
==========================================================

This module fetches raw Sentinel-1 SLC data products required for the
Biondi SAR Doppler Tomography pipeline. Unlike standard InSAR workflows
that use pre-processed coherence or unwrapped phase maps, this pipeline
requires the RAW phase and amplitude information before Doppler bands
are finalized.

Why SLC?
--------
Standard InSAR averages out high-frequency Doppler content during
multi-looking and interferometric processing. The Biondi methodology
requires the full azimuth bandwidth to perform sub-aperture
decomposition — splitting the synthetic aperture into temporal slices
to track sub-millimeter surface vibrations caused by subsurface
resonance.

Supported data sources:
  - Sentinel-1 C-band SLC (free, via Alaska Satellite Facility / asf_search)
  - Capella Space X-band SLC (commercial, higher resolution)
  - Umbra SAR X-band SLC (commercial, highest resolution)

Usage:
    python slc_data_fetcher.py --lat 29.9792 --lon 31.1342 --buffer 0.5
    python slc_data_fetcher.py --bbox 30.8 29.5 31.5 30.5 --max-results 10
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np

from project_paths import DATA_DIR, RAW_DIR, ensure_directories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# Configuration
# ============================================================
SLC_DIR = DATA_DIR / "slc"
SLC_RAW_DIR = SLC_DIR / "raw"
SLC_BURSTS_DIR = SLC_DIR / "bursts"
SLC_METADATA_DIR = SLC_DIR / "metadata"

DEFAULT_SEARCH_PARAMS = {
    "platform": "Sentinel-1",
    "processingLevel": "SLC",
    "beamMode": "IW",         # Interferometric Wide swath — standard for InSAR
    "flightDirection": None,   # ASCENDING or DESCENDING (None = both)
    "polarization": "VV",      # VV is best for surface deformation monitoring
    "maxResults": 50,
    "lookback_days": 365,      # Search window (1 year default)
}


def _build_search_bbox(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    buffer_deg: float = 0.5,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Build a WGS84 bounding box from either lat/lon+buffer or explicit bbox.

    Returns (west, south, east, north).
    """
    if bbox is not None:
        return tuple(bbox)
    if lat is not None and lon is not None:
        return (
            lon - buffer_deg,
            lat - buffer_deg,
            lon + buffer_deg,
            lat + buffer_deg
        )
    raise ValueError("Provide either (lat, lon) or bbox")


def search_sentinel1_slc(
    bbox: Tuple[float, float, float, float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 50,
    flight_direction: Optional[str] = None,
    polarization: str = "VV"
) -> List[Dict]:
    """
    Search for Sentinel-1 SLC products covering a bounding box.

    Uses the asf_search library to query the Alaska Satellite Facility
    DAAC (Distributed Active Archive Center), which hosts the complete
    Sentinel-1 archive.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84 degrees.
    start_date : str, optional
        ISO format start date. Defaults to 1 year ago.
    end_date : str, optional
        ISO format end date. Defaults to today.
    max_results : int
        Maximum number of results to return.
    flight_direction : str, optional
        'ASCENDING' or 'DESCENDING'. None for both.
    polarization : str
        Polarization mode ('VV', 'VH', 'VV+VH').

    Returns
    -------
    list of dict
        Each dict contains product metadata: granule name, URL, dates,
        orbit info, and geometry.
    """
    try:
        import asf_search as asf
    except ImportError:
        logger.error(
            "asf_search not installed. Install via: pip install asf_search"
        )
        raise

    # Default date range: last year
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if start_date is None:
        start_dt = datetime.utcnow() - timedelta(
            days=DEFAULT_SEARCH_PARAMS["lookback_days"]
        )
        start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    west, south, east, north = bbox
    logger.info(
        f"Searching Sentinel-1 SLC: bbox=({west:.3f},{south:.3f},{east:.3f},{north:.3f}), "
        f"dates={start_date} to {end_date}"
    )

    # Build search parameters
    search_kwargs = {
        "platform": asf.PLATFORM.SENTINEL1,
        "processingLevel": asf.PRODUCT_TYPE.SLC,
        "beamMode": asf.BEAMMODE.IW,
        "intersectsWith": f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))",
        "start": start_date,
        "end": end_date,
        "maxResults": max_results,
    }

    if flight_direction:
        if flight_direction.upper() == "ASCENDING":
            search_kwargs["flightDirection"] = asf.FLIGHT_DIRECTION.ASCENDING
        elif flight_direction.upper() == "DESCENDING":
            search_kwargs["flightDirection"] = asf.FLIGHT_DIRECTION.DESCENDING

    results = asf.search(**search_kwargs)
    logger.info(f"Found {len(results)} SLC products")

    products = []
    for result in results:
        props = result.properties
        product_info = {
            "granule_name": props.get("sceneName", "unknown"),
            "download_url": result.properties.get("url", ""),
            "file_id": props.get("fileID", ""),
            "start_time": props.get("startTime", ""),
            "stop_time": props.get("stopTime", ""),
            "flight_direction": props.get("flightDirection", ""),
            "orbit_number": props.get("orbit", 0),
            "relative_orbit": props.get("pathNumber", 0),
            "frame_number": props.get("frameNumber", 0),
            "polarization": props.get("polarization", ""),
            "beam_mode": props.get("beamModeType", ""),
            "geometry_wkt": props.get("geometry", {}).get("coordinates", []),
            "size_mb": props.get("bytes", 0) / (1024 * 1024) if props.get("bytes") else 0,
            "processing_level": "SLC",
        }
        products.append(product_info)

    # Sort by date (most recent first)
    products.sort(key=lambda p: p["start_time"], reverse=True)
    return products


def download_slc_product(
    product: Dict,
    output_dir: Optional[Path] = None,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
) -> Path:
    """
    Download a single Sentinel-1 SLC product from ASF.

    Requires NASA Earthdata credentials. Set via environment variables:
        EARTHDATA_USERNAME, EARTHDATA_PASSWORD
    or pass directly.

    Parameters
    ----------
    product : dict
        Product metadata from search_sentinel1_slc().
    output_dir : Path, optional
        Directory to save the downloaded file.
    earthdata_username : str, optional
        NASA Earthdata username.
    earthdata_password : str, optional
        NASA Earthdata password.

    Returns
    -------
    Path
        Path to the downloaded .zip file.
    """
    try:
        import asf_search as asf
    except ImportError:
        raise ImportError("asf_search required: pip install asf_search")

    if output_dir is None:
        output_dir = SLC_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    granule_name = product["granule_name"]

    # ---- CACHE CHECK: skip download if file already exists ----
    expected_zip = output_dir / f"{granule_name}.zip"
    expected_safe = output_dir / f"{granule_name}.SAFE"

    if expected_zip.exists() and expected_zip.stat().st_size > 0:
        logger.info(f"SLC already downloaded (cached): {expected_zip}  ({expected_zip.stat().st_size / 1024**2:.0f} MB)")
        return expected_zip
    if expected_safe.exists():
        logger.info(f"SLC already downloaded (cached): {expected_safe}")
        return expected_safe

    # Resolve credentials
    username = earthdata_username or os.environ.get("EARTHDATA_USERNAME", "")
    password = earthdata_password or os.environ.get("EARTHDATA_PASSWORD", "")
    if not username or not password:
        raise EnvironmentError(
            "NASA Earthdata credentials required. Set EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD environment variables, or pass them directly."
        )

    logger.info(f"Downloading SLC: {granule_name} ({product.get('size_mb', 0):.0f} MB)")

    # Search again by granule name to get downloadable result object
    results = asf.granule_search([granule_name])
    if not results:
        raise FileNotFoundError(f"Could not find granule: {granule_name}")

    # Create ASF session
    session = asf.ASFSession()
    session.auth_with_creds(username, password)

    # Download
    results[0].download(str(output_dir), session=session)

    # Find the downloaded file
    expected_zip = output_dir / f"{granule_name}.zip"
    expected_safe = output_dir / f"{granule_name}.SAFE"

    if expected_zip.exists():
        logger.info(f"Downloaded: {expected_zip}")
        return expected_zip
    elif expected_safe.exists():
        logger.info(f"Downloaded: {expected_safe}")
        return expected_safe
    else:
        # Find any recently created file
        files = sorted(output_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            logger.info(f"Downloaded: {files[0]}")
            return files[0]
        raise FileNotFoundError(f"Download completed but file not found in {output_dir}")


def extract_slc_burst(
    safe_path: Path,
    swath: str = "IW2",
    burst_indices: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Extract individual bursts from a Sentinel-1 SLC SAFE product.

    Sentinel-1 IW mode acquires data in bursts within 3 sub-swaths
    (IW1, IW2, IW3). For vibrometry, we need the raw complex data
    from individual bursts for sub-aperture decomposition.

    Parameters
    ----------
    safe_path : Path
        Path to the .SAFE directory or .zip file.
    swath : str
        Sub-swath to extract ('IW1', 'IW2', or 'IW3').
        IW2 typically has best overlap for interferometry.
    burst_indices : list of int, optional
        Specific burst indices to extract (0-based). None = all bursts.
    output_dir : Path, optional
        Output directory for extracted burst data.

    Returns
    -------
    list of Path
        Paths to extracted burst complex data files (.npy format).
    """
    import zipfile
    import xml.etree.ElementTree as ET

    if output_dir is None:
        output_dir = SLC_BURSTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_dir = safe_path
    tmp_extract = None

    # Handle .zip files
    if safe_path.suffix == ".zip":
        tmp_extract = SLC_RAW_DIR / "tmp_extract"
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(safe_path, 'r') as zf:
            zf.extractall(tmp_extract)
        # Find the .SAFE directory inside
        safe_dirs = list(tmp_extract.glob("*.SAFE"))
        if not safe_dirs:
            raise FileNotFoundError(f"No .SAFE directory found in {safe_path}")
        safe_dir = safe_dirs[0]

    logger.info(f"Extracting bursts from {safe_dir.name}, swath={swath}")

    # Locate the measurement TIFF for the requested swath and VV polarization
    measurement_dir = safe_dir / "measurement"
    annotation_dir = safe_dir / "annotation"

    if not measurement_dir.exists():
        raise FileNotFoundError(f"No measurement directory in {safe_dir}")

    # Find matching TIFF (pattern: s1*-iw2-slc-vv-*.tiff)
    swath_lower = swath.lower()
    tiff_files = list(measurement_dir.glob(f"*-{swath_lower}-slc-vv-*.tiff"))
    if not tiff_files:
        tiff_files = list(measurement_dir.glob(f"*-{swath_lower}-slc-*.tiff"))
    if not tiff_files:
        raise FileNotFoundError(
            f"No SLC TIFF found for swath {swath} in {measurement_dir}"
        )

    tiff_path = tiff_files[0]
    logger.info(f"SLC TIFF: {tiff_path.name}")

    # Find matching annotation XML for burst boundaries
    ann_files = list(annotation_dir.glob(f"*-{swath_lower}-slc-vv-*.xml"))
    if not ann_files:
        ann_files = list(annotation_dir.glob(f"*-{swath_lower}-slc-*.xml"))

    burst_boundaries = []
    if ann_files:
        tree = ET.parse(ann_files[0])
        root = tree.getroot()

        # Parse burst list from annotation XML
        burst_list = root.find(".//burstList")
        if burst_list is not None:
            num_bursts = int(burst_list.get("count", 0))
            logger.info(f"Found {num_bursts} bursts in annotation")

            for burst_elem in burst_list.findall("burst"):
                first_valid = burst_elem.find("firstValidSample")
                byte_offset_elem = burst_elem.find("byteOffset")
                azimuth_time = burst_elem.find("azimuthTime")

                burst_info = {
                    "azimuth_time": azimuth_time.text if azimuth_time is not None else "",
                    "byte_offset": int(byte_offset_elem.text) if byte_offset_elem is not None else 0,
                }
                burst_boundaries.append(burst_info)

    # Read the SLC TIFF as complex data using rasterio
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for reading SLC TIFFs")

    with rasterio.open(str(tiff_path)) as src:
        # SLC data is stored as complex values (I + jQ)
        # Rasterio may read this as 2-band (I, Q) or as complex64
        num_bands = src.count
        height = src.height
        width = src.width

        logger.info(f"SLC dimensions: {height}x{width}, bands={num_bands}")

        if num_bands >= 2:
            # I and Q stored as separate bands
            band_i = src.read(1).astype(np.float32)
            band_q = src.read(2).astype(np.float32)
            slc_complex = band_i + 1j * band_q
        else:
            # Single band — may already be complex or need reinterpretation
            raw = src.read(1)
            if np.iscomplexobj(raw):
                slc_complex = raw.astype(np.complex64)
            else:
                # Reinterpret as complex (interleaved I/Q pairs)
                raw_flat = raw.flatten().astype(np.float32)
                if len(raw_flat) % 2 == 0:
                    slc_complex = raw_flat[0::2] + 1j * raw_flat[1::2]
                    new_width = width // 2
                    slc_complex = slc_complex.reshape(height, new_width)
                else:
                    slc_complex = raw.astype(np.complex64)

        profile = src.profile.copy()

    # If we have burst boundaries, split the data
    extracted_paths = []

    if burst_boundaries and len(burst_boundaries) > 1:
        # Estimate lines per burst
        lines_per_burst = height // len(burst_boundaries)

        if burst_indices is None:
            burst_indices = list(range(len(burst_boundaries)))

        for idx in burst_indices:
            if idx >= len(burst_boundaries):
                logger.warning(f"Burst index {idx} out of range, skipping")
                continue

            start_line = idx * lines_per_burst
            end_line = min((idx + 1) * lines_per_burst, height)

            burst_data = slc_complex[start_line:end_line, :]
            burst_file = output_dir / f"{safe_dir.stem}_{swath}_burst{idx:02d}.npy"
            np.save(burst_file, burst_data)
            extracted_paths.append(burst_file)
            logger.info(
                f"  Burst {idx}: lines {start_line}-{end_line}, "
                f"shape={burst_data.shape}, saved to {burst_file.name}"
            )
    else:
        # No burst info — save entire swath as single file
        burst_file = output_dir / f"{safe_dir.stem}_{swath}_full.npy"
        np.save(burst_file, slc_complex)
        extracted_paths.append(burst_file)
        logger.info(f"  Full swath: shape={slc_complex.shape}, saved to {burst_file.name}")

    # Save metadata
    metadata = {
        "safe_name": safe_dir.name,
        "swath": swath,
        "total_bursts": len(burst_boundaries),
        "extracted_bursts": len(extracted_paths),
        "slc_dimensions": f"{height}x{width}",
        "extracted_files": [str(p) for p in extracted_paths],
    }

    metadata_file = SLC_METADATA_DIR / f"{safe_dir.stem}_{swath}_metadata.txt"
    SLC_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")

    # Cleanup temp extraction
    if tmp_extract and tmp_extract.exists():
        import shutil
        shutil.rmtree(tmp_extract, ignore_errors=True)

    return extracted_paths


def search_capella_slc(
    bbox: Tuple[float, float, float, float],
    api_key: Optional[str] = None,
    max_results: int = 10
) -> List[Dict]:
    """
    Search Capella Space catalog for X-band SLC imagery.

    Capella provides SAR data at sub-meter resolution, ideal for
    achieving the extreme detail of Biondi's methodology. Their
    Open Data Program provides some free scenes.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84 degrees.
    api_key : str, optional
        Capella API key. Falls back to CAPELLA_API_KEY env var.
    max_results : int
        Maximum number of results.

    Returns
    -------
    list of dict
        Product metadata from Capella catalog.
    """
    import requests

    api_key = api_key or os.environ.get("CAPELLA_API_KEY", "")
    if not api_key:
        logger.warning(
            "No Capella API key provided. Set CAPELLA_API_KEY environment variable "
            "or pass api_key parameter. Returning empty results."
        )
        return []

    west, south, east, north = bbox

    # Capella STAC API
    stac_url = "https://api.capellaspace.com/catalog/stac/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "bbox": [west, south, east, north],
        "limit": max_results,
        "collections": ["capella-slc"],
        "query": {
            "sar:product_type": {"eq": "SLC"},
            "sar:frequency_band": {"eq": "X"}
        }
    }

    try:
        response = requests.post(stac_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        products = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            product_info = {
                "granule_name": feature.get("id", "unknown"),
                "provider": "Capella Space",
                "band": "X-band",
                "resolution_m": props.get("sar:resolution_range", 0.5),
                "start_time": props.get("datetime", ""),
                "polarization": props.get("sar:polarizations", ["HH"])[0],
                "orbit_state": props.get("sat:orbit_state", ""),
                "processing_level": "SLC",
                "assets": {k: v.get("href", "") for k, v in feature.get("assets", {}).items()},
            }
            products.append(product_info)

        logger.info(f"Capella search: found {len(products)} X-band SLC products")
        return products

    except requests.RequestException as e:
        logger.error(f"Capella API search failed: {e}")
        return []


def build_slc_inventory(
    lat: float,
    lon: float,
    buffer_deg: float = 0.5,
    include_commercial: bool = False,
) -> Dict:
    """
    Build a complete inventory of available SLC data for a target location.

    Searches all configured data sources and returns a unified catalog.

    Parameters
    ----------
    lat : float
        Target latitude.
    lon : float
        Target longitude.
    buffer_deg : float
        Search buffer in degrees.
    include_commercial : bool
        Whether to include commercial providers (Capella, Umbra).

    Returns
    -------
    dict
        Inventory with keys: 'sentinel1', 'capella', 'summary'.
    """
    bbox = _build_search_bbox(lat=lat, lon=lon, buffer_deg=buffer_deg)

    inventory = {
        "target": {"lat": lat, "lon": lon, "buffer_deg": buffer_deg},
        "bbox": bbox,
        "sentinel1": [],
        "capella": [],
        "summary": {},
    }

    # Sentinel-1 (free)
    try:
        s1_products = search_sentinel1_slc(bbox, max_results=50)
        inventory["sentinel1"] = s1_products
    except Exception as e:
        logger.warning(f"Sentinel-1 search failed: {e}")

    # Commercial providers
    if include_commercial:
        try:
            capella_products = search_capella_slc(bbox)
            inventory["capella"] = capella_products
        except Exception as e:
            logger.warning(f"Capella search failed: {e}")

    # Summary
    inventory["summary"] = {
        "total_sentinel1": len(inventory["sentinel1"]),
        "total_capella": len(inventory["capella"]),
        "total_all": len(inventory["sentinel1"]) + len(inventory["capella"]),
    }

    logger.info(
        f"SLC Inventory: {inventory['summary']['total_sentinel1']} Sentinel-1, "
        f"{inventory['summary']['total_capella']} Capella"
    )
    return inventory


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search and download SLC SAR data for vibrometry analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for SLC products")
    search_parser.add_argument("--lat", type=float, help="Target latitude")
    search_parser.add_argument("--lon", type=float, help="Target longitude")
    search_parser.add_argument("--buffer", type=float, default=0.5, help="Search buffer (degrees)")
    search_parser.add_argument(
        "--bbox", nargs=4, type=float,
        help="Bounding box: west south east north"
    )
    search_parser.add_argument("--max-results", type=int, default=20)
    search_parser.add_argument("--start-date", type=str, default=None)
    search_parser.add_argument("--end-date", type=str, default=None)
    search_parser.add_argument(
        "--direction", choices=["ASCENDING", "DESCENDING"], default=None
    )
    search_parser.add_argument("--commercial", action="store_true")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a specific SLC product")
    dl_parser.add_argument("granule", help="Granule name to download")
    dl_parser.add_argument("--output-dir", type=str, default=None)

    # Extract command
    ext_parser = subparsers.add_parser("extract", help="Extract bursts from SLC .SAFE")
    ext_parser.add_argument("safe_path", help="Path to .SAFE directory or .zip")
    ext_parser.add_argument("--swath", default="IW2", choices=["IW1", "IW2", "IW3"])
    ext_parser.add_argument("--bursts", nargs="*", type=int, default=None)

    args = parser.parse_args()

    if args.command == "search":
        bbox = _build_search_bbox(
            lat=args.lat, lon=args.lon,
            buffer_deg=args.buffer, bbox=args.bbox
        )
        products = search_sentinel1_slc(
            bbox,
            start_date=args.start_date,
            end_date=args.end_date,
            max_results=args.max_results,
            flight_direction=args.direction,
        )

        print(f"\n{'='*80}")
        print(f"Found {len(products)} Sentinel-1 SLC products")
        print(f"{'='*80}")
        for i, p in enumerate(products[:20]):
            print(
                f"  [{i+1:2d}] {p['granule_name']}\n"
                f"       Date: {p['start_time'][:10]}  "
                f"Orbit: {p['relative_orbit']}  "
                f"Dir: {p['flight_direction']}  "
                f"Size: {p['size_mb']:.0f} MB"
            )

        if args.commercial:
            capella = search_capella_slc(bbox)
            if capella:
                print(f"\n{'='*80}")
                print(f"Found {len(capella)} Capella X-band SLC products")
                print(f"{'='*80}")
                for i, p in enumerate(capella):
                    print(
                        f"  [{i+1:2d}] {p['granule_name']}\n"
                        f"       Date: {p['start_time'][:10]}  "
                        f"Res: {p['resolution_m']}m  "
                        f"Band: {p['band']}"
                    )

    elif args.command == "download":
        product = {"granule_name": args.granule}
        output_dir = Path(args.output_dir) if args.output_dir else None
        path = download_slc_product(product, output_dir=output_dir)
        print(f"Downloaded to: {path}")

    elif args.command == "extract":
        paths = extract_slc_burst(
            Path(args.safe_path),
            swath=args.swath,
            burst_indices=args.bursts,
        )
        print(f"Extracted {len(paths)} burst files:")
        for p in paths:
            print(f"  {p}")

    else:
        parser.print_help()
