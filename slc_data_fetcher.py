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
   decomposition - splitting the synthetic aperture into temporal slices
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
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Mapping, Sequence
from datetime import datetime, timedelta, timezone

import numpy as np

from json_utils import dumps_strict_json
from project_paths import DATA_DIR

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
    "beamMode": "IW",         # Interferometric Wide swath - standard for InSAR
    "flightDirection": None,   # ASCENDING or DESCENDING (None = both)
    "polarization": "VV",      # VV is best for surface deformation monitoring
    "maxResults": 50,
    "lookback_days": 365,      # Search window (1 year default)
}

EARTHDATA_USERNAME_ENV = "EARTHDATA_USERNAME"
EARTHDATA_PASSWORD_ENV = "EARTHDATA_PASSWORD"
EARTHDATA_TOKEN_ENV_VARS = ("EARTHDATA_TOKEN", "EARTHDATA_BEARER_TOKEN")
ASF_SEARCH_TIMEOUT_SECONDS_ENV = "ASF_SEARCH_TIMEOUT_SECONDS"
ASF_SEARCH_MAX_RETRIES_ENV = "ASF_SEARCH_MAX_RETRIES"
ASF_SEARCH_RETRY_BACKOFF_SECONDS_ENV = "ASF_SEARCH_RETRY_BACKOFF_SECONDS"
ASF_DOWNLOAD_MAX_RETRIES_ENV = "ASF_DOWNLOAD_MAX_RETRIES"
ASF_DOWNLOAD_RETRY_BACKOFF_SECONDS_ENV = "ASF_DOWNLOAD_RETRY_BACKOFF_SECONDS"

DEFAULT_ASF_SEARCH_TIMEOUT_SECONDS = 90
DEFAULT_ASF_SEARCH_MAX_RETRIES = 3
DEFAULT_ASF_SEARCH_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_ASF_DOWNLOAD_MAX_RETRIES = 2
DEFAULT_ASF_DOWNLOAD_RETRY_BACKOFF_SECONDS = 10.0


def _read_env_int(name: str, default: int, min_value: int = 1) -> int:
    """Read a bounded integer env var without failing on malformed local config."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid integer %s=%r; using default %s", name, raw, default)
        return default
    return max(min_value, value)


def _read_env_float(name: str, default: float, min_value: float = 0.0) -> float:
    """Read a bounded float env var without failing on malformed local config."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid float %s=%r; using default %s", name, raw, default)
        return default
    return max(min_value, value)


def get_asf_search_config(
    timeout_seconds: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_backoff_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve ASF/CMR search timeout and retry settings from args/env/defaults."""
    return {
        "timeout_seconds": max(
            1,
            int(
                timeout_seconds
                if timeout_seconds is not None
                else _read_env_int(
                    ASF_SEARCH_TIMEOUT_SECONDS_ENV,
                    DEFAULT_ASF_SEARCH_TIMEOUT_SECONDS,
                    min_value=1,
                )
            ),
        ),
        "max_retries": max(
            1,
            int(
                max_retries
                if max_retries is not None
                else _read_env_int(
                    ASF_SEARCH_MAX_RETRIES_ENV,
                    DEFAULT_ASF_SEARCH_MAX_RETRIES,
                    min_value=1,
                )
            ),
        ),
        "retry_backoff_seconds": max(
            0.0,
            float(
                retry_backoff_seconds
                if retry_backoff_seconds is not None
                else _read_env_float(
                    ASF_SEARCH_RETRY_BACKOFF_SECONDS_ENV,
                    DEFAULT_ASF_SEARCH_RETRY_BACKOFF_SECONDS,
                    min_value=0.0,
                )
            ),
        ),
    }


def get_asf_download_config(
    max_retries: Optional[int] = None,
    retry_backoff_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve lightweight retry settings for ASF granule lookup/download attempts."""
    return {
        "max_retries": max(
            1,
            int(
                max_retries
                if max_retries is not None
                else _read_env_int(
                    ASF_DOWNLOAD_MAX_RETRIES_ENV,
                    DEFAULT_ASF_DOWNLOAD_MAX_RETRIES,
                    min_value=1,
                )
            ),
        ),
        "retry_backoff_seconds": max(
            0.0,
            float(
                retry_backoff_seconds
                if retry_backoff_seconds is not None
                else _read_env_float(
                    ASF_DOWNLOAD_RETRY_BACKOFF_SECONDS_ENV,
                    DEFAULT_ASF_DOWNLOAD_RETRY_BACKOFF_SECONDS,
                    min_value=0.0,
                )
            ),
        ),
    }


def configure_asf_cmr_timeout(timeout_seconds: int) -> None:
    """Set the asf_search CMR timeout constant when the installed package exposes it."""
    try:
        from asf_search.constants import INTERNAL
    except Exception as exc:
        logger.debug("Could not import asf_search CMR timeout constant: %s", exc)
        return

    try:
        timeout_value = max(1, int(timeout_seconds))
        previous = getattr(INTERNAL, "CMR_TIMEOUT", None)
        if previous != timeout_value:
            setattr(INTERNAL, "CMR_TIMEOUT", timeout_value)
            logger.info(
                "Configured ASF/CMR timeout: %ss (previous=%s)",
                timeout_value,
                previous,
            )
    except Exception as exc:
        logger.debug("Could not set asf_search CMR timeout: %s", exc)


def _redact_known_secrets(text: str) -> str:
    """Remove locally configured secret values from diagnostic exception text."""
    redacted = str(text)
    secret_candidates = []
    for env_name in (EARTHDATA_USERNAME_ENV, EARTHDATA_PASSWORD_ENV, *EARTHDATA_TOKEN_ENV_VARS):
        cleaned = _clean_secret_value(os.environ.get(env_name, ""))
        if cleaned:
            secret_candidates.append(cleaned)
            if env_name in EARTHDATA_TOKEN_ENV_VARS:
                normalized = normalize_earthdata_token(cleaned)
                if normalized:
                    secret_candidates.append(normalized)
    for secret in sorted(set(secret_candidates), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def safe_exception_summary(exc: BaseException, max_chars: int = 800) -> str:
    """Return a compact, secret-redacted exception summary for logs/results."""
    summary = f"{exc.__class__.__name__}: {exc}"
    summary = _redact_known_secrets(summary).replace("\n", " ").replace("\r", " ")
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def classify_asf_acquisition_error(exc: BaseException) -> str:
    """Classify ASF/CMR acquisition failures for retry and final diagnostics."""
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text or "too long to respond" in text:
        return "timeout"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth_or_permission"
    if "429" in text or "rate" in text and "limit" in text:
        return "rate_limited"
    if any(code in text for code in ("500", "502", "503", "504")) or "server error" in text:
        return "transient_server"
    if "connection" in text or "network" in text or "ssl" in text or "dns" in text:
        return "network"
    if "400" in text or "404" in text or "bad request" in text or "not found" in text:
        return "client_query_or_not_found"
    return "unknown"


def _is_retryable_asf_error(exc: BaseException) -> bool:
    """Return True for transient ASF/CMR failures that are worth retrying."""
    return classify_asf_acquisition_error(exc) in {
        "timeout",
        "rate_limited",
        "transient_server",
        "network",
        "unknown",
    }


def _sleep_before_retry(attempt: int, backoff_seconds: float) -> None:
    """Back off between retries; zero backoff is useful for unit tests."""
    delay = max(0.0, float(backoff_seconds)) * max(1, attempt)
    if delay > 0:
        time.sleep(delay)


def load_env_file(env_path: Optional[Path] = None) -> bool:
    """Load a local dotenv file for CLI commands without printing secret values."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    env_path = Path(env_path)
    if not env_path.exists():
        return False
    loaded = 0
    with open(env_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
            loaded += 1
    logger.info("Loaded %d environment entries from %s (secret values redacted)", loaded, env_path)
    return True


def _clean_secret_value(value: Optional[str]) -> str:
    """Normalize a locally supplied secret without logging or persisting it."""
    if value is None:
        return ""
    cleaned = str(value).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def normalize_earthdata_token(token: Optional[str]) -> str:
    """Return a bearer token value with whitespace/optional Bearer prefix removed."""
    cleaned = _clean_secret_value(token)
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    return cleaned


def describe_secret_presence(value: Optional[str]) -> str:
    """Describe a secret for logs without exposing any of its contents."""
    cleaned = _clean_secret_value(value)
    if not cleaned:
        return "not set"
    return f"set (length={len(cleaned)}, masked=<redacted>)"


def resolve_earthdata_auth(
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    Select the safest available NASA Earthdata auth mode.

    Bearer-token auth is preferred when EARTHDATA_TOKEN or
    EARTHDATA_BEARER_TOKEN is configured. Username/password remains supported
    as a fallback for older local setups.
    """
    env = os.environ if environ is None else environ

    token = normalize_earthdata_token(earthdata_token)
    token_source = "argument" if token else ""
    if not token:
        for env_name in EARTHDATA_TOKEN_ENV_VARS:
            candidate = normalize_earthdata_token(env.get(env_name, ""))
            if candidate:
                token = candidate
                token_source = env_name
                break

    username = _clean_secret_value(earthdata_username) or _clean_secret_value(
        env.get(EARTHDATA_USERNAME_ENV, "")
    )
    password = _clean_secret_value(earthdata_password) or _clean_secret_value(
        env.get(EARTHDATA_PASSWORD_ENV, "")
    )

    if token:
        mode = "token"
    elif username and password:
        mode = "credentials"
    elif username or password:
        mode = "incomplete_credentials"
    else:
        mode = "none"

    return {
        "mode": mode,
        "token": token,
        "token_source": token_source,
        "username": username,
        "password": password,
    }


def describe_earthdata_auth(auth_info: Dict[str, Any]) -> str:
    """Return a log-safe description of the selected Earthdata auth mode."""
    mode = auth_info.get("mode", "none")
    if mode == "token":
        source = auth_info.get("token_source") or "provided value"
        return (
            f"token via {source} "
            f"({describe_secret_presence(auth_info.get('token'))})"
        )
    if mode == "credentials":
        return (
            "username/password "
            f"(username: {describe_secret_presence(auth_info.get('username'))}; "
            f"password: {describe_secret_presence(auth_info.get('password'))})"
        )
    if mode == "incomplete_credentials":
        return (
            "incomplete username/password "
            f"(username: {describe_secret_presence(auth_info.get('username'))}; "
            f"password: {describe_secret_presence(auth_info.get('password'))})"
        )
    return "none (no bearer token or complete username/password configured)"


def create_earthdata_asf_session(asf: Any, auth_info: Dict[str, Any]) -> Any:
    """Create an ASF session using token auth when available, otherwise credentials."""
    mode = auth_info.get("mode")
    if mode not in {"token", "credentials"}:
        raise EnvironmentError(
            "NASA Earthdata authentication required. Set EARTHDATA_TOKEN or "
            "EARTHDATA_BEARER_TOKEN, or set EARTHDATA_USERNAME and "
            f"EARTHDATA_PASSWORD. Current status: {describe_earthdata_auth(auth_info)}"
        )

    session = asf.ASFSession()
    logger.info("Using Earthdata authentication: %s", describe_earthdata_auth(auth_info))

    if mode == "token":
        if not hasattr(session, "auth_with_token"):
            raise EnvironmentError(
                "Installed asf_search does not expose token authentication. "
                "Upgrade asf_search or use EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
            )
        try:
            session.auth_with_token(auth_info["token"])
        except Exception as exc:
            if auth_info.get("username") and auth_info.get("password"):
                logger.warning(
                    "Earthdata token auth failed (%s); retrying with username/password credentials.",
                    safe_exception_summary(exc),
                )
                session = asf.ASFSession()
                session.auth_with_creds(auth_info["username"], auth_info["password"])
            else:
                raise
    else:
        session.auth_with_creds(auth_info["username"], auth_info["password"])

    return session


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
    polarization: str = "VV",
    search_timeout_seconds: Optional[int] = None,
    search_max_retries: Optional[int] = None,
    search_retry_backoff_seconds: Optional[float] = None,
    asf_session: Optional[Any] = None,
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
    search_timeout_seconds : int, optional
        ASF/CMR request timeout. Defaults to ASF_SEARCH_TIMEOUT_SECONDS or 90.
    search_max_retries : int, optional
        Total search attempts. Defaults to ASF_SEARCH_MAX_RETRIES or 3.
    search_retry_backoff_seconds : float, optional
        Linear backoff multiplier between attempts.
    asf_session : object, optional
        Authenticated ASFSession to use for CMR queries, primarily for preflight.

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
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if start_date is None:
        start_dt = datetime.now(timezone.utc) - timedelta(
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

    if asf_session is not None and hasattr(asf, "ASFSearchOptions"):
        search_kwargs["opts"] = asf.ASFSearchOptions(session=asf_session)

    search_config = get_asf_search_config(
        timeout_seconds=search_timeout_seconds,
        max_retries=search_max_retries,
        retry_backoff_seconds=search_retry_backoff_seconds,
    )
    configure_asf_cmr_timeout(search_config["timeout_seconds"])

    last_exc: Optional[BaseException] = None
    results = None
    for attempt in range(1, search_config["max_retries"] + 1):
        try:
            logger.info(
                "ASF/CMR search attempt %d/%d (timeout=%ss, max_results=%s)",
                attempt,
                search_config["max_retries"],
                search_config["timeout_seconds"],
                max_results,
            )
            results = asf.search(**search_kwargs)
            break
        except Exception as exc:
            last_exc = exc
            classification = classify_asf_acquisition_error(exc)
            summary = safe_exception_summary(exc)
            if attempt >= search_config["max_retries"] or not _is_retryable_asf_error(exc):
                logger.error(
                    "ASF/CMR search failed after attempt %d/%d; classification=%s; error=%s",
                    attempt,
                    search_config["max_retries"],
                    classification,
                    summary,
                )
                raise
            logger.warning(
                "ASF/CMR search attempt %d/%d failed; classification=%s; retrying; error=%s",
                attempt,
                search_config["max_retries"],
                classification,
                summary,
            )
            _sleep_before_retry(attempt, search_config["retry_backoff_seconds"])

    if results is None:
        raise RuntimeError(
            "ASF/CMR search failed without returning results"
            + (f": {safe_exception_summary(last_exc)}" if last_exc else "")
        )
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
    earthdata_token: Optional[str] = None,
    download_max_retries: Optional[int] = None,
    download_retry_backoff_seconds: Optional[float] = None,
    expected_product_id: Optional[str] = None,
    require_product_identity: bool = False,
) -> Path:
    """
    Download a single Sentinel-1 SLC product from ASF.

    Requires NASA Earthdata authentication. Preferred environment variables:
        EARTHDATA_TOKEN or EARTHDATA_BEARER_TOKEN
    Username/password is still supported via:
        EARTHDATA_USERNAME, EARTHDATA_PASSWORD
    or pass values directly.

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
    earthdata_token : str, optional
        NASA Earthdata bearer token. A leading "Bearer " prefix is accepted
        and stripped before authenticating with ASF.
    download_max_retries : int, optional
        Total granule lookup/download attempts. Defaults to ASF_DOWNLOAD_MAX_RETRIES or 2.
    download_retry_backoff_seconds : float, optional
        Linear backoff multiplier between attempts.
    expected_product_id : str, optional
        Locked product identifier that must match product metadata before any
        download is attempted. The check accepts product_id, file_id,
        granule_name, or product_name aliases because ASF metadata exposes both
        scene and file identifiers.
    require_product_identity : bool
        When True, fail before download if the expected locked product identity
        cannot be verified from local metadata and ASF granule lookup metadata.

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

    identity_candidates = [
        product.get("product_id"),
        product.get("file_id"),
        product.get("granule_name"),
        product.get("product_name"),
        product.get("sceneName"),
        product.get("fileID"),
        product.get("id"),
    ]
    cleaned_identity_candidates = {str(item) for item in identity_candidates if item}
    if expected_product_id is not None and str(expected_product_id) not in cleaned_identity_candidates:
        raise ValueError(
            "Locked product identity mismatch before download: "
            f"expected {expected_product_id!r}, available identifiers={sorted(cleaned_identity_candidates)!r}"
        )
    if require_product_identity and expected_product_id is None:
        raise ValueError("require_product_identity=True requires expected_product_id")

    granule_name = (
        product.get("granule_name")
        or product.get("product_name")
        or product.get("sceneName")
        or product.get("file_id")
        or product.get("product_id")
    )
    if not granule_name:
        raise ValueError("SLC product metadata must include granule_name/product_name/file_id/product_id")
    granule_name = str(granule_name)

    # ---- CACHE CHECK: skip download if file already exists ----
    expected_zip = output_dir / f"{granule_name}.zip"
    expected_safe = output_dir / f"{granule_name}.SAFE"

    if expected_zip.exists() and expected_zip.stat().st_size > 100 * 1024**2:
        import zipfile
        try:
            with zipfile.ZipFile(expected_zip, 'r'):
                pass  # opening validates the ZIP central directory
            logger.info(f"SLC already downloaded (cached): {expected_zip}  ({expected_zip.stat().st_size / 1024**2:.0f} MB)")
            return expected_zip
        except Exception as e:
            logger.warning(f"Cached ZIP is corrupt, deleting: {expected_zip} ({e})")
            expected_zip.unlink(missing_ok=True)
            
    if expected_safe.exists():
        logger.info(f"SLC already downloaded (cached): {expected_safe}")
        return expected_safe

    auth_info = resolve_earthdata_auth(
        earthdata_username=earthdata_username,
        earthdata_password=earthdata_password,
        earthdata_token=earthdata_token,
    )
    if auth_info["mode"] not in {"token", "credentials"}:
        raise EnvironmentError(
            "NASA Earthdata authentication required. Set EARTHDATA_TOKEN or "
            "EARTHDATA_BEARER_TOKEN, or set EARTHDATA_USERNAME and "
            f"EARTHDATA_PASSWORD. Current status: {describe_earthdata_auth(auth_info)}"
        )

    download_config = get_asf_download_config(
        max_retries=download_max_retries,
        retry_backoff_seconds=download_retry_backoff_seconds,
    )

    logger.info(f"Downloading SLC: {granule_name} ({product.get('size_mb', 0):.0f} MB)")

    last_exc: Optional[BaseException] = None
    for attempt in range(1, download_config["max_retries"] + 1):
        try:
            logger.info(
                "ASF granule lookup/download attempt %d/%d for %s",
                attempt,
                download_config["max_retries"],
                granule_name,
            )
            # Search again by granule name to get downloadable result object.
            results = asf.granule_search([granule_name])
            if not results:
                raise FileNotFoundError(f"Could not find granule: {granule_name}")
            if expected_product_id is not None:
                result_props = getattr(results[0], "properties", {}) or {}
                result_identifiers = {
                    str(item)
                    for item in (
                        result_props.get("sceneName"),
                        result_props.get("fileID"),
                        result_props.get("id"),
                        result_props.get("granule_name"),
                    )
                    if item
                }
                if result_identifiers and str(expected_product_id) not in result_identifiers:
                    raise ValueError(
                        "Locked product identity mismatch after ASF granule lookup: "
                        f"expected {expected_product_id!r}, ASF identifiers={sorted(result_identifiers)!r}"
                    )
                if require_product_identity and not result_identifiers:
                    raise ValueError(
                        "ASF granule lookup returned no identifiers; cannot verify locked product identity"
                    )

            # Create ASF session. This logs only auth mode/presence/length, never secret contents.
            session = create_earthdata_asf_session(asf, auth_info)

            # Download.
            results[0].download(str(output_dir), session=session)
            break
        except Exception as exc:
            last_exc = exc
            classification = classify_asf_acquisition_error(exc)
            summary = safe_exception_summary(exc)
            if attempt >= download_config["max_retries"] or not _is_retryable_asf_error(exc):
                logger.error(
                    "ASF granule lookup/download failed after attempt %d/%d; "
                    "classification=%s; error=%s",
                    attempt,
                    download_config["max_retries"],
                    classification,
                    summary,
                )
                raise
            logger.warning(
                "ASF granule lookup/download attempt %d/%d failed; classification=%s; "
                "retrying; error=%s",
                attempt,
                download_config["max_retries"],
                classification,
                summary,
            )
            _sleep_before_retry(attempt, download_config["retry_backoff_seconds"])
    else:
        raise RuntimeError(
            "ASF granule lookup/download failed without completing"
            + (f": {safe_exception_summary(last_exc)}" if last_exc else "")
        )

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


def _parse_float_text(element: Any, child_name: str) -> Optional[float]:
    child = element.find(child_name)
    if child is None or child.text is None:
        return None
    try:
        return float(child.text)
    except (TypeError, ValueError):
        return None


def _parse_sentinel1_geolocation_points(root: Any) -> List[Dict[str, float]]:
    """Parse Sentinel-1 annotation geolocation-grid points into line/pixel/lat/lon dicts."""
    points: List[Dict[str, float]] = []
    for point in root.findall(".//geolocationGridPoint"):
        line = _parse_float_text(point, "line")
        pixel = _parse_float_text(point, "pixel")
        latitude = _parse_float_text(point, "latitude")
        longitude = _parse_float_text(point, "longitude")
        if None in (line, pixel, latitude, longitude):
            continue
        points.append(
            {
                "line": float(line),
                "pixel": float(pixel),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )
    return points


def _bounded_window(center_line: float, center_pixel: float, shape: Tuple[int, int], chip_shape: Tuple[int, int]) -> Dict[str, int]:
    height, width = int(shape[0]), int(shape[1])
    chip_h = min(max(1, int(chip_shape[0])), height)
    chip_w = min(max(1, int(chip_shape[1])), width)

    row_start = int(round(float(center_line))) - chip_h // 2
    col_start = int(round(float(center_pixel))) - chip_w // 2
    row_start = max(0, min(row_start, height - chip_h))
    col_start = max(0, min(col_start, width - chip_w))
    return {
        "row_start": row_start,
        "row_stop": row_start + chip_h,
        "col_start": col_start,
        "col_stop": col_start + chip_w,
    }


def extract_target_local_slc_chip(
    slc_complex: np.ndarray,
    target_lat: float,
    target_lon: float,
    geolocation_points: Sequence[Mapping[str, Any]],
    chip_shape: Tuple[int, int] = (4096, 4096),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract a target-local SLC chip using Sentinel-1 geolocation grid metadata.

    Refuses to fall back to a product-level full burst/center crop when no usable
    geolocation grid is available, because that would make separated targets in
    the same locked product reuse identical vibrometry inputs.
    """
    if slc_complex.ndim != 2:
        raise ValueError(f"SLC chip extraction expects a 2D array; got shape={slc_complex.shape}")

    usable_points: List[Dict[str, float]] = []
    for point in geolocation_points:
        try:
            line = float(point["line"])
            pixel = float(point["pixel"])
            latitude = float(point["latitude"])
            longitude = float(point["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        usable_points.append(
            {
                "line": line,
                "pixel": pixel,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    if not usable_points:
        raise ValueError(
            "target-local SLC chip requires geolocation metadata; refusing to reuse "
            "a non-target-local full burst or center crop"
        )

    target_lat = float(target_lat)
    target_lon = float(target_lon)
    nearest = min(
        usable_points,
        key=lambda point: (point["latitude"] - target_lat) ** 2 + (point["longitude"] - target_lon) ** 2,
    )
    window = _bounded_window(
        nearest["line"],
        nearest["pixel"],
        (int(slc_complex.shape[0]), int(slc_complex.shape[1])),
        chip_shape,
    )
    chip = slc_complex[
        window["row_start"]:window["row_stop"],
        window["col_start"]:window["col_stop"],
    ]
    if chip.shape == slc_complex.shape:
        raise ValueError(
            "target-local SLC chip would cover the full source; refusing to reuse "
            "a non-target-local full burst"
        )
    metadata = {
        "method": "geolocation_grid_nearest",
        "fallback_index_mapping_used": False,
        "source_shape": [int(slc_complex.shape[0]), int(slc_complex.shape[1])],
        "requested_chip_shape": [int(chip_shape[0]), int(chip_shape[1])],
        "target": {"lat": target_lat, "lon": target_lon},
        "selected_geolocation_point": nearest,
        "window": {
            **window,
            "shape": [int(chip.shape[0]), int(chip.shape[1])],
            "center_line": float(nearest["line"]),
            "center_pixel": float(nearest["pixel"]),
        },
    }
    return chip, metadata


def extract_slc_burst(
    safe_path: Path,
    swath: str = "IW2",
    burst_indices: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    target_bbox: Optional[Tuple[float, float, float, float]] = None,
    target_chip_shape: Tuple[int, int] = (4096, 4096),
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
    target_lat, target_lon : float, optional
        Target coordinates. When supplied, extraction writes one target-local chip
        using annotation geolocation metadata instead of returning the first/full
        burst. Missing geolocation metadata raises ValueError rather than silently
        producing a non-target-local input.
    target_bbox : tuple, optional
        Target bbox metadata persisted for traceability; target_lat/target_lon are
        still required for geolocation-grid selection.
    target_chip_shape : tuple of int
        Maximum target-local chip shape as (rows, columns).

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

    if safe_path.suffix == ".zip":
        import uuid
        tmp_extract = SLC_RAW_DIR / f"tmp_extract_{uuid.uuid4().hex[:8]}"
        tmp_extract.mkdir(parents=True, exist_ok=True)

        # Selective extraction: only measurement TIFF + annotation XML for requested swath
        # This avoids extracting the full 5.5GB SAFE archive (saves ~4GB disk)
        swath_pattern = swath.lower()  # e.g. "iw2"
        with zipfile.ZipFile(safe_path, 'r') as zf:
            members_to_extract = []
            for name in zf.namelist():
                name_lower = name.lower()
                # Extract only measurement TIFFs and annotation XMLs for our swath
                if ("measurement/" in name_lower and swath_pattern in name_lower and
                        name_lower.endswith('.tiff')):
                    members_to_extract.append(name)
                elif ("annotation/" in name_lower and swath_pattern in name_lower and
                      name_lower.endswith('.xml') and 'calibration' not in name_lower):
                    members_to_extract.append(name)
                # Also extract directory entries for structure
                elif name.endswith('/') and ('.SAFE' in name):
                    members_to_extract.append(name)

            logger.info(f"Selective extraction: {len(members_to_extract)} files from ZIP "
                        f"(skipping {len(zf.namelist()) - len(members_to_extract)} unneeded files)")
            for member in members_to_extract:
                zf.extract(member, tmp_extract)

        # Delete ZIP immediately to free ~5.5GB
        # try:
        #     safe_path.unlink()
        #     logger.info(f"Deleted ZIP after selective extraction: {safe_path.name}")
        # except Exception as e:
        #     logger.warning(f"Could not delete ZIP: {e}")

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
    geolocation_points: List[Dict[str, float]] = []
    if ann_files:
        tree = ET.parse(ann_files[0])
        root = tree.getroot()
        geolocation_points = _parse_sentinel1_geolocation_points(root)

        # Parse burst list from annotation XML
        burst_list = root.find(".//burstList")
        if burst_list is not None:
            num_bursts = int(burst_list.get("count", 0))
            logger.info(f"Found {num_bursts} bursts in annotation")

            for burst_elem in burst_list.findall("burst"):
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
            # Single band - may already be complex or need reinterpretation
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

    # If target coordinates are provided, extract a target-local chip from the
    # geolocated swath. This avoids silently feeding the same first burst or
    # center crop to different campaign targets locked to the same SAR product.
    extracted_paths = []
    target_local_metadata: Optional[Dict[str, Any]] = None

    if target_lat is not None or target_lon is not None:
        if target_lat is None or target_lon is None:
            raise ValueError("target-local SLC extraction requires both target_lat and target_lon")
        target_chip, target_local_metadata = extract_target_local_slc_chip(
            slc_complex,
            target_lat=float(target_lat),
            target_lon=float(target_lon),
            geolocation_points=geolocation_points,
            chip_shape=target_chip_shape,
        )
        if target_bbox is not None:
            target_local_metadata["target_bbox"] = [float(value) for value in target_bbox]
        stem = f"{safe_dir.stem}_{swath}_target_local"
        burst_file = output_dir / f"{stem}.npy"
        np.save(burst_file, target_chip)
        extracted_paths.append(burst_file)
        metadata_path = output_dir / f"{stem}_metadata.json"
        metadata_path.write_text(dumps_strict_json(target_local_metadata, indent=2), encoding="utf-8")
        logger.info(
            "  Target-local chip: window=%s, shape=%s, metadata=%s",
            target_local_metadata["window"],
            target_chip.shape,
            metadata_path.name,
        )

    # If we have burst boundaries, split the data
    elif burst_boundaries and len(burst_boundaries) > 1:
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
        # No burst info - save entire swath as single file
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
    if target_local_metadata is not None:
        metadata["target_local_window"] = target_local_metadata["window"]
        metadata["target_local_method"] = target_local_metadata["method"]
        metadata["target_local_fallback_index_mapping_used"] = target_local_metadata[
            "fallback_index_mapping_used"
        ]

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


def search_umbra_open_data(
    bbox: Tuple[float, float, float, float],
    max_results: int = 10,
) -> List[Dict]:
    """
    Search the Umbra Space Open Data catalog on AWS S3 for high-resolution
    X-band SAR SLC imagery.

    Umbra provides some of the highest resolution SAR data in the world
    (down to 16cm resolution) and hosts a massive free archive on AWS:
        s3://umbra-open-data-catalog/

    This is the PREFERRED data source for Biondi SAR Doppler Tomography
    because:
      1. X-band (~3cm wavelength) penetrates shallow structures better than C-band
      2. Spotlight mode: the satellite "stares" at the target for up to 60s,
         providing the massive dwell time needed to extract sub-millimeter
         Doppler shifts from subsurface resonance
      3. 16cm-1m resolution is 20-100x finer than Sentinel-1's 5x20m pixels

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84 degrees.
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    list of dict
        Product metadata from the Umbra open data catalog.
    """
    import requests

    west, south, east, north = bbox
    logger.info(
        f"Searching Umbra Open Data: bbox=({west:.3f},{south:.3f},{east:.3f},{north:.3f})"
    )

    # Try the STAC search endpoint first
    stac_search_url = "https://api.canopy.umbra.space/archive/search"
    headers = {"Content-Type": "application/json"}

    payload = {
        "bbox": [west, south, east, north],
        "limit": max_results,
        "query": {
            "sar:product_type": {"eq": "SLC"},
        }
    }

    products = []

    try:
        response = requests.post(stac_search_url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                assets = feature.get("assets", {})

                # Find the SLC GeoTIFF asset
                slc_asset = None
                for asset_key, asset_val in assets.items():
                    href = asset_val.get("href", "")
                    if href.endswith(".tif") or href.endswith(".tiff"):
                        slc_asset = href
                        break

                product_info = {
                    "granule_name": feature.get("id", "unknown"),
                    "provider": "Umbra Space",
                    "band": "X-band",
                    "resolution_m": props.get("sar:resolution_range", 0.5),
                    "start_time": props.get("datetime", ""),
                    "polarization": props.get("sar:polarizations", ["VV"])[0] if isinstance(
                        props.get("sar:polarizations"), list) else "VV",
                    "orbit_state": props.get("sat:orbit_state", ""),
                    "processing_level": "SLC",
                    "download_url": slc_asset or "",
                    "s3_prefix": f"s3://umbra-open-data-catalog/{feature.get('id', '')}/",
                    "assets": {k: v.get("href", "") for k, v in assets.items()},
                }
                products.append(product_info)

            logger.info(f"Umbra STAC search: found {len(products)} X-band SLC products")
        else:
            logger.warning(f"Umbra STAC search returned {response.status_code}, trying S3 listing fallback")
            # Fallback: try direct S3 listing (no auth needed for open data)
            products = _search_umbra_s3_fallback(bbox, max_results)

    except requests.RequestException as e:
        logger.warning(f"Umbra API search failed: {e}, trying S3 listing fallback")
        products = _search_umbra_s3_fallback(bbox, max_results)

    return products


def _search_umbra_s3_fallback(
    bbox: Tuple[float, float, float, float],
    max_results: int = 10,
) -> List[Dict]:
    """
    Fallback: list Umbra open data directly from AWS S3 (no API key needed).

    The S3 bucket is public and contains STAC item JSONs with bounding box
    metadata that we can filter client-side.
    """
    import requests

    bucket_url = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
    west, south, east, north = bbox

    products = []
    try:
        # List the root catalog to find collection indices
        catalog_url = f"{bucket_url}/stac/catalog.json"
        resp = requests.get(catalog_url, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Umbra S3 catalog not accessible: HTTP {resp.status_code}")
            return []

        catalog = resp.json()
        # Get links to child catalogs/collections
        child_links = [
            link for link in catalog.get("links", [])
            if link.get("rel") == "child"
        ]

        # Sample a subset of collections to find matches
        checked = 0
        for link in child_links[:50]:  # Check first 50 collections
            if len(products) >= max_results:
                break
            try:
                child_url = link.get("href", "")
                if not child_url.startswith("http"):
                    child_url = f"{bucket_url}/stac/{child_url}"

                child_resp = requests.get(child_url, timeout=10)
                if child_resp.status_code != 200:
                    continue

                child_data = child_resp.json()
                child_bbox = child_data.get("extent", {}).get("spatial", {}).get("bbox", [[]])
                if child_bbox and child_bbox[0]:
                    cb = child_bbox[0]  # [west, south, east, north]
                    # Check if bboxes overlap
                    if (cb[0] <= east and cb[2] >= west and
                        cb[1] <= north and cb[3] >= south):
                        # Found an overlapping collection - get its items
                        item_links = [
                            link_entry for link_entry in child_data.get("links", [])
                            if link_entry.get("rel") == "item"
                        ]
                        for item_link in item_links[:5]:
                            item_url = item_link.get("href", "")
                            if not item_url.startswith("http"):
                                # Resolve relative URL
                                base = child_url.rsplit("/", 1)[0]
                                item_url = f"{base}/{item_url}"

                            item_resp = requests.get(item_url, timeout=10)
                            if item_resp.status_code == 200:
                                item = item_resp.json()
                                props = item.get("properties", {})
                                assets = item.get("assets", {})

                                slc_url = ""
                                for ak, av in assets.items():
                                    href = av.get("href", "")
                                    if href.endswith(".tif") or href.endswith(".tiff"):
                                        slc_url = href
                                        break

                                products.append({
                                    "granule_name": item.get("id", "unknown"),
                                    "provider": "Umbra Space (S3)",
                                    "band": "X-band",
                                    "resolution_m": props.get("sar:resolution_range", 0.5),
                                    "start_time": props.get("datetime", ""),
                                    "polarization": "VV",
                                    "processing_level": "SLC",
                                    "download_url": slc_url,
                                    "assets": {k: v.get("href", "") for k, v in assets.items()},
                                })
                checked += 1
            except Exception:
                continue

        logger.info(f"Umbra S3 fallback: checked {checked} collections, found {len(products)} matching products")

    except Exception as e:
        logger.warning(f"Umbra S3 fallback failed: {e}")

    return products


def download_umbra_slc(
    product: Dict,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Download an Umbra Space SLC GeoTIFF from their open data catalog.

    No authentication is needed - Umbra's open data is publicly hosted on AWS S3.

    Parameters
    ----------
    product : dict
        Product metadata from search_umbra_open_data().
    output_dir : Path, optional
        Directory to save the downloaded file.

    Returns
    -------
    Path or None
        Path to the downloaded GeoTIFF, or None if download failed.
    """
    import requests

    if output_dir is None:
        output_dir = SLC_RAW_DIR / "umbra"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = product.get("download_url", "")
    if not url:
        logger.warning(f"No download URL for Umbra product: {product.get('granule_name')}")
        return None

    granule_name = product.get("granule_name", "unknown")
    output_path = output_dir / f"{granule_name}.tif"

    # Cache check
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Umbra SLC already downloaded (cached): {output_path} "
                     f"({output_path.stat().st_size / 1024**2:.0f} MB)")
        return output_path

    logger.info(f"Downloading Umbra SLC: {granule_name} from {url[:80]}...")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                    pct = downloaded / total_size * 100
                    logger.info(f"  Download progress: {downloaded / 1024**2:.0f} / "
                                f"{total_size / 1024**2:.0f} MB ({pct:.0f}%)")

        logger.info(f"Downloaded Umbra SLC: {output_path} ({output_path.stat().st_size / 1024**2:.0f} MB)")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Umbra SLC download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def extract_umbra_slc_burst(
    tiff_path: Path,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Extract complex SLC data from an Umbra GeoTIFF.

    Umbra SLC data is stored as complex-valued GeoTIFFs (unlike Sentinel-1's
    two-band I/Q format). The data is typically a single Spotlight acquisition
    covering a small area at very high resolution.

    Parameters
    ----------
    tiff_path : Path
        Path to the Umbra SLC GeoTIFF.
    output_dir : Path, optional
        Output directory for the extracted .npy burst.

    Returns
    -------
    list of Path
        Paths to extracted burst files (.npy).
    """
    if output_dir is None:
        output_dir = SLC_BURSTS_DIR / "umbra"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for reading SLC TIFFs")

    with rasterio.open(str(tiff_path)) as src:
        num_bands = src.count
        height = src.height
        width = src.width

        logger.info(f"Umbra SLC: {height}x{width}, bands={num_bands}, "
                     f"dtype={src.dtypes[0]}")

        if num_bands >= 2:
            # Two-band I/Q format
            band_i = src.read(1).astype(np.float32)
            band_q = src.read(2).astype(np.float32)
            slc_complex = band_i + 1j * band_q
        else:
            raw = src.read(1)
            if np.iscomplexobj(raw):
                slc_complex = raw.astype(np.complex64)
            else:
                # Single-band real - reinterpret as interleaved I/Q
                raw_flat = raw.flatten().astype(np.float32)
                if len(raw_flat) % 2 == 0:
                    slc_complex = raw_flat[0::2] + 1j * raw_flat[1::2]
                    new_width = width // 2
                    slc_complex = slc_complex.reshape(height, new_width)
                else:
                    slc_complex = raw.astype(np.complex64)

    # Umbra Spotlight data is a single continuous acquisition (no bursts),
    # so we save the entire image as one "burst"
    burst_file = output_dir / f"{tiff_path.stem}_slc.npy"
    np.save(burst_file, slc_complex)
    logger.info(f"Extracted Umbra SLC: shape={slc_complex.shape}, saved to {burst_file.name}")

    return [burst_file]


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


def run_acquisition_preflight(
    lat: float,
    lon: float,
    buffer_deg: float = 0.05,
    lookback_days: int = 30,
    max_results: int = 1,
    flight_direction: Optional[str] = None,
    search_timeout_seconds: Optional[int] = None,
    search_max_retries: Optional[int] = None,
    search_retry_backoff_seconds: Optional[float] = None,
    env_path: Optional[Path] = None,
    load_dotenv: bool = True,
) -> Dict[str, Any]:
    """Safely validate Earthdata auth and a tiny ASF/CMR query without downloads."""
    if load_dotenv:
        load_env_file(env_path)

    try:
        import asf_search as asf
    except ImportError as exc:
        return {
            "status": "failed",
            "stage": "import_asf_search",
            "error_classification": "missing_dependency",
            "error": safe_exception_summary(exc),
        }

    auth_info = resolve_earthdata_auth()
    search_config = get_asf_search_config(
        timeout_seconds=search_timeout_seconds,
        max_retries=search_max_retries,
        retry_backoff_seconds=search_retry_backoff_seconds,
    )
    result: Dict[str, Any] = {
        "status": "failed",
        "stage": "init",
        "auth_mode": auth_info.get("mode"),
        "auth_status": describe_earthdata_auth(auth_info),
        "query": {
            "lat": float(lat),
            "lon": float(lon),
            "buffer_deg": float(buffer_deg),
            "lookback_days": int(lookback_days),
            "max_results": int(max_results),
            "flight_direction": flight_direction,
        },
        "search_config": search_config,
        "products_found": 0,
        "sample_products": [],
    }

    try:
        result["stage"] = "auth"
        session = create_earthdata_asf_session(asf, auth_info)
        result["stage"] = "search"
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_date = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        bbox = _build_search_bbox(lat=lat, lon=lon, buffer_deg=buffer_deg)
        products = search_sentinel1_slc(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_results=max(1, int(max_results)),
            flight_direction=flight_direction,
            search_timeout_seconds=search_config["timeout_seconds"],
            search_max_retries=search_config["max_retries"],
            search_retry_backoff_seconds=search_config["retry_backoff_seconds"],
            asf_session=session,
        )
        result.update({
            "status": "success",
            "stage": "complete",
            "products_found": len(products),
            "sample_products": [
                {
                    "granule_name": product.get("granule_name"),
                    "start_time": product.get("start_time"),
                    "flight_direction": product.get("flight_direction"),
                    "relative_orbit": product.get("relative_orbit"),
                    "size_mb": product.get("size_mb"),
                }
                for product in products[: max(1, int(max_results))]
            ],
        })
    except Exception as exc:
        result["error_classification"] = classify_asf_acquisition_error(exc)
        result["error"] = safe_exception_summary(exc)
        logger.error(
            "Acquisition preflight failed at stage=%s; classification=%s; error=%s",
            result.get("stage"),
            result.get("error_classification"),
            result.get("error"),
        )

    return result


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
    search_parser.add_argument("--timeout", type=int, default=None, help="ASF/CMR timeout seconds")
    search_parser.add_argument("--retries", type=int, default=None, help="ASF/CMR search attempts")
    search_parser.add_argument("--retry-backoff", type=float, default=None, help="Search retry backoff seconds")
    search_parser.add_argument(
        "--direction", choices=["ASCENDING", "DESCENDING"], default=None
    )
    search_parser.add_argument("--commercial", action="store_true")

    # Preflight command
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate Earthdata auth plus one tiny ASF/CMR query without downloading data",
    )
    preflight_parser.add_argument("--lat", type=float, default=38.3512, help="Target latitude")
    preflight_parser.add_argument("--lon", type=float, default=-121.986, help="Target longitude")
    preflight_parser.add_argument("--buffer", type=float, default=0.05, help="Search buffer degrees")
    preflight_parser.add_argument("--lookback-days", type=int, default=30)
    preflight_parser.add_argument("--max-results", type=int, default=1)
    preflight_parser.add_argument("--timeout", type=int, default=None, help="ASF/CMR timeout seconds")
    preflight_parser.add_argument("--retries", type=int, default=None, help="ASF/CMR search attempts")
    preflight_parser.add_argument("--retry-backoff", type=float, default=None, help="Search retry backoff seconds")
    preflight_parser.add_argument(
        "--direction", choices=["ASCENDING", "DESCENDING"], default=None
    )
    preflight_parser.add_argument("--env-path", type=Path, default=None)
    preflight_parser.add_argument("--no-dotenv", action="store_true", help="Do not load local .env")

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
        load_env_file()
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
            search_timeout_seconds=args.timeout,
            search_max_retries=args.retries,
            search_retry_backoff_seconds=args.retry_backoff,
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

    elif args.command == "preflight":
        result = run_acquisition_preflight(
            lat=args.lat,
            lon=args.lon,
            buffer_deg=args.buffer,
            lookback_days=args.lookback_days,
            max_results=args.max_results,
            flight_direction=args.direction,
            search_timeout_seconds=args.timeout,
            search_max_retries=args.retries,
            search_retry_backoff_seconds=args.retry_backoff,
            env_path=args.env_path,
            load_dotenv=not args.no_dotenv,
        )
        print(dumps_strict_json(result, indent=2))
        raise SystemExit(0 if result.get("status") == "success" else 2)

    elif args.command == "download":
        load_env_file()
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
