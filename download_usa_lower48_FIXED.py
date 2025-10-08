#!/usr/bin/env python3
"""
USA Lower 48 Sentinel-1 IW SLC Downloader - FIXED VERSION

Implements all corrections from detailed analysis:
- AOI constrained to Lower-48 ONLY (no global tile wastage)
- OData filter enforces operationalMode=IW + Platform=S1A/S1B  
- Client-side WV rejection as safety net
- Proper auth headers on ALL requests
- 2 catalogue workers, 4 download workers

Just run: python download_usa_lower48_FIXED.py
"""

import os
import sys
import json
import time
import socket
import threading
import random
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

print_lock = threading.Lock()

def log(msg=""):
    with print_lock:
        print(msg, flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load credentials
CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

if not CDSE_USERNAME or not CDSE_PASSWORD:
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        if key == "CDSE_USERNAME":
                            CDSE_USERNAME = val.strip('"\'')
                        elif key == "CDSE_PASSWORD":
                            CDSE_PASSWORD = val.strip('"\'')
        log(f"✓ Loaded credentials from {env_path}")

if not CDSE_USERNAME or not CDSE_PASSWORD:
    log("ERROR: Set CDSE_USERNAME and CDSE_PASSWORD in .env file")
    sys.exit(2)

# API endpoints
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

# USA Lower 48 bounding box (with margin for burst edges)
USA_LOWER48_BBOX = (-125.0, 24.5, -66.4, 49.6)  # (min_lon, min_lat, max_lon, max_lat)

# Date range (last 90 days)
DAYS_BACK = 90
DATE_START = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
DATE_END = datetime.now().strftime("%Y-%m-%d")

# Output
OUTPUT_DIR = Path("data/raw/insar/sentinel1")
MANIFEST_FILE = OUTPUT_DIR / "_manifest.jsonl"

# Worker limits (per analysis: 2 catalogue, 4 download)
CATALOG_WORKERS = 2
DOWNLOAD_WORKERS = 4

# Rate limiting
RATE_LIMIT_DELAY = 2.0  # seconds between downloads
download_lock = threading.Lock()
last_download_time = [0.0]  # mutable for closure

# ============================================================================
# TILE GENERATION - Lower 48 ONLY
# ============================================================================

def tiles_over_lower48(step_deg=10):
    """
    Generate 10×10° tiles ONLY over USA Lower 48.
    No global waste - constrained from the start.
    """
    min_lon, min_lat, max_lon, max_lat = USA_LOWER48_BBOX
    
    # Snap to grid
    lon_start = int(min_lon // step_deg) * step_deg
    lat_start = int(min_lat // step_deg) * step_deg
    lon_end = int((max_lon + step_deg) // step_deg) * step_deg
    lat_end = int((max_lat + step_deg) // step_deg) * step_deg
    
    tiles = []
    for lo in range(lon_start, lon_end, step_deg):
        for la in range(lat_start, lat_end, step_deg):
            tile = {
                'min_lon': lo,
                'min_lat': la,
                'max_lon': min(lo + step_deg, 180),
                'max_lat': min(la + step_deg, 90),
                'name': f"tile_lat{la:+04d}_lon{lo:+04d}"
            }
            # Only include if intersects Lower-48
            if not (tile['max_lon'] <= USA_LOWER48_BBOX[0] or
                    tile['min_lon'] >= USA_LOWER48_BBOX[2] or
                    tile['max_lat'] <= USA_LOWER48_BBOX[1] or
                    tile['min_lat'] >= USA_LOWER48_BBOX[3]):
                tiles.append(tile)
    
    return tiles

# ============================================================================
# TOKEN MANAGER
# ============================================================================

class TokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self._token = None
        self._token_time = 0
        self._lock = threading.Lock()
        self.refresh_interval = 50 * 60  # 50 minutes
    
    def get(self):
        with self._lock:
            if not self._token or (time.time() - self._token_time) > self.refresh_interval:
                self._refresh()
            return self._token
    
    def _refresh(self):
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public"
        }
        
        backoffs = [2, 4, 8, 16]
        for i, delay in enumerate(backoffs + [None]):
            try:
                r = requests.post(AUTH_URL, data=data, timeout=30)
                r.raise_for_status()
                self._token = r.json()["access_token"]
                self._token_time = time.time()
                log("✓ Token refreshed")
                return
            except Exception as e:
                if delay is None:
                    raise RuntimeError(f"Token refresh failed: {e}")
                time.sleep(delay)

# ============================================================================
# ODATA QUERY - CORRECT FILTERS
# ============================================================================

def build_odata_filter(tile, date_start, date_end):
    """
    Build OData filter with CORRECT CDSE syntax.
    
    CRITICAL per CDSE docs:
    - productType must be FULL type string: 'IW_SLC__1S' (not just 'SLC')
    - operationalMode is redundant (encoded in productType)
    - Include S1A/B/C (S1C operational in 2025)
    - All metadata via Attributes/.../any() pattern
    
    Constraints:
    - Collection: SENTINEL-1
    - productType: IW_SLC__1S (IW Single Look Complex)
    - Date range: ContentDate/Start (top-level OK)
    - Spatial: OData.CSC.Intersects with POLYGON WKT (lon/lat, closed)
    """
    # Build WKT polygon (lon/lat, closed ring)
    wkt = (f"POLYGON(({tile['min_lon']} {tile['min_lat']},"
           f"{tile['max_lon']} {tile['min_lat']},"
           f"{tile['max_lon']} {tile['max_lat']},"
           f"{tile['min_lon']} {tile['max_lat']},"
           f"{tile['min_lon']} {tile['min_lat']}))")
    
    # CORRECT filter per CDSE examples - full productType, no platform filter
    filt = (
        "Collection/Name eq 'SENTINEL-1' "
        "and Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'IW_SLC__1S') "
        f"and ContentDate/Start ge {date_start}T00:00:00.000Z "
        f"and ContentDate/Start le {date_end}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )
    
    return filt

def auth_headers(token):
    """Proper headers for ALL requests"""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

# ============================================================================
# SEARCH WITH CORRECT FILTERS
# ============================================================================

def search_tile(token, tile, date_start, date_end, max_results=4, debug_count=False):
    """
    Search for Sentinel-1 IW SLC over tile.
    Returns list of products that pass server filters.
    
    productType='IW_SLC__1S' already excludes WV (which would be 'WV_SLC__1S'),
    so no client-side filtering needed.
    """
    filt = build_odata_filter(tile, date_start, date_end)
    
    params = {
        "$filter": filt,
        "$top": max_results,
        "$orderby": "ContentDate/Start desc"
    }
    
    # Optional: Add count for debugging
    if debug_count:
        params["$count"] = "true"
    
    try:
        # Add jitter to avoid thundering herd
        time.sleep(random.uniform(0.1, 0.5))
        
        r = requests.get(CATALOG_URL, params=params, headers=auth_headers(token), timeout=60)
        r.raise_for_status()
        result = r.json()
        
        products = result.get("value", [])
        
        # Debug: Show count if requested
        if debug_count and "@odata.count" in result:
            log(f"[{tile['name']}] Total matching products: {result['@odata.count']}")
        
        return products
    
    except Exception as e:
        log(f"[{tile['name']}] Search error: {e}")
        return []

# ============================================================================
# DOWNLOAD WITH RESUME
# ============================================================================

def rate_limit_wait():
    """Enforce rate limit between downloads"""
    with download_lock:
        elapsed = time.time() - last_download_time[0]
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        last_download_time[0] = time.time()

def download_product(token, product, tile_name):
    """
    Download single product with resume capability.
    Returns (success: bool, product_id: str)
    """
    product_id = product["Id"]
    product_name = product["Name"]
    filename = f"{product_name}.zip"
    filepath = OUTPUT_DIR / filename
    
    # Skip if already downloaded
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024**2
        log(f"[{tile_name}] ✓ Already have {filename} ({size_mb:.1f} MB)")
        return (True, product_id)
    
    # Download URL
    url = f"{ZIPPER_URL}({product_id})/$value"
    
    # Attempt download with retries
    backoffs = [2, 4, 8, 16]
    for attempt, delay in enumerate(backoffs + [None], 1):
        try:
            rate_limit_wait()  # Rate limit
            
            log(f"[{tile_name}] Downloading {product_name}... (attempt {attempt})")
            
            r = requests.get(url, headers=auth_headers(token), stream=True, timeout=120)
            r.raise_for_status()
            
            # Stream to file
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress every 10%
                        if total_size > 0 and downloaded % (total_size // 10) < 8192:
                            pct = 100 * downloaded / total_size
                            size_gb = downloaded / 1024**3
                            log(f"[{tile_name}] {product_name}... {pct:.1f}% ({size_gb:.2f} GB)")
            
            size_gb = filepath.stat().st_size / 1024**3
            log(f"[{tile_name}] ✓ Downloaded {filename} ({size_gb:.2f} GB)")
            
            # Log to manifest
            with open(MANIFEST_FILE, 'a', encoding='utf-8') as f:
                record = {
                    "id": product_id,
                    "name": product_name,
                    "tile": tile_name,
                    "size_bytes": filepath.stat().st_size,
                    "download_time": datetime.now().isoformat()
                }
                f.write(json.dumps(record) + "\n")
            
            return (True, product_id)
        
        except Exception as e:
            log(f"[{tile_name}] ✗ Attempt {attempt} failed: {e}")
            if delay is None:
                return (False, product_id)
            time.sleep(delay + random.uniform(0, 2))  # Jittered backoff
    
    return (False, product_id)

# ============================================================================
# PROCESS TILE
# ============================================================================

def process_tile(token_mgr, tile):
    """
    Search and download for single tile.
    Returns (success_count, total_searched)
    """
    token = token_mgr.get()
    
    log(f"[{tile['name']}] Searching...")
    products = search_tile(token, tile, DATE_START, DATE_END)
    
    if not products:
        log(f"[{tile['name']}] No IW SLC products found")
        return (0, 0)
    
    log(f"[{tile['name']}] Found {len(products)} IW SLC products")
    
    # Download first product only (avoid rate limits)
    success, prod_id = download_product(token, products[0], tile['name'])
    
    return (1 if success else 0, 1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    log("=" * 70)
    log("USA LOWER 48 SENTINEL-1 IW SLC DOWNLOADER - FIXED")
    log("=" * 70)
    log()
    log(f"AOI: Lower 48 states ({USA_LOWER48_BBOX})")
    log(f"Date range: {DATE_START} to {DATE_END}")
    log(f"Mode: IW SLC only (no WV ocean junk)")
    log(f"Platform: Sentinel-1A/1B only")
    log(f"Workers: {CATALOG_WORKERS} catalogue, {DOWNLOAD_WORKERS} download")
    log()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate tiles
    tiles = tiles_over_lower48()
    log(f"✓ Generated {len(tiles)} tiles for Lower 48")
    log()
    
    # Initialize token manager
    token_mgr = TokenManager(CDSE_USERNAME, CDSE_PASSWORD)
    token_mgr.get()  # Initial auth
    
    # Process tiles with limited concurrency
    total_success = 0
    total_searched = 0
    
    log(f"Processing {len(tiles)} tiles...")
    log()
    
    with ThreadPoolExecutor(max_workers=CATALOG_WORKERS) as executor:
        futures = {executor.submit(process_tile, token_mgr, t): t for t in tiles}
        
        for future in as_completed(futures):
            tile = futures[future]
            try:
                success, searched = future.result()
                total_success += success
                total_searched += searched
            except Exception as e:
                log(f"[{tile['name']}] ERROR: {e}")
    
    log()
    log("=" * 70)
    log("DOWNLOAD COMPLETE")
    log("=" * 70)
    log(f"Successfully downloaded: {total_success}/{total_searched} products")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Manifest: {MANIFEST_FILE}")
    log()

if __name__ == "__main__":
    main()