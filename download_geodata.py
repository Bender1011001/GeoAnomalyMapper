"""
GeoAnomalyMapper - MAXIMUM RESOLUTION Global Data Downloader
Fixed: Rate limiting, USA/EU priority, better error handling
"""

import os
import sys
import json
import time
import socket
import threading
import builtins
import requests
from datetime import datetime, timedelta, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.exceptions import NameResolutionError
from requests.exceptions import ConnectionError

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # Fallback to default encoding

print_lock = threading.Lock()

def log(message: str = "") -> None:
    """Thread-safe print helper."""
    with print_lock:
        builtins.print(message, flush=True)

def log_tile(tile_name: str, message: str) -> None:
    """Tile-scoped logging helper."""
    log(f"[{tile_name}] {message}")

# ==============================
# Config
# ==============================

COPERNICUS_USERNAME = os.getenv("CDSE_USERNAME")
COPERNICUS_PASSWORD = os.getenv("CDSE_PASSWORD")

if not COPERNICUS_USERNAME or not COPERNICUS_PASSWORD:
    # Try loading from .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    if key == "CDSE_USERNAME":
                        COPERNICUS_USERNAME = val.strip('"\'')
                    elif key == "CDSE_PASSWORD":
                        COPERNICUS_PASSWORD = val.strip('"\'')
        print(f"✓ Loaded credentials from {env_path}")

if not COPERNICUS_USERNAME or not COPERNICUS_PASSWORD:
    print("ERROR: Set CDSE_USERNAME and CDSE_PASSWORD in environment or .env file")
    sys.exit(2)

# CRITICAL: Reduce workers to avoid 429 rate limiting
DEFAULT_WORKERS = 2  # Changed from 4-6 to 2 (Copernicus rate limits aggressively)
DOWNLOAD_DELAY = 3   # Seconds between downloads to avoid 429

# Date window
SHORT_WINDOW_DAYS = 90  # Expanded from 28 to get more coverage
DATE_START = (datetime.now() - timedelta(days=SHORT_WINDOW_DAYS)).strftime("%Y-%m-%d")
DATE_END = datetime.now().strftime("%Y-%m-%d")

# Prioritization: Lower 48 by default (most common use case)
TARGET_PRESET = os.getenv("TARGET_PRESET", "USA_LOWER48").upper()

# Output
S1_OUT_DIR = "data/insar/sentinel1"
MANIFEST_PATH = os.path.join(S1_OUT_DIR, "_manifest.jsonl")

# Rate limiting lock
download_lock = threading.Lock()
last_download_time = 0

# ==============================
# Build global grid with USA/EU priority
# ==============================

GLOBAL_TILES = []
print("Generating global tile grid...")
for lat in range(-90, 90, 10):
    for lon in range(-180, 180, 10):
        GLOBAL_TILES.append({
            'min_lat': lat,
            'max_lat': min(lat + 10, 90),
            'min_lon': lon,
            'max_lon': min(lon + 10, 180),
            'name': f"tile_lat{lat}_lon{lon}"
        })
print(f"✓ Created {len(GLOBAL_TILES)} tiles for global coverage")

# AOI definitions [min_lon, min_lat, max_lon, max_lat]
AOI_USA_LOWER48 = [
    [-125.0, 24.5, -66.95, 49.5],   # Contiguous United States (Lower 48)
]

AOI_USA_FULL = [
    [-125.0, 24.5, -66.95, 49.5],   # CONUS (Lower 48)
    [-170.0, 52.0, -130.0, 72.0],   # Alaska
    [-161.0, 18.0, -154.0, 23.0],   # Hawaii
    [-67.5, 17.5, -65.2, 18.7],     # Puerto Rico
]

AOI_EU = [
    [-11.0, 35.0, 40.0, 71.0],      # Europe (EGMS overlap)
]

PRESETS = {
    "USA_LOWER48": AOI_USA_LOWER48,           # Contiguous US only (most common)
    "USA": AOI_USA_FULL,                      # All US territories
    "EUROPE": AOI_EU,
    "USA_PLUS_EU": AOI_USA_FULL + AOI_EU,
    "GLOBAL": []
}

# ==============================
# Directories
# ==============================

def create_directories():
    for d in [
        'data/gravity/global',
        'data/magnetic/global',
        S1_OUT_DIR,
        'data/insar/egms',
        'data/seismic/global',
        'data/regional/usgs',
        'data/topography'
    ]:
        os.makedirs(d, exist_ok=True)
    print("✓ Directory structure created\n")

# ==============================
# DNS and Network Helpers
# ==============================

def ensure_dns(host: str, timeout_sec: int = 60) -> None:
    """Block until host resolves, else raise. Prevents token fetch failures."""
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            socket.getaddrinfo(host, 443)
            return
        except socket.gaierror as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"DNS resolution failed for {host}: {last_err}")

# ==============================
# Token Manager with auto-refresh and robust error handling
# ==============================

class TokenManager:
    DEFAULT_AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    def __init__(self, username, password, refresh_interval_sec=50*60):
        self.username = username
        self.password = password
        self.refresh_interval_sec = refresh_interval_sec
        self._token = None
        self._token_time = 0.0
        self._lock = threading.Lock()
        # Allow override if CDSE_AUTH_URL is set
        self.auth_url = os.getenv("CDSE_AUTH_URL", self.DEFAULT_AUTH_URL)

    def get(self):
        with self._lock:
            now = time.time()
            if not self._token or (now - self._token_time) > self.refresh_interval_sec:
                self._refresh_locked()
            return self._token

    def force_refresh(self):
        with self._lock:
            self._refresh_locked()
            return self._token

    def _refresh_locked(self):
        # Preflight DNS to give a clear error fast
        ensure_dns(host=self._host_of(self.auth_url), timeout_sec=60)

        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public"
        }

        # Retry loop for transient network errors
        backoffs = [2, 4, 8, 16]
        last_err = None
        for i, delay in enumerate(backoffs + [None]):  # final try with no sleep after
            try:
                r = requests.post(self.auth_url, data=data, timeout=30)
                r.raise_for_status()
                self._token = r.json()["access_token"]
                self._token_time = time.time()
                print("✓ Token refreshed")
                return
            except (ConnectionError, NameResolutionError, requests.Timeout) as e:
                last_err = e
                msg = f"Auth connect attempt {i+1} failed: {e}"
                print(f"✗ {msg}")
                if delay is None:
                    break
                time.sleep(delay)
            except Exception as e:
                # Non-retryable HTTP errors (401, 403, etc.) bubble up immediately
                print(f"✗ Token refresh failed: {e}")
                raise

        raise RuntimeError(f"Token refresh failed after retries: {last_err}")

    @staticmethod
    def _host_of(url: str) -> str:
        """Extract hostname from URL"""
        return url.split("//", 1)[-1].split("/", 1)[0]

# ==============================
# Utility functions
# ==============================

def bboxes_intersect(tile, bbox):
    """Check if tile intersects with bounding box"""
    return not (
        tile['max_lon'] <= bbox[0] or
        tile['min_lon'] >= bbox[2] or
        tile['max_lat'] <= bbox[1] or
        tile['min_lat'] >= bbox[3]
    )

def prioritized_tiles(sample_tiles):
    """Return tiles prioritizing USA/EU, then fill with global coverage"""
    aois = PRESETS.get(TARGET_PRESET, [])
    
    selected = []
    if aois:
        # Get all tiles that intersect with AOIs
        for t in GLOBAL_TILES:
            if any(bboxes_intersect(t, b) for b in aois):
                selected.append(t)
        selected = sorted(selected, key=lambda x: (x['min_lat'], x['min_lon']))
    
    # Fill remaining with evenly spaced global tiles
    if len(selected) < sample_tiles:
        remaining = [t for t in GLOBAL_TILES if t not in selected]
        step = max(1, len(remaining) // max(1, (sample_tiles - len(selected))))
        selected += remaining[::step][:(sample_tiles - len(selected))]
    
    return selected[:sample_tiles]

def _write_manifest(entry: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_attr(product: dict, name: str):
    """Extract attribute from Copernicus product"""
    attrs = product.get("Attributes", [])
    for a in attrs:
        if a.get("Name") == name:
            return (a.get("Value") or a.get("StringValue") or 
                    a.get("DoubleValue") or a.get("DateTimeValue"))
    return None

def rate_limit_wait():
    """Enforce minimum delay between downloads"""
    global last_download_time
    with download_lock:
        now = time.time()
        elapsed = now - last_download_time
        if elapsed < DOWNLOAD_DELAY:
            time.sleep(DOWNLOAD_DELAY - elapsed)
        last_download_time = time.time()

# ==============================
# Gravity/Mag/EGMS stubs
# ==============================

def download_all_gravity_models():
    print("=" * 80)
    print("GRAVITY MODELS - MANUAL DOWNLOAD REQUIRED")
    print("=" * 80)
    with open('data/gravity/global/GRAVITY_DOWNLOAD_GUIDE.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIGH-RESOLUTION GLOBAL GRAVITY DATA\n")
        f.write("=" * 80 + "\n\n")
        f.write("BEST SOURCE: ICGEM Calculation Service (generates custom grids)\n")
        f.write("URL: http://icgem.gfz-potsdam.de/calcgrid\n\n")
        f.write("Steps:\n")
        f.write("1. Select Model: XGM2019e_2159 (highest resolution)\n")
        f.write("2. Choose: 'Gravity disturbance' or 'Height anomalies'\n")
        f.write("3. Grid step: 2 arc-minutes (0.0333°) = ~4km resolution\n")
        f.write("4. Latitude: -90 to 90, Longitude: -180 to 180\n")
        f.write("5. Output: GeoTIFF or ASCII grid\n")
        f.write("6. Save to: data/gravity/global/XGM2019e_grid.tif\n\n")
        f.write("Alternative: WGM2012 from BGI\n")
        f.write("https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/\n")
    print("✓ Created comprehensive gravity download guide")
    print("  RECOMMENDED: Use ICGEM grid calculator for automated generation")

def download_all_magnetic_data():
    print("\n" + "=" * 80)
    print("MAGNETIC DATA - ATTEMPTING DOWNLOADS")
    print("=" * 80)
    
    # Try multiple mirror URLs
    datasets = [
        {
            "name": "EMAG2v3 GeoTIFF",
            "urls": [
                "https://www.ngdc.noaa.gov/geomag/EMag2/EMAG2_V3_20170530.tif",
                "https://www.ngdc.noaa.gov/mgg/global/EMAG2_V3_20170530.tif"
            ],
            "output": "data/magnetic/global/EMAG2_V3.tif"
        },
        {
            "name": "WDMAM (preview)",
            "urls": [
                "https://www.wdmam.org/WDMAM2_V2_MERC.tif",
                "https://geomag.colorado.edu/wdmam/WDMAM2_V2_MERC.tif"
            ],
            "output": "data/magnetic/global/WDMAM.tif"
        }
    ]
    
    for dataset in datasets:
        print(f"\n{dataset['name']}")
        
        if os.path.exists(dataset['output']):
            size = os.path.getsize(dataset['output']) / (1024**2)
            print(f"✓ Already exists ({size:.1f} MB)")
            continue
        
        success = False
        for url in dataset['urls']:
            try:
                print(f"  Trying: {url}")
                r = requests.get(url, timeout=60, stream=True)
                r.raise_for_status()
                
                os.makedirs(os.path.dirname(dataset['output']), exist_ok=True)
                with open(dataset['output'], 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
                
                # Validate file size (must be >1 KB, otherwise it's likely an error page)
                size_bytes = os.path.getsize(dataset['output'])
                if size_bytes < 1024:
                    os.remove(dataset['output'])
                    raise RuntimeError(f"Downloaded file is empty ({size_bytes} bytes) - source likely moved or unavailable")
                
                size_mb = size_bytes / (1024**2)
                print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
                success = True
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        if not success:
            print(f"  ⚠ Manual download required - check NOAA/WDMAM websites")

def download_egms_insar():
    print("\n" + "=" * 80)
    print("EGMS INSAR - EUROPE (100m, pre-processed)")
    print("=" * 80)
    with open('data/insar/egms/EGMS_ACCESS_INFO.txt', 'w') as f:
        f.write("EGMS Access Information\n")
        f.write("=" * 60 + "\n\n")
        f.write("Portal: https://egms.land.copernicus.eu/\n")
        f.write("Resolution: 100 meters\n")
        f.write("Coverage: All of Europe\n")
        f.write("Data: Pre-processed, ready to use\n\n")
        f.write("WMS: https://image.discomap.eea.europa.eu/arcgis/services/GioLandPublic/EGMS/MapServer/WMSServer\n")
    print("✓ EGMS access info saved")

def download_regional_gravity():
    print("\n" + "=" * 80)
    print("REGIONAL GRAVITY - MANUAL DOWNLOAD")
    print("=" * 80)
    print("[1/3] USGS Gravity (North America)")
    print("Manual: https://mrdata.usgs.gov/gravity/")
    print("\n[2/3] BGS Gravity (United Kingdom)")
    print("Manual: https://www.bgs.ac.uk/datasets/gb-gravity/")
    print("\n[3/3] Geoscience Australia Gravity")
    print("Manual: https://ecat.ga.gov.au/")

def download_global_topography():
    print("\n" + "=" * 80)
    print("TOPOGRAPHY - MANUAL DOWNLOAD")
    print("=" * 80)
    with open('data/topography/TOPOGRAPHY_INFO.txt', 'w') as f:
        f.write("Global High-Resolution Topography\n")
        f.write("=" * 60 + "\n\n")
        f.write("SRTM 1 arc-second (~30m)\n")
        f.write("  https://earthexplorer.usgs.gov/\n\n")
        f.write("ASTER GDEM (~30m)\n")
        f.write("  https://search.earthdata.nasa.gov/\n\n")
        f.write("Copernicus DEM 30m\n")
        f.write("  https://spacedata.copernicus.eu/\n")
    print("✓ Topography access info saved")

# ==============================
# Sentinel-1 search & download
# ==============================

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

def search_sentinel1_tile(token, tile, date_start, date_end, max_results=4):
    """Search for Sentinel-1 IW SLC data with debug output"""
    wkt = (f"POLYGON(({tile['min_lon']} {tile['min_lat']},"
           f"{tile['max_lon']} {tile['min_lat']},"
           f"{tile['max_lon']} {tile['max_lat']},"
           f"{tile['min_lon']} {tile['max_lat']},"
           f"{tile['min_lon']} {tile['min_lat']}))")
    
    # Simplified filter - try without IW mode restriction first
    filter_str = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'SLC') and "
        f"ContentDate/Start ge {date_start}T00:00:00.000Z and "
        f"ContentDate/Start le {date_end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )
    
    params = {
        "$filter": filter_str,
        "$top": max_results,
        "$orderby": "ContentDate/Start desc"
    }
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        r = requests.get(CATALOG_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        results = r.json().get("value", [])
        
        # Debug: print search details if no results
        if not results:
            print(f"  DEBUG: No products in {date_start} to {date_end}")
            print(f"  Trying broader search...")
            # Try without date restriction to see if ANY products exist
            broad_filter = (
                "Collection/Name eq 'SENTINEL-1' and "
                "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'SLC') and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
            )
            broad_params = {"$filter": broad_filter, "$top": 1}
            r2 = requests.get(CATALOG_URL, params=broad_params, headers=headers, timeout=60)
            r2.raise_for_status()
            any_products = r2.json().get("value", [])
            if any_products:
                oldest_date = any_products[0].get("ContentDate", {}).get("Start", "unknown")
                print(f"  Found products exist (oldest: {oldest_date[:10]}), but not in date window")
            else:
                print(f"  No Sentinel-1 SLC products found for this tile at all")
        
        return results
    except Exception as e:
        print(f"  Search error: {e}")
        return []

def pick_best_product(products):
    """Pick one ASC or DES product (not both to reduce 429 errors)"""
    if not products:
        return None
    
    # Prefer ASCENDING if available
    for p in products:
        if (get_attr(p, "orbitDirection") or "").upper() == "ASCENDING":
            return p
    
    # Otherwise take first available
    return products[0]

def download_sentinel1_product(token_manager, product, output_dir, tile_name):
    """Download with rate limiting and retry logic"""
    product_id = product["Id"]
    product_name = product["Name"]
    
    tile_dir = os.path.join(output_dir, tile_name)
    os.makedirs(tile_dir, exist_ok=True)
    
    final_path = os.path.join(tile_dir, f"{product_name}.zip")
    part_path = final_path + ".part"
    
    # Skip if already downloaded
    if os.path.exists(final_path):
        size_gb = os.path.getsize(final_path) / (1024**3)
        print(f"  ✓ Already downloaded ({size_gb:.2f} GB)")
        return True
    
    url = ZIPPER_URL.format(product_id=product_id)
    
    # Enforce rate limiting
    rate_limit_wait()
    
    for attempt in range(4):
        try:
            resume_from = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            headers = {"Authorization": f"Bearer {token_manager.get()}"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            
            log_tile(tile_name, f"Downloading {product_name[:50]}... (attempt {attempt+1})")
            
            with requests.get(url, headers=headers, stream=True, timeout=3600) as r:
                if r.status_code == 401:
                    token_manager.force_refresh()
                    continue
                if r.status_code == 429:
                    wait_time = 2 ** (attempt + 3)  # 8, 16, 32, 64 seconds
                    log_tile(tile_name, f"⚠ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                
                r.raise_for_status()
                
                total_size = r.headers.get("content-length")
                if total_size:
                    total_size = int(total_size) + resume_from
                
                mode = "ab" if resume_from > 0 else "wb"
                downloaded = resume_from
                last_logged_pct = -5.0
                last_logged_mb = downloaded / (1024**2)
                
                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            pct = (downloaded / total_size) * 100
                            gb = downloaded / (1024**3)
                            if pct - last_logged_pct >= 5.0 or pct >= 99.9:
                                log_tile(
                                    tile_name,
                                    f"{product_name[:40]}... {pct:5.1f}% ({gb:5.2f} GB)"
                                )
                                last_logged_pct = pct
                        else:
                            mb = downloaded / (1024**2)
                            if mb - last_logged_mb >= 100:
                                log_tile(
                                    tile_name,
                                    f"{product_name[:40]}... {mb:6.0f} MB downloaded"
                                )
                                last_logged_mb = mb
                log_tile(tile_name, f"{product_name[:40]}... download finished, finalising")
            
            os.replace(part_path, final_path)
            size_gb = os.path.getsize(final_path) / (1024**3)
            log_tile(tile_name, f"Downloaded {product_name[:40]} ({size_gb:.2f} GB)")
            
            _write_manifest({
                "ts": datetime.now(UTC).isoformat(),
                "tile": tile_name,
                "product_id": product_id,
                "product_name": product_name,
                "path": final_path,
                "size_gb": round(size_gb, 3),
                "status": "ok"
            })
            return True
            
        except Exception as e:
            log_tile(tile_name, f"✗ Attempt {attempt+1} failed: {str(e)[:100]}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    
    return False

def process_tile(tile, token_manager, date_start, date_end, output_dir):
    """Process one tile"""
    log_tile(tile['name'], f"Tile bounds: {tile['min_lat']}..{tile['max_lat']}°N, {tile['min_lon']}..{tile['max_lon']}°E")
    
    products = search_sentinel1_tile(token_manager.get(), tile, date_start, date_end)
    
    if not products:
        log_tile(tile['name'], "No products found")
        _write_manifest({
            "ts": datetime.now(UTC).isoformat(),
            "tile": tile["name"],
            "status": "no_products"
        })
        return False
    
    log_tile(tile['name'], f"Found {len(products)} products")
    product = pick_best_product(products)
    
    if product:
        return download_sentinel1_product(token_manager, product, output_dir, tile["name"])
    
    return False

def download_sentinel1_parallel(token_manager, sample_tiles=20, workers=DEFAULT_WORKERS):
    """Download tiles in PARALLEL with rate limiting"""
    log("\n" + "=" * 80)
    log(f"SENTINEL-1 INSAR - {sample_tiles} tiles ({workers} parallel workers)")
    log(f"Preset: {TARGET_PRESET} | Window: last {SHORT_WINDOW_DAYS} days")
    log("=" * 80)
    
    tiles = prioritized_tiles(sample_tiles)
    log(f"Selected {len(tiles)} tiles (USA/EU prioritized)\n")
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tiles
        future_to_tile = {
            executor.submit(process_tile, tile, token_manager, DATE_START, DATE_END, S1_OUT_DIR): tile
            for tile in tiles
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_tile), 1):
            tile = future_to_tile[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
                log(f"Progress: {i}/{len(tiles)} processed ({successful} successful, {failed} no data)")
            except Exception as e:
                failed += 1
                print(f"✗ Tile {tile['name']} error: {e}")
    
    print(f"\n\n✓ Completed: {successful}/{len(tiles)} tiles downloaded, {failed} with no data")
    return successful

def download_sentinel1_sequential(token_manager, sample_tiles=20):
    """Download tiles SEQUENTIALLY (fallback if rate limiting is severe)"""
    print("\n" + "=" * 80)
    print(f"SENTINEL-1 INSAR - {sample_tiles} tiles (SEQUENTIAL mode)")
    print(f"Preset: {TARGET_PRESET} | Window: last {SHORT_WINDOW_DAYS} days")
    print("=" * 80)
    
    tiles = prioritized_tiles(sample_tiles)
    print(f"Selected {len(tiles)} tiles (USA/EU prioritized)\n")
    
    successful = 0
    for i, tile in enumerate(tiles, 1):
        print(f"\n[{i}/{len(tiles)}]", end=" ")
        if process_tile(tile, token_manager, DATE_START, DATE_END, S1_OUT_DIR):
            successful += 1
        
        # Brief pause between tiles
        if i < len(tiles):
            time.sleep(2)
    
    print(f"\n\n✓ Completed: {successful}/{len(tiles)} tiles downloaded")
    return successful

# ==============================
# Main
# ==============================

def main():
    print("=" * 80)
    print("MAXIMUM RESOLUTION GLOBAL ANOMALY MAP BUILDER")
    print("=" * 80)
    print(f"\n⚠ This run limits Sentinel-1 to the last {SHORT_WINDOW_DAYS} days for speed.")
    print("Press Ctrl+C at any time to stop")
    print("\nContinue? (y/n): ", end='')
    
    if input().lower() != 'y':
        print("Aborted")
        return
    
    create_directories()
    
    # Download supporting datasets
    download_all_gravity_models()
    download_all_magnetic_data()
    download_egms_insar()
    download_regional_gravity()
    download_global_topography()
    
    # Sentinel-1 download
    print("\n" + "=" * 80)
    print("Sentinel-1 InSAR (5–20 m resolution, parallel)")
    print("=" * 80)
    print("\nHow many global sample tiles to download tonight?")
    print("Rule of thumb: ~1 product per tile, 5–10 GB each (zip), auto-resume enabled.")
    
    num_tiles_str = input("\nEnter number of tiles (or 0 to skip): ").strip() or "0"
    
    try:
        num_tiles = int(num_tiles_str)
    except ValueError:
        print("Invalid input, skipping Sentinel-1")
        num_tiles = 0
    
    if num_tiles > 0:
        # Ask for worker count
        workers_str = input(f"\nParallel workers [1-10, default {DEFAULT_WORKERS}]: ").strip() or str(DEFAULT_WORKERS)
        try:
            workers = max(1, min(10, int(workers_str)))
        except ValueError:
            workers = DEFAULT_WORKERS
        
        if workers == 1:
            print(f"\n⚠ SEQUENTIAL mode selected (1 worker)")
            print(f"  Estimated time: {num_tiles * 15} minutes ({num_tiles} tiles × ~15 min each)")
        else:
            print(f"\n✓ PARALLEL mode: {workers} workers")
            print(f"  Estimated time: {num_tiles * 15 // workers} minutes (parallel speedup)")
            print(f"  Note: Built-in rate limiting prevents 429 errors")
        
        print("\nProceed? (y/n): ", end='')
        
        if input().lower() == 'y':
            # Preflight DNS checks for all CDSE hosts
            print("\n✓ Checking network connectivity...")
            for host in [
                "identity.dataspace.copernicus.eu",
                "catalogue.dataspace.copernicus.eu",
                "zipper.dataspace.copernicus.eu"
            ]:
                try:
                    ensure_dns(host, 30)
                    print(f"  ✓ {host} reachable")
                except Exception as e:
                    print(f"  ✗ Preflight DNS failed for {host}: {e}")
                    print("\n⚠️  Network connectivity issue detected. Please check:")
                    print("  - Internet connection is working")
                    print("  - DNS servers are responding")
                    print("  - No firewall blocking HTTPS (port 443)")
                    sys.exit(3)
            
            print("\n✓ Network OK, initializing token manager...")
            tm = TokenManager(COPERNICUS_USERNAME, COPERNICUS_PASSWORD)
            tm.force_refresh()
            
            # Use parallel or sequential based on worker count
            if workers == 1:
                download_sentinel1_sequential(tm, sample_tiles=num_tiles)
            else:
                download_sentinel1_parallel(tm, sample_tiles=num_tiles, workers=workers)
    
    # Summary
    print("\n" + "=" * 80)
    print("MAXIMUM RESOLUTION GLOBAL DATASET - COMPLETE")
    print("=" * 80)
    print(f"\nCheck the manifest for results:")
    print(f"  {MANIFEST_PATH}")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)