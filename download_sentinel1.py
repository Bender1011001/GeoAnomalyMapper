#!/usr/bin/env python3
"""
Sentinel-1 Data Downloader - Secure CLI Tool

Downloads Sentinel-1 SLC data from Copernicus Data Space for InSAR processing.
Uses secure credential storage (never hardcodes passwords!)

Setup:
    1. Set environment variables:
       export COPERNICUS_USER="your_email@example.com"
       export COPERNICUS_PASS="your_password"
    
    2. Or create .env file in GeoAnomalyMapper/ directory (gitignored):
       COPERNICUS_USER=your_email@example.com
       COPERNICUS_PASS=your_password

Usage:
    python download_sentinel1.py --region "-105.0,32.0,-104.0,33.0" --days 30

"""

import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Tuple
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "insar" / "sentinel1"
ENV_FILE = BASE_DIR / ".env"

# Copernicus Data Space API
API_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1"


# ============================================================================
# CREDENTIAL MANAGEMENT (SECURE)
# ============================================================================

def load_credentials() -> Tuple[str, str]:
    """
    Load credentials securely from environment or .env file.
    NEVER hardcodes credentials in code!
    """
    # Try environment variables first
    user = os.getenv('COPERNICUS_USER')
    password = os.getenv('COPERNICUS_PASS')
    
    if user and password:
        logger.info("Credentials loaded from environment variables")
        return user, password
    
    # Try .env file
    if ENV_FILE.exists():
        logger.info("Loading credentials from .env file")
        env_vars = {}
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        user = env_vars.get('COPERNICUS_USER')
        password = env_vars.get('COPERNICUS_PASS')
        
        if user and password:
            return user, password
    
    # Prompt user
    logger.warning("No credentials found in environment or .env file")
    logger.info("For security, please set credentials as environment variables:")
    logger.info("  export COPERNICUS_USER='your_email@example.com'")
    logger.info("  export COPERNICUS_PASS='your_password'")
    logger.info("\nOr create a .env file (see INSAR_DATA_GUIDE.md)")
    
    sys.exit(1)


def get_access_token(username: str, password: str) -> str:
    """Authenticate and get access token."""
    logger.info(f"Authenticating as {username}...")
    
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(AUTH_URL, data=data, timeout=30)
        response.raise_for_status()
        token = response.json()['access_token']
        logger.info("Authentication successful!")
        return token
    except requests.exceptions.RequestException as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)


# ============================================================================
# DATA SEARCH
# ============================================================================

def search_sentinel1(
    bounds: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    product_type: str = "SLC",
    orbit: str = None
) -> List[Dict]:
    """
    Search for Sentinel-1 products.
    
    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max)
        start_date: ISO format "YYYY-MM-DD"
        end_date: ISO format "YYYY-MM-DD"
        product_type: "SLC" or "GRD"
        orbit: "ASCENDING", "DESCENDING", or None (both)
    
    Returns:
        List of product metadata dicts
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # Build WKT polygon
    polygon = f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    
    # Build query
    filters = [
        f"Collection/Name eq 'SENTINEL-1'",
        f"contains(Name,'{product_type}')",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        f"ContentDate/Start gt {start_date}T00:00:00.000Z",
        f"ContentDate/Start lt {end_date}T23:59:59.999Z"
    ]
    
    if orbit:
        filters.append(f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/OData.CSC.StringAttribute/Value eq '{orbit}')")
    
    query = " and ".join(filters)
    url = f"{API_BASE}/Products?$filter={query}&$orderby=ContentDate/Start desc&$top=100"
    
    logger.info(f"Searching for Sentinel-1 {product_type} products...")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Region: {bounds}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        results = response.json()
        
        products = results.get('value', [])
        logger.info(f"Found {len(products)} products")
        
        return products
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Search failed: {e}")
        return []


def download_product(product_id: str, product_name: str, token: str, output_dir: Path):
    """Download a single Sentinel-1 product."""
    
    output_file = output_dir / f"{product_name}.zip"
    
    if output_file.exists():
        logger.info(f"Already downloaded: {product_name}")
        return True
    
    url = f"{DOWNLOAD_BASE}/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    
    logger.info(f"Downloading: {product_name} ({output_file.name})")
    logger.info("This may take several minutes (4-8 GB per product)...")
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        # Download with progress
        downloaded = 0
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded/(1024**3):.2f} GB / {total_size/(1024**3):.2f} GB)", end='', flush=True)
        
        print()  # New line
        logger.info(f"✓ Downloaded: {output_file}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        if output_file.exists():
            output_file.unlink()  # Remove partial download
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 data from Copernicus Data Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download data for Carlsbad Caverns area, last 30 days
    python download_sentinel1.py --region "-105.0,32.0,-104.0,33.0" --days 30
    
    # Specific date range
    python download_sentinel1.py --region "-105.0,32.0,-104.0,33.0" \\
        --start 2024-01-01 --end 2024-12-31
    
    # Only ascending orbit
    python download_sentinel1.py --region "-105.0,32.0,-104.0,33.0" \\
        --days 60 --orbit ASCENDING

Before running:
    Set credentials as environment variables or create .env file
    See INSAR_DATA_GUIDE.md for setup instructions
        """
    )
    
    parser.add_argument(
        '--region',
        type=str,
        required=False,
        help='Region: "lon_min,lat_min,lon_max,lat_max"'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days before today to search (default: 30)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD) - overrides --days'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--orbit',
        type=str,
        choices=['ASCENDING', 'DESCENDING'],
        help='Orbit direction (default: both)'
    )
    parser.add_argument(
        '--product-type',
        type=str,
        default='SLC',
        choices=['SLC', 'GRD'],
        help='Product type (default: SLC for InSAR)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum products to download (default: 10)'
    )
    parser.add_argument(
        '--setup-credentials',
        action='store_true',
        help='Interactive credential setup'
    )
    
    args = parser.parse_args()
    
    # Setup credentials interactively (doesn't need region)
    if args.setup_credentials:
        print("\n=== Copernicus Credential Setup ===")
        print("This will create a .env file to store your credentials securely.\n")
        
        email = input("Copernicus email: ").strip()
        password = input("Copernicus password: ").strip()
        
        with open(ENV_FILE, 'w') as f:
            f.write(f"# Copernicus Data Space Credentials\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"COPERNICUS_USER={email}\n")
            f.write(f"COPERNICUS_PASS={password}\n")
        
        # Set restrictive permissions (Unix/Linux/Mac)
        try:
            os.chmod(ENV_FILE, 0o600)
        except:
            pass
        
        print(f"\n✓ Credentials saved to: {ENV_FILE}")
        print("✓ File permissions set to read-only for owner")
        print("\nYou can now run download commands!\n")
        return
    
    # Require region for download operations
    if not args.region:
        logger.error("--region is required for download operations")
        parser.print_help()
        sys.exit(1)
    
    # Parse region
    try:
        bounds = tuple(map(float, args.region.split(',')))
        if len(bounds) != 4:
            raise ValueError
    except:
        logger.error("Invalid region format. Use: lon_min,lat_min,lon_max,lat_max")
        sys.exit(1)
    
    # Parse dates
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    if args.start:
        start_date = args.start
    else:
        start = datetime.now() - timedelta(days=args.days)
        start_date = start.strftime('%Y-%m-%d')
    
    # Load credentials
    username, password = load_credentials()
    
    # Authenticate
    token = get_access_token(username, password)
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Search for products
    products = search_sentinel1(bounds, start_date, end_date, args.product_type, args.orbit)
    
    if not products:
        logger.warning("No products found matching criteria")
        return
    
    # Show products
    print("\n" + "="*70)
    print(f"Found {len(products)} Sentinel-1 {args.product_type} products")
    print("="*70)
    
    for i, product in enumerate(products[:args.limit], 1):
        name = product['Name']
        date = product['ContentDate']['Start'][:10]
        orbit = "ASC" if "ASCENDING" in product.get('Attributes', {}) else "DESC"
        print(f"{i}. {name}")
        print(f"   Date: {date}  Orbit: {orbit}")
    
    print("="*70)
    print(f"\nWill download first {min(len(products), args.limit)} products")
    print(f"Estimated size: {min(len(products), args.limit) * 5:.0f} GB")
    print(f"Output directory: {DATA_DIR}")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        logger.info("Download cancelled")
        return
    
    # Download products
    print("\n" + "="*70)
    print("DOWNLOADING")
    print("="*70 + "\n")
    
    success_count = 0
    for i, product in enumerate(products[:args.limit], 1):
        product_id = product['Id']
        product_name = product['Name']
        
        print(f"\n[{i}/{min(len(products), args.limit)}] {product_name}")
        
        if download_product(product_id, product_name, token, DATA_DIR):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Successfully downloaded: {success_count} / {min(len(products), args.limit)}")
    print(f"Location: {DATA_DIR}")
    print("\nNext steps:")
    print("1. Extract .zip files")
    print("2. Process with SNAP, ISCE2, or MintPy (see INSAR_DATA_GUIDE.md)")
    print("3. Run detect_voids.py with InSAR data")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()