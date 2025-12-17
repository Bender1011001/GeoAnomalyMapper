import os
import re
import argparse
import requests
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# LiCSAR Public Data Access URL
BASE_URL = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/"

def get_track_id(frame_id):
    """Extracts track ID (first number) from frame ID (e.g., '087D_05354_131313' -> '87')."""
    match = re.match(r"^(\d{3})", frame_id)
    if match:
        return str(int(match.group(1))) # Remove leading zeros for URL construction
    return None

def list_interferograms(frame_id):
    """Scrapes the HTTP directory to find available interferogram dates."""
    track = get_track_id(frame_id)
    if not track:
        logger.error(f"Invalid Frame ID format: {frame_id}")
        return []

    # Construct URL: base / track / frame / interferograms /
    # NOTE: The server structure seems to be track/frame/interferograms/
    # But sometimes it might be different.
    # Let's verify if the frame directory exists first.
    
    # Try standard path
    frame_url = f"{BASE_URL}{track}/{frame_id}/"
    logger.info(f"Checking frame URL: {frame_url}")
    
    try:
        # Check if frame exists
        r = requests.get(frame_url, timeout=60)
        if r.status_code == 404:
             logger.warning(f"Frame {frame_id} not found at {frame_url}. Trying alternative structure...")
             return []
        r.raise_for_status()
        
        # Now check interferograms
        # Sometimes it's just 'interferograms', sometimes it might be different or flat?
        # Let's list the frame directory first to see what's inside
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a')
        
        interferograms_dir = None
        for link in links:
            href = link.get('href').strip('/')
            # Check for common variations or just look for anything that looks like a directory
            if href == 'interferograms':
                interferograms_dir = 'interferograms'
                break
            elif href == 'products':
                interferograms_dir = 'products' # Sometimes it's here?
                break
        
        if not interferograms_dir:
             logger.warning(f"Could not find 'interferograms' directory in {frame_url}")
             # Let's try to list files in the frame folder itself just in case
             target_url = frame_url
        else:
             target_url = f"{frame_url}{interferograms_dir}/"

        logger.info(f"Scanning: {target_url}")
        
        response = requests.get(target_url, timeout=60)
        if response.status_code == 404:
             logger.error(f"Interferograms directory not found at {target_url}")
             return []
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        
        # Regex to match date folders (YYYYMMDD_YYYYMMDD)
        date_pattern = re.compile(r"(\d{8}_\d{8})")
        
        valid_dates = []
        for link in links:
            href = link.get('href').strip('/')
            if date_pattern.match(href):
                valid_dates.append(href)
        
        return list(set(valid_dates)) # Remove duplicates
        
    except Exception as e:
        logger.error(f"Failed to list files for {frame_id}: {e}")
        return []

def download_coherence(frame_id, output_dir, limit=None):
    """Downloads coherence maps for a specific frame."""
    track = get_track_id(frame_id)
    
    # Get list of date pairs
    dates = list_interferograms(frame_id)
    if not dates:
        logger.warning(f"No data found for {frame_id}")
        return

    # Sort to get most recent first
    dates.sort(reverse=True)
    if limit:
        dates = dates[:limit]

    logger.info(f"Found {len(dates)} interferograms. Starting download...")

    # Ensure output directory exists (Uses env var or arg)
    frame_dir = output_dir / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    
    for date_pair in dates:
        # Construct File URL
        # Format: .../interferograms/DATE_DATE/DATE_DATE.geo.cc.tif
        file_name = f"{date_pair}.geo.cc.tif"
        
        # Try standard path first
        file_url_flat = f"{BASE_URL}{track}/{frame_id}/interferograms/{file_name}"
        file_url_nested = f"{BASE_URL}{track}/{frame_id}/interferograms/{date_pair}/{file_name}"
        
        save_path = frame_dir / file_name

        if save_path.exists():
            logger.info(f"Skipping (exists): {file_name}")
            success_count += 1
            continue

        logger.info(f"Downloading: {file_name}")
        
        try:
            # Try nested first (standard LiCSAR)
            with requests.get(file_url_nested, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    # Try flat structure
                    with requests.get(file_url_flat, stream=True, timeout=120) as r2:
                        if r2.status_code == 404:
                             # Try without 'interferograms' if that failed (root/date_pair/file)
                             file_url_alt = f"{BASE_URL}{track}/{frame_id}/{date_pair}/{file_name}"
                             with requests.get(file_url_alt, stream=True, timeout=120) as r3:
                                 if r3.status_code == 404:
                                     logger.warning(f"File not found: {file_url_nested}, {file_url_flat}, or {file_url_alt}")
                                     continue
                                 r3.raise_for_status()
                                 with open(save_path, 'wb') as f:
                                     for chunk in r3.iter_content(chunk_size=8192):
                                         f.write(chunk)
                        else:
                             r2.raise_for_status()
                             with open(save_path, 'wb') as f:
                                 for chunk in r2.iter_content(chunk_size=8192):
                                     f.write(chunk)
                else:
                    r.raise_for_status()
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")

    logger.info(f"Frame {frame_id}: Downloaded {success_count}/{len(dates)} files.")

def main():
    parser = argparse.ArgumentParser(description="LiCSAR Automated Downloader")
    parser.add_argument("--frames", nargs='+', required=True, help="List of Frame IDs (e.g. 087D_05354_131313)")
    parser.add_argument("--output", default=None, help="Output directory override")
    parser.add_argument("--limit", type=int, default=5, help="Max files per frame (0 for all)")
    
    args = parser.parse_args()

    # Determine Output Directory (Priority: Arg > Env Var > Default)
    if args.output:
        base_path = Path(args.output)
    elif os.environ.get("GEOANOMALYMAPPER_DATA_DIR"):
        base_path = Path(os.environ["GEOANOMALYMAPPER_DATA_DIR"]) / "raw" / "insar"
    else:
        base_path = Path("data/raw/insar")

    logger.info(f"Saving data to: {base_path}")

    for frame in args.frames:
        download_coherence(frame, base_path, limit=args.limit if args.limit > 0 else None)

if __name__ == "__main__":
    main()