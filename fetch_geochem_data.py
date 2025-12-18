
import os
import requests
import zipfile
import io
from pathlib import Path

# Metadata
# USGS NURE Sediment Data (Open-File Report 97-492)
# Direct download link usually: https://pubs.usgs.gov/of/1997/ofr-97-0492/data/nure_sed.zip
# Or MRDATA API: https://mrdata.usgs.gov/nure/sediment/nure_sed.zip
# Let's try the MRDATA reliable link first.

# Correct URL found from webpage
URL_NURE = "https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip"

OUTPUT_DIR = Path("data/validation")
ZIP_PATH = OUTPUT_DIR / "nuresed-csv.zip"

def fetch_data():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    print(f"Downloading NURE Sediment Data from {URL_NURE}...")
    try:
        r = requests.get(URL_NURE, stream=True)
        print(f"Status Code: {r.status_code}")
        r.raise_for_status()
        
        with open(ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download Complete. Size: {ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        print("Extraction Complete.")
        print(f"Files: {os.listdir(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_data()
