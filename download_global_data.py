import os
import requests
from tqdm import tqdm

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR: Download incomplete")
        return False
    return True

def main():
    target_dir = r"D:\GeoAnomalyData"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    datasets = [
        # GGMplus is large, skip if exists and size is > 4GB
        {
            "name": "GGMplus Gravity Disturbance (4.2GB)",
            "url": "https://ddfe.blazejbucha.com/models/GGMplus/data/dg.zip",
            "filename": "ggmplus_dg.zip",
            "min_size": 3 * 1024 * 1024 * 1024
        },
        {
            "name": "EMAG2v3 Magnetics (~500MB)",
            "url": "https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530/EMAG2_V3_20170530_UpCont.tif",
            "filename": "emag2_v3_upcont.tif",
            "min_size": 100 * 1024 * 1024
        },
        {
            "name": "USGS Major Mineral Deposits",
            "url": "https://mrdata.usgs.gov/major-deposits/ofr20051294-csv.zip",
            "filename": "usgs_major_deposits.zip",
            "min_size": 100 * 1024
        },
        {
            "name": "GLiM Global Lithology (~300MB)",
            "url": "https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1",
            "filename": "glim_lithology.zip",
            "min_size": 50 * 1024 * 1024
        }
    ]

    for ds in datasets:
        dest = os.path.join(target_dir, ds["filename"])
        
        # Check if file exists and has reasonable size
        if os.path.exists(dest):
            size = os.path.getsize(dest)
            min_size = ds.get("min_size", 1000) # Default 1KB min to filter out 404s
            if size > min_size:
                print(f"Skipping {ds['name']}, already exists and looks valid (Size: {size / 1024 / 1024:.2f} MB).")
                continue
            else:
                print(f"File {ds['filename']} exists but is too small ({size} bytes). Redownloading...")
                os.remove(dest)
            
        success = download_file(ds["url"], dest)
        if success:
            print(f"Successfully downloaded {ds['name']}")
        else:
            print(f"Failed to download {ds['name']}")

if __name__ == "__main__":
    main()
