import os
import argparse
import pandas as pd
import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_global_prediction(prediction_tif, mrds_csv):
    logger.info(f"Validating {prediction_tif} against {mrds_csv}")
    
    # Load MRDS
    df = pd.read_csv(mrds_csv, low_memory=False)
    # Filter for contiguous USA roughly
    df = df[(df['latitude'] >= 24) & (df['latitude'] <= 50) & (df['longitude'] >= -125) & (df['longitude'] <= -66)]
    # Filter for relevant commods? Or all? Let's use all 'Producer' and 'Past Producer' sites as positive class.
    # Actually, keep it broad for "Mineral Systems"
    df = df[df['dev_stat'].isin(['Producer', 'Past Producer', 'Prospect'])]
    
    logger.info(f"Loaded {len(df)} MRDS sites in USA region.")
    
    # Load Prediction Raster
    with rasterio.open(prediction_tif) as src:
        band1 = src.read(1)
        transform = src.transform
        nodata = src.nodata
        
        # Sample raster at MRDS locations
        # rasterio.index(lon, lat) -> row, col
        rows, cols = src.index(df['longitude'].values, df['latitude'].values)
        
        # Handle out of bounds
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        rows = np.array(rows)[valid]
        cols = np.array(cols)[valid]
        
        deposit_vals = band1[rows, cols]
        
        # Collect Background Samples (Random)
        # Assuming most of the map is barren.
        n_bg = len(deposit_vals) * 10
        bg_rows = np.random.randint(0, src.height, n_bg)
        bg_cols = np.random.randint(0, src.width, n_bg)
        bg_vals = band1[bg_rows, bg_cols]
        
        # Filter NoData/NaN
        if nodata is not None:
            deposit_vals = deposit_vals[deposit_vals != nodata]
            bg_vals = bg_vals[bg_vals != nodata]
            
        deposit_vals = deposit_vals[~np.isnan(deposit_vals)]
        bg_vals = bg_vals[~np.isnan(bg_vals)]
        
        logger.info(f"Valid Samples: {len(deposit_vals)} Deposits, {len(bg_vals)} Background")
        
        # Metrics
        y_true = np.concatenate([np.ones_like(deposit_vals), np.zeros_like(bg_vals)])
        y_scores = np.concatenate([deposit_vals, bg_vals])
        
        auc = roc_auc_score(y_true, y_scores)
        
        # Sensitivity at 5% Area Flagged
        # Find threshold where 5% of background is flagged
        threshold_5pct = np.percentile(bg_vals, 95)
        n_deposits_flagged = np.sum(deposit_vals > threshold_5pct)
        sensitivity = n_deposits_flagged / len(deposit_vals)
        
        print(f"\n=== Validation Results ===")
        print(f"Global Model (200m) Results:")
        print(f"ROC AUC: {auc:.4f}")
        print(f"Sensitivity (at 5% Area): {sensitivity:.2%}")
        print(f"Deposit Mean: {deposit_vals.mean():.4f}")
        print(f"Background Mean: {bg_vals.mean():.4f}")
        print(f"==========================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", default="D:/Geo_data/usa_prediction_200m.tif")
    parser.add_argument("--mrds", default="data/usgs_mrds_full.csv")
    args = parser.parse_args()
    
    if os.path.exists(args.tif) and os.path.exists(args.mrds):
        validate_global_prediction(args.tif, args.mrds)
    else:
        print("Files not found.")
