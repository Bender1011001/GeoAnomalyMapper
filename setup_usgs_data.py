#!/usr/bin/env python3
import logging
from pathlib import Path
from utils.data_fetcher import fetch_usgs_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing USGS Goldilocks Dataset Setup...")
    
    try:
        # Default cache location
        cache_path = 'data/usgs_mrds_full.csv'
        
        df = fetch_usgs_training_data(local_cache_path=cache_path)
        
        output_path = Path('data/usgs_goldilocks.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Successfully saved {len(df)} sites to {output_path}")
            
            # Print preview
            print("\nDataset Preview:")
            print(df.head())
            print("\nCommodity Distribution:")
            print(df['commod1'].value_counts().head())
            print("\nDevelopment Status:")
            print(df['dev_stat'].value_counts())
        else:
            logger.warning("❌ No data fetched. Check your connection or filters.")
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")

if __name__ == "__main__":
    main()
