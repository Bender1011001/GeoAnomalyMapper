import pandas as pd
import os

def main():
    cache_path = 'data/usgs_mrds_full.csv'
    if not os.path.exists(cache_path):
        print(f"Cache file {cache_path} not found.")
        return

    print("Loading full dataset...")
    df = pd.read_csv(cache_path, low_memory=False)
    
    print(f"Total rows: {len(df)}")
    
    if 'country' in df.columns:
        print("\nTop 20 Countries:")
        print(df['country'].value_counts().head(20))
        
        # Check filtered count for 'United States'
        us_count = len(df[df['country'] == 'United States'])
        print(f"\nRows with country='United States': {us_count}")
        
    else:
        print("Column 'country' not found.")
        
    # Analyze deposit types for US producers
    if 'country' in df.columns and 'dev_stat' in df.columns and 'dep_type' in df.columns:
        df['dev_stat'] = df['dev_stat'].astype(str).str.title()
        
        us_producers = df[
            (df['country'] == 'United States') & 
            (df['dev_stat'].isin(['Producer', 'Past Producer']))
        ]
        
        print(f"\nUS Producers Count: {len(us_producers)}")
        print("\nTop 50 Deposit Types for US Producers:")
        print(us_producers['dep_type'].value_counts().head(50))

if __name__ == "__main__":
    main()
