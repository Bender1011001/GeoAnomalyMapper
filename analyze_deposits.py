import pandas as pd
from pathlib import Path

def main():
    csv_path = Path('data/usgs_goldilocks.csv')
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check for country column
    possible_country_cols = [c for c in df.columns if 'country' in c.lower() or 'state' in c.lower()]
    print(f"Possible country/location columns: {possible_country_cols}")
    
    if 'type' in df.columns:
        print("\nTop 50 Deposit Types:")
        print(df['type'].value_counts().head(50))
    else:
        print("Column 'type' not found.")

if __name__ == "__main__":
    main()
