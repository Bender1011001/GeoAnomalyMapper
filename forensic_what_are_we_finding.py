"""
FORENSIC ANALYSIS: What are we ACTUALLY finding?
=================================================
The paradox: We pass statistical tests but miss famous districts.
What's in our "hits" that inflates our numbers?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from collections import Counter

TARGETS_FILE = Path("data/outputs/usa_targets.csv")
MRDS_FILE = Path("data/usgs_mrds_full.csv")

def main():
    print("\n" + "="*60)
    print("   FORENSIC ANALYSIS: What are we finding?")
    print("="*60)
    
    # Load data
    targets = pd.read_csv(TARGETS_FILE)
    mrds = pd.read_csv(MRDS_FILE, low_memory=False)
    
    # Filter MRDS to USA
    mrds = mrds[(mrds['latitude'] >= 24.5) & (mrds['latitude'] <= 70) &
                (mrds['longitude'] >= -170) & (mrds['longitude'] <= -66)]
    
    print(f"Targets: {len(targets)}")
    print(f"MRDS in USA: {len(mrds)}")
    
    # Get coordinates
    lat_col = 'Latitude' if 'Latitude' in targets.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in targets.columns else 'lon'
    target_coords = targets[[lat_col, lon_col]].values
    
    # Build MRDS index
    mrds_coords = mrds[['latitude', 'longitude']].dropna()
    mrds_clean = mrds.loc[mrds_coords.index].copy()
    mrds_tree = cKDTree(mrds_coords.values)
    
    # Find what each target is near
    threshold_deg = 0.45  # 50km
    distances, indices = mrds_tree.query(target_coords, k=1)
    
    hits_mask = distances < threshold_deg
    hit_indices = indices[hits_mask]
    
    matched = mrds_clean.iloc[hit_indices].copy()
    
    print(f"\nTargets with MRDS hit (50km): {len(matched)}")
    
    # ANALYSIS 1: Commodity breakdown
    print("\n" + "="*60)
    print("ANALYSIS 1: What commodities are we hitting?")
    print("="*60)
    
    commod_counts = matched['commod1'].value_counts()
    print("\nTop 20 Commodities:")
    total_hits = len(matched)
    cumulative = 0
    for i, (commod, count) in enumerate(commod_counts.head(20).items()):
        pct = count / total_hits * 100
        cumulative += pct
        print(f"  {i+1:2d}. {commod}: {count} ({pct:.1f}%) [cumulative: {cumulative:.0f}%]")
    
    # ANALYSIS 2: Development status breakdown
    print("\n" + "="*60)
    print("ANALYSIS 2: What development status?")
    print("="*60)
    
    status_counts = matched['dev_stat'].value_counts()
    print("\nDevelopment Status:")
    for status, count in status_counts.items():
        pct = count / total_hits * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    
    # ANALYSIS 3: Cross-tab commodity x status
    print("\n" + "="*60)
    print("ANALYSIS 3: Valuable commodities that are Producers?")
    print("="*60)
    
    valuable_keywords = ['Gold', 'Silver', 'Copper', 'Zinc', 'Lead', 'Nickel', 
                         'Uranium', 'Tungsten', 'Molybdenum', 'Lithium', 'Platinum']
    junk_keywords = ['Sand', 'Gravel', 'Stone', 'Clay', 'Aggregate', 'Crushed',
                     'Cement', 'Limestone', 'Dimension', 'Fill', 'Geothermal']
    
    def categorize(row):
        commod = str(row.get('commod1', ''))
        status = str(row.get('dev_stat', ''))
        
        is_valuable = any(v.lower() in commod.lower() for v in valuable_keywords)
        is_junk = any(j.lower() in commod.lower() for j in junk_keywords)
        is_producer = 'producer' in status.lower()
        
        if is_valuable and is_producer:
            return "VALUABLE_PRODUCER"
        elif is_valuable:
            return "VALUABLE_OTHER"
        elif is_junk and is_producer:
            return "JUNK_PRODUCER"
        elif is_junk:
            return "JUNK_OTHER"
        else:
            return "UNKNOWN"
    
    matched['category'] = matched.apply(categorize, axis=1)
    cat_counts = matched['category'].value_counts()
    
    print("\nTarget Categories:")
    for cat, count in cat_counts.items():
        pct = count / total_hits * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # ANALYSIS 4: Geographic clustering
    print("\n" + "="*60)
    print("ANALYSIS 4: Where are our targets?")
    print("="*60)
    
    # State-level (rough approximation)
    def get_region(lat, lon):
        if lat > 60:
            return "Alaska"
        elif lon < -120:
            return "West Coast (CA/OR/WA)"
        elif lon < -109:
            return "Mountain West (NV/AZ/UT/CO/MT/WY/ID)"
        elif lon < -100:
            return "Great Plains"
        elif lon < -90:
            return "Midwest"
        elif lon < -80:
            return "Southeast"
        else:
            return "Northeast"
    
    regions = [get_region(lat, lon) for lat, lon in target_coords]
    region_counts = Counter(regions)
    
    print("\nTarget Distribution by Region:")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        pct = count / len(targets) * 100
        print(f"  {region}: {count} ({pct:.1f}%)")
    
    # ANALYSIS 5: The smoking gun - what are the "hits" that inflate our stats?
    print("\n" + "="*60)
    print("ANALYSIS 5: THE SMOKING GUN")
    print("="*60)
    
    # What percentage of our "hits" are actually valuable?
    valuable_producers = matched[matched['category'] == 'VALUABLE_PRODUCER']
    junk_hits = matched[matched['category'].isin(['JUNK_PRODUCER', 'JUNK_OTHER'])]
    
    print(f"\nOf {len(matched)} total 'hits':")
    print(f"  Valuable Producers: {len(valuable_producers)} ({len(valuable_producers)/len(matched)*100:.1f}%)")
    print(f"  Junk (sand/gravel/etc): {len(junk_hits)} ({len(junk_hits)/len(matched)*100:.1f}%)")
    
    # The inflation problem
    print("\n" + "-" * 50)
    print("THE INFLATION PROBLEM:")
    print("-" * 50)
    print(f"""
Our statistical tests passed because:
1. MRDS has {len(mrds):,} deposits in USA
2. {len(junk_hits)} of our {len(matched)} hits ({len(junk_hits)/len(matched)*100:.0f}%) are JUNK
3. Sand/gravel pits are EVERYWHERE - any random point hits them
4. We're not finding ORE DEPOSITS, we're finding GEOLOGY

TRUE VALUABLE HIT RATE: {len(valuable_producers)}/{len(targets)} = {len(valuable_producers)/len(targets)*100:.1f}%
""")
    
    # FINAL VERDICT
    print("\n" + "="*60)
    print("   FINAL DIAGNOSIS")
    print("="*60)
    
    true_rate = len(valuable_producers) / len(targets) * 100
    
    if true_rate > 30:
        print(f"\nModel finds {true_rate:.0f}% valuable producers - ACCEPTABLE")
    elif true_rate > 15:
        print(f"\nModel finds {true_rate:.0f}% valuable producers - MARGINAL")
    else:
        print(f"\nModel finds {true_rate:.0f}% valuable producers - POOR")
        print("\nThe model is detecting GEOLOGICAL FEATURES, not ORE BODIES.")
        print("It's a rock type detector, not a mine finder.")

if __name__ == "__main__":
    main()
