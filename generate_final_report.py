"""
Generate Final Commercial Target Package.

Selecting high-confidence targets from V3 Model (Residual Gravity).
Criteria:
1. Commodity Match (Gold, Copper, Nickel, Lithium, Cobalt, PGE)
2. Geologic Favorability (Mass Deficit for Gold/Epithermal, Mass Excess for VMS/IOCG)
3. Strength of Anomaly (Top 10% by density contrast)
"""

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load V3 results
high = pd.read_csv('data/outputs/usa_v3_dual_high.csv')
low = pd.read_csv('data/outputs/usa_v3_dual_low.csv')

# Load MRDS for cross-referencing
print("Loading MRDS database...")
mrds = pd.read_csv('validation_results/mrds_data/mrds.csv', low_memory=False)
mrds = mrds.dropna(subset=['latitude', 'longitude'])
# Clean commodity text
mrds['commod1'] = mrds['commod1'].fillna('Unknown').astype(str)

# Build Tree
print("Building spatial index...")
tree = cKDTree(mrds[['latitude', 'longitude']].values)

def enrich_targets(df):
    if len(df) == 0: return df
    dist, idx = tree.query(df[['Latitude', 'Longitude']].values, k=1)
    df['Dist_to_MRDS_km'] = dist * 111.0 # approx deg to km
    
    # Get commodity info
    nearest_commods = mrds.iloc[idx]['commod1'].values
    refined_commods = []
    
    # Simple mapping for display
    for c in nearest_commods:
        c_lower = c.lower()
        if 'gold' in c_lower: refined_commods.append('Gold')
        elif 'silver' in c_lower: refined_commods.append('Silver')
        elif 'copper' in c_lower: refined_commods.append('Copper')
        elif 'zinc' in c_lower: refined_commods.append('Zinc')
        elif 'nickel' in c_lower: refined_commods.append('Nickel')
        elif 'lithium' in c_lower: refined_commods.append('Lithium')
        elif 'iron' in c_lower: refined_commods.append('Iron')
        else: refined_commods.append(c.split(',')[0]) # Take first term
        
    df['Nearest_MRDS_Commodity'] = refined_commods
    return df

print("Enriching targets with MRDS data...")
high = enrich_targets(high)
low = enrich_targets(low)

print(f"Loaded and enriched {len(high)} HIGH targets and {len(low)} LOW targets.")

# Commercial Commodities of Interest
COMMODITIES = [
    'Gold', 'Silver', 'Copper', 'Zinc', 'Lead', 'Nickel', 
    'Lithium', 'Cobalt', 'Platinum', 'Palladium', 'Tungsten'
]

def format_target(row, type_):
    return f"| {row['Latitude']:.3f}, {row['Longitude']:.3f} | {row['Density_Contrast']:.1f} | {row['Nearest_MRDS_Commodity']} ({row['Dist_to_MRDS_km']:.1f} km) | {row['Deposit_Class']} |"

report = []
report.append("# 💎 GeoAnomalyMapper: Commercial Prospectivity Report")
report.append("## Executive Summary")
report.append("This report highlights the top mineral exploration targets identified by the **GeoAnomalyMapper V3 (Residual Gravity)** model.")
report.append("The model uses deep learning to identify density anomalies associated with mineral deposits, corrected for regional crustal thickness.")
report.append("")

# --- SECTION 1: EPITHERMAL / CARLIN-STYLE TARGETS (LOW DENSITY) ---
report.append("## 1. Mass-Deficit Targets (Potential Epithermal/Carlin Gold)")
report.append("> **Geophysics:** Low density anomalies often indicate hydrothermal alteration (silicification, decalcification) or intrusives.")
report.append("")
report.append("| Location (Lat, Lon) | Contrast (mGal) | Nearest Commodity | Class |")
report.append("|---|---|---|---|")

# Filter LOW targets
gold_targets = low[
    (low['Nearest_MRDS_Commodity'].isin(['Gold', 'Silver', 'Mercury', 'Antimony'])) &
    (low['Density_Contrast'] <= -2.0)  # Strong anomalies
].sort_values('Density_Contrast', ascending=True)

for i, row in gold_targets.head(10).iterrows():
    report.append(format_target(row, 'LOW'))

report.append("")
report.append(f"*Total Gold/Epithermal Candidates Found: {len(gold_targets)}*") 
report.append("")

# --- SECTION 2: MASSIVE SULFIDE / MAGMATIC TARGETS (HIGH DENSITY) ---
report.append("## 2. Mass-Excess Targets (Potential VMS/IOCG/Nickel)")
report.append("> **Geophysics:** High density anomalies often indicate massive sulfide bodies, iron-oxide copper gold (IOCG), or mafic intrusions.")
report.append("")
report.append("| Location (Lat, Lon) | Contrast (mGal) | Nearest Commodity | Class |")
report.append("|---|---|---|---|")

# Filter HIGH targets
base_targets = high[
    (high['Nearest_MRDS_Commodity'].isin(['Copper', 'Zinc', 'Nickel', 'Cobalt', 'Iron'])) &
    (high['Density_Contrast'] >= 2.0)
].sort_values('Density_Contrast', ascending=False)

for i, row in base_targets.head(10).iterrows():
    report.append(format_target(row, 'HIGH'))

report.append("")
report.append(f"*Total Base Metal Candidates Found: {len(base_targets)}*")
report.append("")

# --- SECTION 3: STRATEGIC UNDISCOVERED TARGETS ---
report.append("## 3. Top Undiscovered Prospects (Wildcats)")
report.append("These targets have strong signal but are >10km from any known deposit.")
report.append("")
report.append("| Location | Type | Contrast | Region |")
report.append("|---|---|---|---|")

# Combine and filter
all_t = pd.concat([high, low])
wildcats = all_t[all_t['Dist_to_MRDS_km'] > 10.0].copy()
wildcats['Abs_Contrast'] = wildcats['Density_Contrast'].abs()
wildcats = wildcats.sort_values('Abs_Contrast', ascending=False).head(10)

for i, row in wildcats.iterrows():
    type_ = "Mass Excess" if row['Density_Contrast'] > 0 else "Mass Deficit"
    report.append(f"| {row['Latitude']:.3f}, {row['Longitude']:.3f} | {type_} | {row['Density_Contrast']:.1f} | {row['Region_ID']} |")

# Check for empty results
if len(wildcats) == 0:
    report.append("*(No strong wildcat targets found >10km from known deposits)*")

# Save report
with open('FINAL_TARGET_REPORT.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

print("Report generated: FINAL_TARGET_REPORT.md")
