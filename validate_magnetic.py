import pandas as pd
from scipy.spatial import cKDTree

# Load
c = pd.read_csv('data/outputs/metallic_anomalies/metallic_candidates.csv')
d = pd.read_csv('data/known_dubs.csv')

# Filter
usa = d[(d.lat >= 25) & (d.lat <= 49) & (d.lon >= -124) & (d.lon <= -67)]

# Tree
tree = cKDTree(c[['Latitude','Longitude']].values)

print(f"Validating {len(usa)} DUBs against {len(c)} Metallic Anomalies...")
hits = 0
for i, row in usa.iterrows():
    dist, idx = tree.query([row['lat'], row['lon']])
    dist_km = dist * 111
    
    status = "MISS"
    if dist_km < 30: # Use slightly larger radius for regional magnetic data
        status = "NEAR"
        if dist_km < 15:
            status = "HIT"
            hits += 1
            
    print(f"  {status:5s} {row['name'][:30]:30s} {dist_km:6.1f} km")

print(f"\nMagnetic Recovery Rate: {hits}/{len(usa)} ({100*hits/len(usa):.1f}%)")
