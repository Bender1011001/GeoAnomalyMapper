#!/usr/bin/env python3
"""
analyze_targets.py - Deep dive analysis and mapping of targets
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import sys
import os
import numpy as np

# Try importing sklearn, fallback if missing
try:
    from sklearn.cluster import DBSCAN
    SKLEARN = True
except ImportError:
    SKLEARN = False

def analyze_and_map(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    print(f"Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # normalize columns
    df.columns = df.columns.str.lower()
    
    # 1. Clustering (Define "Districts")
    if SKLEARN and len(df) > 0:
        print("Clustering targets into districts...")
        # DBSCAN: Epsilon 10km (~0.1 deg), Min samples 3
        coords = df[['latitude', 'longitude']].values
        # simple approx: 1 deg ~ 111km. 10km ~ 0.09 deg.
        epsilon = 0.09 
        db = DBSCAN(eps=epsilon, min_samples=3).fit(coords)
        df['cluster'] = db.labels_
    else:
        df['cluster'] = -1
    
    # 2. Interactive Map (Folium)
    print("Generating interactive map...")
    
    # Center map on US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, 
                   tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                   attr='Esri')
    
    # Add labels
    folium.TileLayer('cartodbpositron').add_to(m)
    
    # Group into feature group
    fg = folium.FeatureGroup(name="Targets")
    
    for idx, row in df.iterrows():
        # Color based on probability (if avail) or random
        prob = row.get('probability', 0.5)
        color = 'red' if prob > 0.9 else 'orange' if prob > 0.8 else 'blue'
        
        popup_txt = f"""
        <b>Target ID:</b> {row.get('target_id', idx)}<br>
        <b>Prob:</b> {prob:.2f}<br>
        <b>Cluster:</b> {row.get('cluster', 'N/A')}<br>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=popup_txt,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(fg)
        
    fg.add_to(m)
    folium.LayerControl().add_to(m)
    
    map_path = 'data/outputs/target_map.html'
    m.save(map_path)
    print(f"Map saved to {map_path}")
    
    # 3. District Summary
    if 'cluster' in df.columns:
        # Ignore noise (-1)
        districts = df[df['cluster'] != -1].groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'probability': 'mean',
            'target_id': 'count'
        }).rename(columns={'target_id': 'target_count'})
        
        districts = districts.sort_values('target_count', ascending=False)
        
        print("\nTop Potential Mining Districts:")
        print(districts.head(10))
        
        districts.to_csv('data/outputs/district_summary.csv')

if __name__ == "__main__":
    # Prefer the verified geography file as it has the 737 targets
    input_file = 'data/outputs/targets_geography_verified.csv'
    analyze_and_map(input_file)
