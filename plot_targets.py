import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import rasterio
from rasterio.plot import show

def create_maps(targets_csv, tif_path=None, output_dir='data/outputs'):
    """
    Creates HTML interactive map and Static PNG map.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading targets from {targets_csv}...")
    df = pd.read_csv(targets_csv)
    
    # Filter for valid lat/lon
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if df.empty:
        print("No targets to plot.")
        return

    # --- 1. Interactive Map (Folium) ---
    print("Generating Interactive Map...")
    mean_lat = df['Latitude'].mean()
    mean_lon = df['Longitude'].mean()
    
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6, tiles='OpenStreetMap')
    
    # Add TIF overlay if provided? (Hard with Folium without keeping local server or encoding)
    # Skipping TIF overlay on HTML for simplicity/speed.
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in df.iterrows():
        score = row.get('Confidence_Score', 0)
        
        # Color coding
        if score >= 75: color = 'green'
        elif score >= 50: color = 'orange'
        else: color = 'red'
        
        popup_html = f"""
        <b>Target ID:</b> {row.get('Region_ID', 'N/A')}<br>
        <b>Score:</b> {score}<br>
        <b>Commodity:</b> {row.get('Predicted_Commodity', 'Unknown')}<br>
        <b>Type:</b> {row.get('Predicted_Type', 'Unknown')}<br>
        <b>Status:</b> {row.get('Nearby_Mine_Status', 'N/A')}
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6 + (score/20), # size by score
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color
        ).add_to(marker_cluster)
        
    out_html = Path(output_dir) / "targets_map.html"
    m.save(out_html)
    print(f"Saved interactive map to {out_html}")

    # --- 2. Static Map (Matplotlib) ---
    print("Generating Static Map...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # If TIF provided, plot as base
    if tif_path and Path(tif_path).exists():
        with rasterio.open(tif_path) as src:
            show(src, ax=ax, cmap='gray', alpha=0.5)
            # Extent is handled by rasterio plot
    else:
        # Just set aspect roughly equal
        ax.set_aspect('equal')
        ax.grid(True)
        
    # Scatter plot
    # Color by score
    sc = ax.scatter(
        df['Longitude'], 
        df['Latitude'], 
        c=df['Confidence_Score'], 
        cmap='RdYlGn', 
        s=df['Confidence_Score']*2, 
        alpha=0.7, 
        edgecolors='black'
    )
    
    plt.colorbar(sc, label='Confidence Score')
    plt.title("High Value Exploration Targets")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    out_png = Path(output_dir) / "high_value_targets_map.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved static map to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", help="Path to targets CSV")
    parser.add_argument("--tif", help="Optional background GeoTIFF")
    args = parser.parse_args()
    
    create_maps(args.targets, args.tif)
