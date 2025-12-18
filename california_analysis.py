import pandas as pd
import numpy as np

# Vacaville, CA Coordinates
VACAVILLE_LAT = 38.3566
VACAVILLE_LON = -121.9877

def analyze_california_targets():
    print("Loading targets...")
    try:
        targets = pd.read_csv('data/outputs/usa_targets.csv')
    except FileNotFoundError:
        print("Error: data/outputs/usa_targets.csv not found.")
        return

    # Filter for California (approximate bounding box)
    # Latitude: 32.5 to 42.0
    # Longitude: -124.5 to -114.0
    ca_targets = targets[
        (targets['Latitude'] >= 32.5) & (targets['Latitude'] <= 42.0) &
        (targets['Longitude'] >= -124.5) & (targets['Longitude'] <= -114.0)
    ].copy()

    print(f"Found {len(ca_targets)} targets in the California region.")

    if len(ca_targets) == 0:
        print("No targets found in California.")
        return

    # Calculate distance to Vacaville (Haversine-like approximation)
    # 1 deg lat ~= 111 km
    # 1 deg lon ~= 111 * cos(lat) km
    
    avg_lat_rad = np.radians(ca_targets['Latitude'])
    lon_scale = np.cos(avg_lat_rad)
    
    d_lat = (ca_targets['Latitude'] - VACAVILLE_LAT) * 111
    d_lon = (ca_targets['Longitude'] - VACAVILLE_LON) * 111 * lon_scale
    
    ca_targets['Distance_km'] = np.sqrt(d_lat**2 + d_lon**2)

    # 1. Top Targets by Density Contrast
    print("\n--- Top 5 Targets in CA by Density Contrast ---")
    top_density = ca_targets.nlargest(5, 'Density_Contrast')
    for _, row in top_density.iterrows():
        print(f"Lat: {row['Latitude']:.4f}, Lon: {row['Longitude']:.4f}, "
              f"Density: {row['Density_Contrast']:.4f}, Dist to Vacaville: {row['Distance_km']:.1f} km")

    # 2. Closest Targets to Vacaville
    print("\n--- Closest Targets to Vacaville, CA ---")
    closest = ca_targets.nsmallest(5, 'Distance_km')
    for _, row in closest.iterrows():
        print(f"Lat: {row['Latitude']:.4f}, Lon: {row['Longitude']:.4f}, "
              f"Density: {row['Density_Contrast']:.4f}, Dist to Vacaville: {row['Distance_km']:.1f} km")

if __name__ == "__main__":
    analyze_california_targets()
