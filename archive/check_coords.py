import pandas as pd

def check_coords():
    print("Checking Coordinate Ranges...")
    try:
        deposits = pd.read_csv('data/usgs_goldilocks.csv')
        targets = pd.read_csv('data/outputs/usa_targets.csv')
    except Exception as e:
        print(e)
        return

    # DEPOSITS
    d_lat = 'latitude' if 'latitude' in deposits.columns else 'lat'
    d_lon = 'longitude' if 'longitude' in deposits.columns else 'lon'
    
    print("\n--- KNOWN DEPOSITS ---")
    print(deposits[[d_lat, d_lon]].describe())

    # TARGETS
    print("\n--- PREDICTED TARGETS ---")
    print(targets[['Latitude', 'Longitude']].describe())

    # Check for Lat/Lon Swap
    # US Lat: 24 to 49
    # US Lon: -125 to -66
    
    t_lat_mean = targets['Latitude'].mean()
    t_lon_mean = targets['Longitude'].mean()
    
    print(f"\nTarget Mean Lat: {t_lat_mean}")
    print(f"Target Mean Lon: {t_lon_mean}")
    
    if t_lat_mean < 0 and t_lon_mean > 0:
        print("ALERT: Coordinates appear swapped! (Lat is negative, Lon is positive?)")
    elif t_lat_mean < -60:
         print("ALERT: Lat looks like Longitude!")

if __name__ == "__main__":
    check_coords()
