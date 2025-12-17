import pandas as pd

def debug_coords():
    targets = pd.read_csv('data/outputs/usa_targets.csv')
    
    with open('coords_full_debug.txt', 'w') as f:
        f.write("--- LATITUDE STATS ---\n")
        f.write(str(targets['Latitude'].describe()) + "\n")
        
        f.write("\n--- LONGITUDE STATS ---\n")
        f.write(str(targets['Longitude'].describe()) + "\n")
        
        f.write("\n--- SAMPLE ROWS ---\n")
        f.write(targets[['Latitude', 'Longitude']].head(10).to_string() + "\n")
        
    print("Debug written to coords_full_debug.txt")

if __name__ == "__main__":
    debug_coords()
