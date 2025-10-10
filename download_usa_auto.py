#!/usr/bin/env python3
"""
Automatic USA Lower 48 Data Downloader

Just run this and walk away. It will:
- Download Sentinel-1 for entire Lower 48
- Auto-resume if network fails
- Retry with exponential backoff
- Download in parallel (4 workers)
- Save progress continuousl

NO INTERACTION NEEDED - Just run: python download_usa_auto.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("AUTOMATIC USA LOWER 48 DATA DOWNLOAD")
    print("=" * 70)
    print()
    print("This will download Sentinel-1 InSAR data for the entire Lower 48.")
    print("Coverage: ~50 tiles, ~250-500 GB total")
    print()
    print("Features:")
    print("  - Auto-resume on network failures")
    print("  - Parallel downloads (4 workers)")
    print("  - Built-in retry logic")
    print("  - Save progress continuously")
    print()
    print("You can stop and restart anytime - it remembers what's downloaded.")
    print()
    
    input("Press Enter to start (or Ctrl+C to cancel)...")
    print()
    
    # Path to the main download script
    download_script = Path(__file__).parent / "download_geodata.py"
    
    if not download_script.exists():
        print(f"ERROR: {download_script} not found!")
        sys.exit(1)
    
    # Run with automatic settings for USA Lower 48
    # This simulates user input to the interactive script
    print("Starting download with optimized settings...")
    print("  - Region: USA Lower 48 (contiguous states)")
    print("  - Tiles: 50 (full coverage)")
    print("  - Workers: 4 (parallel)")
    print()
    
    # Create input for the interactive script
    # Answers: y (continue), 50 tiles, 4 workers, y to confirm
    user_input = "y\n50\n4\ny\n"
    
    try:
        # Run the download script with automated inputs
        result = subprocess.run(
            [sys.executable, str(download_script)],
            input=user_input,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print()
            print("=" * 70)
            print("DOWNLOAD COMPLETE!")
            print("=" * 70)
            print()
            print("Data saved to: data/raw/insar/sentinel1/")
            print()
            print("Next steps:")
            print("  1. Run fusion: python multi_resolution_fusion.py --output usa_hires")
            print("  2. Detect voids: python detect_voids.py --output usa_voids")
        else:
            print()
            print("=" * 70)
            print("Download interrupted or failed")
            print("=" * 70)
            print()
            print("Don't worry! Just run this script again to resume.")
            print("Already downloaded files will be skipped.")
    
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Download paused")
        print("=" * 70)
        print()
        print("Run this script again to resume from where you left off.")
        sys.exit(0)

if __name__ == "__main__":
    main()