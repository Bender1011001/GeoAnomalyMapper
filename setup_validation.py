"""
Quick setup script to install validation dependencies
Run once before using the automated validation suite
"""

import subprocess
import sys

def install_packages():
    """Install required packages"""
    
    packages = [
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'geopandas',
        'requests'
    ]
    
    print("Installing validation dependencies...")
    print("="*60)
    
    for package in packages:
        print(f"\nInstalling {package}...", end=" ")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package, '-q'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓")
        except subprocess.CalledProcessError:
            print(f"✗ (may already be installed)")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("\nYou can now run:")
    print("  Windows: run_validation.bat")
    print("  Linux/Mac: ./run_validation.sh")
    print("="*60)

if __name__ == "__main__":
    install_packages()