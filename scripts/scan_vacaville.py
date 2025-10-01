import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr

def vacaville_bbox(pad_deg: float = 0.25):
    # Approx Vacaville, CA
    lat, lon = 38.3566, -121.9870
    return (lat - pad_deg, lat + pad_deg, lon - pad_deg, lon + pad_deg)

def generate_synthetic_data(modality, bbox):
    """Generate synthetic data for demonstration."""
    lat_min, lat_max, lon_min, lon_max = bbox
    n_points = 100
    lats = np.linspace(lat_min, lat_max, n_points)
    lons = np.linspace(lon_min, lon_max, n_points)
    
    if modality == 'gravity':
        # Synthetic gravity anomaly: mGal
        anomaly = np.random.normal(0, 0.1, n_points) + 9.8  # Base + noise
        values = anomaly
        units = 'mGal'
        source = 'synthetic'
    elif modality == 'magnetic':
        # Synthetic magnetic TMI: nT
        anomaly = np.random.normal(0, 5, n_points) + 50000
        values = anomaly
        units = 'nT'
        source = 'synthetic'
    elif modality == 'insar':
        # Synthetic LOS displacement: mm
        values = np.random.normal(0, 1, n_points)
        units = 'mm'
        source = 'synthetic'
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    # Create RawData-like structure
    metadata = {
        'bbox': bbox,
        'units': units,
        'source': source,
        'timestamp': datetime.now(),
        'parameters': {'synthetic': True}
    }
    
    # Create a 2D grid of data values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Flatten the grids and values for easier handling
    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    
    # Generate values for all points in the grid (not just n_points)
    n_total_points = len(lats) * len(lons)
    if modality == 'gravity':
        grid_values = np.random.normal(0, 0.1, n_total_points) + 9.8  # Base + noise
    elif modality == 'magnetic':
        grid_values = np.random.normal(0, 5, n_total_points) + 50000
    elif modality == 'insar':
        grid_values = np.random.normal(0, 1, n_total_points)
    
    # Create a simple xarray Dataset with the synthetic data
    data = xr.Dataset({
        'values': (['point'], grid_values),
        'latitude': (['point'], points[:, 0]),
        'longitude': (['point'], points[:, 1])
    })
    
    class MockRawData:
        def __init__(self, values, metadata):
            self.values = values
            self.metadata = metadata
        
        def validate(self):
            # Mock validation method
            return True
    
    return MockRawData(data, metadata)

def main():
    bbox = vacaville_bbox()
    print(f"Scanning Vacaville, CA bbox={bbox}")
    
    # Import ingestion manager via spec while faking package to satisfy relative imports
    root = Path(__file__).resolve().parents[1]
    gam_root = root / 'gam'
    manager_path = gam_root / 'ingestion' / 'manager.py'
    
    import types
    pkg_gam = types.ModuleType('gam')
    pkg_gam.__path__ = [str(gam_root)]
    sys.modules.setdefault('gam', pkg_gam)
    pkg_ing = types.ModuleType('gam.ingestion')
    pkg_ing.__path__ = [str(gam_root / 'ingestion')]
    sys.modules.setdefault('gam.ingestion', pkg_ing)
    
    spec = importlib.util.spec_from_file_location('gam.ingestion.manager', str(manager_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    mod.__package__ = 'gam.ingestion'
    sys.modules['gam.ingestion.manager'] = mod
    spec.loader.exec_module(mod)
    IngestionManager = getattr(mod, 'IngestionManager')
    mgr = IngestionManager()
    
    modalities = ['gravity', 'magnetic', 'insar']
    results = {}
    
    for mod in modalities:
        try:
            # Use real data from fetchers with appropriate parameters
            if mod == 'insar':
                # Use a valid date range for InSAR (past dates that are more likely to exist)
                from datetime import timedelta
                end_date = datetime.now() - timedelta(days=90)  # Go back 90 days to ensure we have past data
                start_date = end_date - timedelta(days=30)
                data = mgr.fetch_modality(mod, bbox,
                                         start_date=start_date.strftime('%Y-%m-%d'),
                                         end_date=end_date.strftime('%Y-%m-%d'))
            else:
                data = mgr.fetch_modality(mod, bbox)
            data.validate()
            count = len(data.values) if hasattr(data.values, '__len__') else 'N/A'
            print(f"[{mod}] OK - {count} values; source={data.metadata.get('source')}\n")
            results[mod] = data
        except Exception as e:
            print(f"[{mod}] FAILED - {e}")
    
    # Summary
    print("\n\nSummary:")
    for mod in modalities:
        status = 'ok' if mod in results else 'skipped/failed'
        print(f" - {mod}: {status}")

if __name__ == '__main__':
    main()
