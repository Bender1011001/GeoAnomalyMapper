#!/usr/bin/env python3
"""
Multi-Modal Geophysical Data Integration

Integrates additional datasets into GeoAnomalyMapper:
1. Seismic velocity models (SL2013sv_0.5d-grd_v2.1.tar.bz2)
2. Lithological maps (LiMW_GIS 2015.gdb)
3. SAR/InSAR data (future integration)

Extracts, processes, and aligns data to the global 0.1° grid.
"""

import logging
import sys
import tarfile
import subprocess
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
import fiona
import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def extract_seismic_model(tar_path: Path, output_dir: Path):
    """Extract and process seismic velocity model."""
    logger.info(f"Extracting seismic model from {tar_path}")
    
    if not tar_path.exists():
        logger.warning(f"Seismic model file not found: {tar_path}")
        return None
    
    extract_dir = output_dir / 'seismic_model'
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract tar.bz2
    with tarfile.open(tar_path, 'r:bz2') as tar:
        tar.extractall(path=extract_dir)
        members = tar.getnames()
        logger.info(f"Extracted {len(members)} files")
    
    # Find grid files (typically .grd or .nc format)
    grid_files = list(extract_dir.glob('**/*.grd')) + list(extract_dir.glob('**/*.nc'))
    
    if not grid_files:
        logger.warning("No grid files found in archive")
        return None
    
    logger.info(f"Found {len(grid_files)} grid files")
    for gf in grid_files:
        logger.info(f"  - {gf.name}")
    
    # Process the main grid file (usually the largest or first one)
    main_grid = grid_files[0]
    logger.info(f"Processing main grid: {main_grid}")
    
    # Convert to GeoTIFF if needed
    output_tif = output_dir / 'seismic_velocity_model.tif'
    
    try:
        # Use gdal_translate to convert to GeoTIFF
        cmd = [
            'gdal_translate',
            '-of', 'GTiff',
            '-co', 'COMPRESS=DEFLATE',
            str(main_grid),
            str(output_tif)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Converted to GeoTIFF: {output_tif}")
        return output_tif
    except Exception as e:
        logger.error(f"Failed to convert seismic model: {e}")
        return main_grid  # Return original if conversion fails


def process_lithology_gdb(gdb_path: Path, output_dir: Path):
    """Process ESRI File Geodatabase with lithology data."""
    logger.info(f"Processing lithology database: {gdb_path}")
    
    if not gdb_path.exists():
        logger.warning(f"Lithology GDB not found: {gdb_path}")
        return None
    
    try:
        # List layers in GDB
        layers = fiona.listlayers(str(gdb_path))
        logger.info(f"Found {len(layers)} layers in GDB:")
        for layer in layers:
            logger.info(f"  - {layer}")
        
        # Read the main lithology layer (usually the largest/main one)
        # Common names: GLiM, Lithology, LithologicalMap, etc.
        main_layer = None
        for layer_name in layers:
            if any(keyword in layer_name.lower() for keyword in ['glim', 'lith', 'geology']):
                main_layer = layer_name
                break
        
        if not main_layer and layers:
            main_layer = layers[0]
        
        if not main_layer:
            logger.warning("No suitable layer found in GDB")
            return None
        
        logger.info(f"Reading layer: {main_layer}")
        gdf = gpd.read_file(str(gdb_path), layer=main_layer)
        
        logger.info(f"Loaded {len(gdf)} features")
        logger.info(f"Columns: {list(gdf.columns)}")
        logger.info(f"CRS: {gdf.crs}")
        
        # Save as GeoPackage for easier access
        output_gpkg = output_dir / 'lithology_map.gpkg'
        gdf.to_file(output_gpkg, driver='GPKG')
        logger.info(f"Saved lithology data: {output_gpkg}")
        
        # Rasterize to match global grid
        output_tif = output_dir / 'lithology_raster.tif'
        rasterize_lithology(gdf, output_tif)
        
        return output_tif
        
    except Exception as e:
        logger.error(f"Failed to process lithology GDB: {e}")
        logger.exception(e)
        return None


def rasterize_lithology(gdf: gpd.GeoDataFrame, output_path: Path):
    """Rasterize lithology vector data to global grid."""
    logger.info("Rasterizing lithology data to global grid...")
    
    # Define global grid parameters (matching our anomaly grid)
    width = 3600  # 0.1° resolution
    height = 1800
    transform = from_origin(-180, 90, 0.1, 0.1)
    
    # Find lithology code column
    code_col = None
    for col in ['xx', 'lithology', 'lith_code', 'code', 'class']:
        if col in gdf.columns:
            code_col = col
            break
    
    if not code_col:
        logger.warning("No lithology code column found, using index")
        gdf['litho_id'] = range(len(gdf))
        code_col = 'litho_id'
    
    # Convert to raster using rasterio
    from rasterio.features import rasterize
    
    # Prepare shapes (geometry, value) tuples
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[code_col]))
    
    # Create raster
    burned = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=rasterio.uint16
    )
    
    # Write to file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint16,
        crs='EPSG:4326',
        transform=transform,
        compress='DEFLATE'
    ) as dst:
        dst.write(burned, 1)
    
    logger.info(f"Rasterized lithology saved: {output_path}")


def create_sar_acquisition_guide(output_dir: Path):
    """Create documentation and scripts for SAR data acquisition."""
    logger.info("Creating SAR data acquisition guide...")
    
    guide_content = """# SAR/InSAR Data Acquisition Guide for GeoAnomalyMapper

This guide provides automated tools and instructions for acquiring SAR/InSAR data
from multiple global sources to enhance anomaly detection.

## Data Sources

### 1. European Ground Motion Service (EGMS)
- **Coverage**: Europe
- **Resolution**: High-resolution InSAR ground motion
- **Access**: Web portal + command-line via wget

**Acquisition Steps**:
```bash
# 1. Go to EGMS Explorer and generate download links
# 2. Save the download_links.txt file
# 3. Download using wget:
wget -i path/to/egms_download_links.txt -P data/raw/sar/egms/

# 4. Or use EGMStream (automated tool):
# Install: pip install egmstream
# egmstream download --input download_links.txt --output data/raw/sar/egms/
```

### 2. EarthScope / UNAVCO (North America)
- **Coverage**: North America, select global sites
- **Resolution**: Various SAR missions
- **Access**: SSARA API (command-line)

**Acquisition Steps**:
```bash
# Install SSARA client
git clone https://www.unavco.org/gitlab/unavco_public/ssara_client.git
cd ssara_client
python setup.py install

# Query and download SAR data
ssara --platform=SENTINEL-1 \
      --start=2020-01-01 \
      --end=2023-12-31 \
      --intersectsWith="POINT(-104.4 32.4)" \
      --download \
      --parallel=4

# Save to organized directory
ssara --platform=SENTINEL-1 \
      --start=2020-01-01 \
      --end=2023-12-31 \
      --bbox="-105,-104,32,33" \
      --download \
      --output=data/raw/sar/earthscope/
```

### 3. JAXA ALOS PALSAR (Japan & Global)
- **Coverage**: Global
- **Resolution**: 25m mosaics
- **Access**: JAXA EORC Portal, Google Earth Engine, AWS

**Acquisition Steps**:

**Option A: AWS S3 (Recommended for bulk)**
```bash
# Install AWS CLI
pip install awscli

# List available PALSAR-2 data
aws s3 ls s3://alos-palsar2-scansar/ --no-sign-request

# Download specific tiles
aws s3 cp s3://alos-palsar2-scansar/path/to/tile.zip \
    data/raw/sar/palsar/ --no-sign-request --recursive
```

**Option B: Google Earth Engine**
```python
import ee
ee.Initialize()

# Load PALSAR mosaic
palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')
filtered = palsar.filterDate('2020-01-01', '2020-12-31')

# Export to Drive or download
task = ee.batch.Export.image.toDrive(
    image=filtered.mosaic(),
    description='PALSAR_Export',
    scale=25,
    region=your_geometry
)
task.start()
```

### 4. COMET LiCSAR (Global)
- **Coverage**: Global tectonic/volcanic regions
- **Resolution**: Sentinel-1 interferograms
- **Access**: CEDA Archive + LiCSAR Web Tools

**Acquisition Steps**:

**Option A: LiCSAR Web Tools (Python)**
```bash
# Install LiCSAR tools
git clone https://github.com/matthew-gaddes/LiCSAR-web-tools.git
cd LiCSAR-web-tools
pip install -r requirements.txt

# Download interferograms for a frame
python download_frame.py \
    --frame=001D_05123_131313 \
    --start=2020-01-01 \
    --end=2023-12-31 \
    --products=interferograms,coherence \
    --output=data/raw/sar/licssar/
```

**Option B: Direct wget from CEDA**
```bash
# Browse and download from CEDA archive
# URL: https://data.ceda.ac.uk/neodc/comet/data/licsar_products/

# Example: Download specific frame
wget -r -np -nH --cut-dirs=5 \
    -P data/raw/sar/licssar/ \
    https://data.ceda.ac.uk/neodc/comet/data/licsar_products/001/001D_05123_131313/
```

## Integration Workflow

After acquiring SAR data, integrate it into GeoAnomalyMapper:

1. **Preprocess SAR Data**:
```bash
python scripts/preprocess_sar.py \
    --input data/raw/sar/ \
    --output data/processed/sar/ \
    --grid-resolution 0.1
```

2. **Generate InSAR Time Series**:
```bash
python scripts/generate_insar_timeseries.py \
    --input data/processed/sar/ \
    --output data/processed/insar_timeseries/
```

3. **Fuse with Existing Anomalies**:
```bash
python scripts/fuse_multimodal_data.py \
    --magnetic data/outputs/cog/mag/ \
    --gravity data/outputs/cog/grav/ \
    --insar data/processed/insar_timeseries/ \
    --output data/outputs/cog/multimodal/
```

## Automated Download Scripts

See the `scripts/sar_acquisition/` directory for automated download scripts:
- `download_egms.sh` - EGMS data downloader
- `download_earthscope.py` - EarthScope/UNAVCO automation
- `download_palsar_aws.sh` - JAXA PALSAR from AWS
- `download_licssar.py` - LiCSAR frame downloader

## Data Organization

Organize SAR data in the following structure:
```
data/raw/sar/
├── egms/           # European Ground Motion Service
├── earthscope/     # EarthScope/UNAVCO data
├── palsar/         # JAXA ALOS PALSAR
└── licssar/        # COMET LiCSAR products
    ├── interferograms/
    ├── coherence/
    └── metadata/
```

## Requirements

Install required tools:
```bash
pip install awscli earthengine-api geopandas rasterio
pip install egmstream  # For EGMS
```

For GDAL-based tools:
```bash
conda install -c conda-forge gdal
```

## References

- EGMS Explorer: https://egms.land.copernicus.eu/
- EarthScope SSARA: https://www.unavco.org/gitlab/unavco_public/ssara_client
- JAXA EORC: https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm
- LiCSAR: https://comet.nerc.ac.uk/COMET-LiCS-portal/
- CEDA Archive: https://data.ceda.ac.uk/neodc/comet/data/licsar_products/
"""
    
    guide_path = output_dir / 'SAR_ACQUISITION_GUIDE.md'
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.info(f"SAR acquisition guide created: {guide_path}")
    return guide_path


def main():
    """Main integration function."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GEOANOMALYMAPPER - MULTI-MODAL DATA INTEGRATION")
    print("="*70 + "\n")
    
    # 1. Process Seismic Velocity Model
    seismic_tar = raw_data_dir / 'SL2013sv_0.5d-grd_v2.1.tar.bz2'
    seismic_output = extract_seismic_model(seismic_tar, processed_dir)
    
    # 2. Process Lithology GDB
    litho_gdb = raw_data_dir / 'LiMW_GIS 2015.gdb'
    litho_output = process_lithology_gdb(litho_gdb, processed_dir)
    
    # 3. Create SAR acquisition documentation
    sar_guide = create_sar_acquisition_guide(project_root / 'docs')
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\nProcessed Datasets:")
    if seismic_output:
        print(f"  ✓ Seismic Velocity Model: {seismic_output}")
    else:
        print(f"  ✗ Seismic Model not found (expected: {seismic_tar})")
    
    if litho_output:
        print(f"  ✓ Lithology Map: {litho_output}")
    else:
        print(f"  ✗ Lithology GDB not found (expected: {litho_gdb})")
    
    print(f"\n  ✓ SAR Acquisition Guide: {sar_guide}")
    
    print("\nNext Steps:")
    print("  1. Review SAR acquisition guide for InSAR data integration")
    print("  2. Run multi-modal fusion to combine all datasets")
    print("  3. Generate enhanced anomaly maps with all data sources")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()