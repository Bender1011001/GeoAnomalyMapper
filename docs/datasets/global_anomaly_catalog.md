# Global Anomaly Data Catalog

## Overview

This catalog compiles global datasets for magnetic anomalies, gravity anomalies, terrain/bathymetry (including masks for corrections and context), optional SAR/InSAR sources for deformation inputs, and auxiliary geoid/crustal models. The goal is to enable maximum world coverage in the GeoAnomalyMapper project for anomaly detection and fusion workflows.

Datasets are tiered for usage:
- **MVP (Minimum Viable Product)**: Focus on freely accessible, no-credential-required rasters for quick global prototyping. Emphasizes pre-gridded formats to minimize processing.
- **Extended**: Includes higher-resolution or specialized data that may require free registration, gridding, or additional processing for refined analysis.

Cross-references:
- Source definitions are maintained in [`data_sources.yaml`](data_sources.yaml).
- Ingestion fetchers are implemented in [`fetchers.py`](GeoAnomalyMapper/gam/ingestion/fetchers.py).
- CLI entrypoints for data handling are in [`cli.py`](GeoAnomalyMapper/gam/cli/cli.py).
- A machine-readable JSON catalog for gravity and magnetic data is at [`gravity_magnetic_catalog.json`](datasets/gravity_magnetic_catalog.json), to be updated in a future task.

This Markdown serves as a standalone reference; wiring to configs and code will follow.

## Magnetics (Global Anomaly Rasters and Tracklines)

Global magnetic anomaly datasets provide crustal magnetization insights, essential for lithospheric structure mapping.

| Name | Coverage | Resolution | Units | Format | Access URL(s) | Credentials | License | Recommended use |
|------|----------|------------|-------|--------|---------------|-------------|---------|-----------------|
| EMAG2 v3 Sea Level | Global (land and ocean) | 2 arc-min (~3.7 km) | nT | GeoTIFF | https://www.ncei.noaa.gov/products/emag2; Direct: https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Sea_Level.tif | No | Public Domain (NOAA) | mag (MVP raster base) |
| EMAG2 v3 Upward Continued | Global (land and ocean) | 2 arc-min (~3.7 km) | nT | GeoTIFF | https://www.ncei.noaa.gov/products/emag2; Direct: https://www.ngdc.noaa.gov/mgg/global/emag2_v3/EMAG2_V3_Upward_Continued_4km.tif | No | Public Domain (NOAA) | mag (extended, reduced noise) |
| WDMAM v2.1 Global Magnetic Anomaly Map | Global (land and ocean) | 2 arc-min (~3.7 km) | nT | GeoTIFF/NetCDF | https://www.worldmagneticmodel.com/WDMAM/; Portal: https://geomag.bgs.ac.uk/data_service/models_compass/wdmam.html | No | CC-BY 4.0 (consortium) | mag (extended, updated compilation) |
| Marine Trackline Magnetics (NOAA NCEI) | Global oceans (tracklines) | Variable (point data) | nT | ASCII/Shapefile | https://www.ncei.noaa.gov/maps/geophysical-tracklines/; Search portal: https://www.ncei.noaa.gov/access/geophysical-track-lines.html | No | Public Domain (NOAA) | mag (extended tracks; requires gridding) |

## Gravity (Global Anomaly Rasters and Models)

Gravity datasets capture density variations, crucial for subsurface anomaly detection. Pre-gridded rasters are prioritized for MVP.

| Name | Coverage | Resolution | Units | Format | Access URL(s) | Credentials | License | Recommended use |
|------|----------|------------|-------|--------|---------------|-------------|---------|-----------------|
| EGM2008-derived Free-Air Anomaly (pre-gridded) | Global | 5 arc-min (~9.2 km) | mGal | GeoTIFF | ICGEM: http://icgem.gfz-potsdam.de/home; Direct: http://icgem.gfz-potsdam.de/root_folder/models/EGM2008/g087cof; Mirror: https://earth-info.nga.mil/index.php?dir=geoid&action=egm2008 (use wget for grid) | No | Free (GFZ/ICGEM) | grav (MVP raster; note DNS flakiness) |
| SIO V31 Global Marine Gravity (Sandwell & Smith) | Global oceans | 1 arc-min (~1.85 km) | mGal | NetCDF/GeoTIFF | https://topex.ucsd.edu/marine_gravity/; Direct: https://topex.ucsd.edu/WWW_grav_cgi/map_get.cgi?scale=1min&res=15&fmt=img (API for tiles) | No | Free with Scripps attribution | grav (extended ocean focus) |
| GOCE/EGG/EIGEN-6C4 Models | Global | Variable (harmonics up to degree 2190) | mGal | Spherical harmonics (GFC) | https://icgem.gfz-potsdam.de/home; Direct: http://icgem.gfz-potsdam.de/root_folder/models/EIGEN/eigen6c4/ | Free registration (ESA Earthdata) | Free (ESA/GFZ) | grav (extended; requires gridding) |
| WGM2012 Global Bouguer Anomaly | Global | 5 arc-min (~9.2 km) | mGal | Binary grid | https://www.earthbyte.org/webdav/ftp/Data_collections/2012_gravity/WGM2012/; DOI: 10.5880/fidgeo.2012.003 | Research use (registration/terms) | CC-BY-NC (GFZ) | grav (extended; caveats on inland use) |

## Terrain/Bathymetry and Masks (for Corrections/Context)

These provide topography, bathymetry, and land/ocean masks for terrain corrections and visualization.

| Name | Coverage | Resolution | Units | Format | Access URL(s) | Credentials | License | Recommended use |
|------|----------|------------|-------|--------|---------------|-------------|---------|-----------------|
| ETOPO2022 Global Relief | Global (land/ocean/ice) | 15 arc-sec (~500 m) | m | GeoTIFF/NetCDF | https://www.ncei.noaa.gov/products/etopo-global-relief-model; Direct: https://www.ncei.noaa.gov/data/ocean-floor/topography/etopo2022/ | No | Public Domain (NOAA) | aux (MVP terrain/bathymetry) |
| GEBCO 2024 Global Bathymetry | Global oceans/lands | 15 arc-sec (~500 m) | m | NetCDF/GeoTIFF | https://www.gebco.net/data_and_products/gridded_bathymetry_data/; Direct: https://download.gebco.net/release/2024/gebco_2024.nc.gz | No | CC-BY 4.0 (GEBCO) | aux (extended bathymetry) |
| SRTM 1 Arc-sec Global Topography | Global land (60°N-56°S) | 1 arc-sec (~30 m) | m | GeoTIFF | https://earthexplorer.usgs.gov/; NASA Earthdata portal | Free registration (NASA Earthdata) | Free (NASA/USGS) | aux (extended land topo) |
| GSHHG/Natural Earth Coastlines and Masks | Global | Variable (1:10m to 1:110m) | - | Shapefile/GeoJSON | https://www.soest.hawaii.edu/pwessel/gshhg/; https://www.naturalearthdata.com/downloads/ | No | Public Domain/CC0 (GSHHG/Natural Earth) | aux (MVP masks/coastlines) |

## SAR/InSAR (Optional Global Deformation Inputs)

SAR/InSAR data for surface deformation monitoring; global coverage is patchy, requiring processing.

| Name | Coverage | Resolution | Units | Format | Access URL(s) | Credentials | License | Recommended use |
|------|----------|------------|-------|--------|---------------|-------------|---------|-----------------|
| Sentinel-1 GRD/IW SLC | Global (repeat pass) | 5-20 m (GRD), ~5x20 m (SLC) | mm (LOS) | SAF/GeoTIFF | ASF Vertex: https://search.asf.alaska.edu/; Copernicus SciHub: https://scihub.copernicus.eu/ | Free accounts (NASA Earthdata/Copernicus Open Access Hub) | Free (ESA) | optional (deformation fusion; processing required) |
| Global Velocity Products (e.g., COMET LiCS) | Regional/global patches | ~100 m | mm/yr | GeoTIFF/NetCDF | https://comet.earth/data/lics/; Portal: https://portal.comet.earth/lics-database/ | No (public releases) | CC-BY 4.0 (COMET) | optional (pre-processed velocities; cite source) |

## Geoid and Auxiliary Models

Supporting models for conversions, corrections, and crustal context.

| Name | Coverage | Resolution | Units | Format | Access URL(s) | Credentials | License | Recommended use |
|------|----------|------------|-------|--------|---------------|-------------|---------|-----------------|
| EGM2008/EGM96 Geoid Grids | Global | 5 arc-min (~9.2 km) | m | Binary/GeoTIFF | https://earth-info.nga.mil/index.php?dir=geoid&action=download; Direct: https://cddis.nasa.gov/archive/gnss/products/egm/EGM2008/ | No | Free (NGA) | aux (geoid corrections) |
| CRUST1.0/CRUST2.0 Global Crustal Models | Global | 1° x 1° (CRUST1.0) | km (thickness) | ASCII/NetCDF | https://igppwiki.unavco.org/index.php/CRUST1.0; Direct: https://wwwigpp.usc.edu/~geodynamics/crust/crust1.tar.gz | No | Free (Laske et al.) | aux (crustal context) |

## Recommended MVP Set

For immediate global coverage with zero barriers:
- **Magnetic**: EMAG2 v3 Sea Level GeoTIFF (NOAA) – Direct raster download.
- **Gravity**: Pre-gridded EGM2008 Free-Air Anomaly GeoTIFF (ICGEM/GFZ) – Use local caching or mirrors.
- **Terrain/Mask**: ETOPO2022 Global Relief (NOAA) + GSHHG/Natural Earth Coastlines – For corrections and masking.
- **Credentials required**: None for MVP. Extended options (e.g., Sentinel-1) need free Earthdata/Copernicus registration.

This set enables end-to-end anomaly fusion without registration or heavy processing.

## Operational Notes and Mirrors

- **Stable Endpoints**: Prefer NOAA/NCEI and GFZ/ICGEM for rasters; use `wget` or `curl` for bulk downloads. For EGM2008, canonical path is `http://icgem.gfz-potsdam.de/root_folder/models/EGM2008/g087cof` – if DNS flaky, mirror via NGA: `https://earth-info.nga.mil/` or local register (e.g., via `pip install icgem` for programmatic access).
- **Mirrors/Manual Fallbacks**: SIO gravity: Use Scripps API tiles if full grid fails. Sentinel-1: ASF Vertex for bulk, SciHub for search.
- **Checksums/Manifests**: Always verify downloads (e.g., MD5 for ETOPO: listed on NOAA page). Implement provenance logging in ingestion (e.g., via `fetchers.py`).
- **Local Workflow**: Register datasets locally post-download; use checksums for integrity. For harmonics (e.g., EIGEN), grid via tools like `harmonics2grid` before pipeline input.
- **Coverage Gaps**: MVP covers ~90% global; extended fills oceans/high-res land via registration.

This catalog will be wired into [`data_sources.yaml`](data_sources.yaml) and [`gravity_magnetic_catalog.json`](datasets/gravity_magnetic_catalog.json) in subsequent tasks.