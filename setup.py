from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r") as f:
    lines = f.readlines()
    install_requires = [line.strip() for line in lines if line.strip() and not line.startswith("#") and not "gdal" in line.lower() and not any(test in line.lower() for test in ["pytest", "black", "flake8", "coverage", "sphinx"])]

# Extras for optional features and dev
extras_require = {
    "geophysics": [
        "simpeg>=0.21.0",
        "obspy>=1.4.1",
        "mintpy>=1.6.0",
        "pygimli>=1.5.0",
    ],
    "visualization": [
        "pygmt>=0.12.0",
        "pyvista>=0.44.0",
        "folium>=0.17.0",
        "rasterio>=1.3.10",
        "vtk>=9.3.0",
    ],
    "dev": [
        "pytest>=8.3.0",
        "black>=24.4.2",
        "flake8>=7.0.0",
        "coverage>=7.6.0",
        "sphinx>=7.4.0",
        "psutil>=6.0.0",
        "pyproj>=3.6.0",
        "tenacity>=8.2.0",
    ],
}

setup(
    name="GeoAnomalyMapper",
    version="0.1.0",
    author="GeoAnomalyMapper Team",
    author_email="team@example.com",  # Update with real
    description="Geophysical data fusion for subsurface anomaly detection.",
    long_description="""
GeoAnomalyMapper (GAM) is an open-source Python package for integrating and analyzing geophysical datasets to detect subsurface anomalies. It supports gravity, magnetic, seismic, and InSAR data fusion through modular pipelines for ingestion, preprocessing, inversion modeling, and visualization.

Key features:
- Automated data fetching from USGS, IRIS, ESA with caching.
- Multi-modal inversion and Bayesian fusion.
- Parallel processing with Dask for global-scale analysis.
- Interactive 2D/3D visualizations and exports (GeoTIFF, VTK, CSV).

Installation:
pip install GeoAnomalyMapper[geophysics,visualization]

Usage:
from gam import GAMPipeline
pipeline = GAMPipeline()
results = pipeline.run_analysis(bbox=(29,31,30,32), modalities=['gravity'])

See README.md for full documentation and examples.
""",
    long_description_content_type="text/plain",
    url="https://github.com/GeoAnomalyMapper/GAM",  # Update to real repo
    project_urls={
        "Documentation": "https://geoanomalymapper.readthedocs.io",
        "Source": "https://github.com/GeoAnomalyMapper/GAM",
        "Tracker": "https://github.com/GeoAnomalyMapper/GAM/issues",
    },
    packages=find_packages(include=["gam", "gam.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Geoscience",
        "Topic :: Software Development :: Testing",
    ],
    keywords="geophysics, anomaly detection, inversion, data fusion, seismic, gravity, InSAR, GIS",
    license="MIT",
    license_files=["LICENSE"],
    entry_points={
        "console_scripts": [
            "gam = gam.core.cli:cli",  # Click group for subcommands
        ],
    },
    zip_safe=False,
    include_package_data=True,
    package_data={
        "gam": ["*.yaml", "data_sources.yaml"],
        "tests": ["data/*", "configs/*"],
    },
    data_files=[("config", ["config.yaml"])],
)