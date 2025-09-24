from setuptools import setup, find_packages
import os
from pathlib import Path

# Read requirements for core deps only (exclude optional sections)
with open("requirements.txt", "r") as f:
    lines = f.readlines()
    install_requires = []
    in_optional = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# Optional:"):
            in_optional = True
            continue
        if in_optional and not stripped.startswith("#"):
            continue
        if stripped and not stripped.startswith("#") and not "gdal" in stripped.lower() and not any(test in stripped.lower() for test in ["pytest", "black", "flake8", "coverage", "sphinx", "obspy", "streamlit", "pygmt", "pyvista", "fastapi", "uvicorn"]):
            install_requires.append(stripped)

# Extras for optional features and dev
extras_require = {
    "geophysics": [
        "simpeg>=0.21.0",
        "obspy>=1.4.1",
        "mintpy>=1.6.0",
        "pygimli>=1.5.0",
        "sentinelsat>=1.0.0",
    ],
    "visualization": [
        "pygmt>=0.12.0",
        "pyvista>=0.44.0",
        "folium>=0.17.0",
        "rasterio>=1.3.10",
        "vtk>=9.3.0",
    ],
    "dashboard": [
        "streamlit>=1.28.0",
        "streamlit-folium>=0.17.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "itkwidgets>=0.13.0",
        "pyvistaqt>=0.11.0",
        "stpyvista>=0.1.0",
    ],
    "gui": [
        "streamlit>=1.28.0",
        "streamlit-folium>=0.17.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pygmt>=0.12.0",
        "pyvista>=0.44.0",
        "folium>=0.17.0",
        "itkwidgets>=0.13.0",
        "vtk>=9.3.0",
        "pyvistaqt>=0.11.0",
        "stpyvista>=0.1.0",
    ],
    "all": [
        "simpeg>=0.21.0",
        "obspy>=1.4.1",
        "mintpy>=1.6.0",
        "pygimli>=1.5.0",
        "sentinelsat>=1.0.0",
        "pygmt>=0.12.0",
        "pyvista>=0.44.0",
        "folium>=0.17.0",
        "rasterio>=1.3.10",
        "vtk>=9.3.0",
        "streamlit>=1.28.0",
        "streamlit-folium>=0.17.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "itkwidgets>=0.13.0",
        "pyvistaqt>=0.11.0",
        "stpyvista>=0.1.0",
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

# Read README for long_description
this_directory = Path(__file__).parent
with open(this_directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="GeoAnomalyMapper",
    version="0.1.0",
    author="GeoAnomalyMapper Team",
    author_email="team@example.com",  # Update with real contact
    description="Geophysical data fusion for subsurface anomaly detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Typing :: Typed",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Geoscience",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Matplotlib",
        "Framework :: Jupyter",
    ],
    keywords="geophysics, anomaly detection, inversion, data fusion, seismic, gravity, InSAR, GIS, subsurface, earth science",
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