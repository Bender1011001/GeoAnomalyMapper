from setuptools import setup, find_packages

# Minimal setup.py for compatibility; dependencies managed via pyproject.toml
# See pyproject.toml for project metadata, dependencies, and scripts.

setup(
    name="GeoAnomalyMapper",
    packages=find_packages(include=["gam", "gam.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10,<3.13",
)