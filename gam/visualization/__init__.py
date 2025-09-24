"""
Visualization and Output Module.

Generates 2D/3D maps, interactive dashboards, and exports anomalous results.
Supports PyGMT for high-quality 2D basemaps, PyVista for 3D volume rendering, Folium for interactive web maps,
and exporters to GeoTIFF, VTK, or SQL databases via SQLAlchemy.
"""
from .manager import VisualizationManager