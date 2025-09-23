"""2D mapping functionality for GAM visualization module.

This module implements static and interactive 2D map generators for geophysical
data. Static maps use PyGMT for publication-quality cartography with support
for multiple projections and contouring. Interactive maps use Folium for web-based
Leaflet visualizations with markers, layers, and controls.

Both generators subclass Visualizer and implement generate() for consistency.
Static returns matplotlib.Figure; interactive returns folium.Map (save to HTML).

Supported features:
- Projections: Mercator, UTM, geographic (static); geographic with plugins (interactive).
- Contour maps for ProcessedGrid/InversionResults.
- Point maps for AnomalyOutput with sizing/color by confidence.
- Colorbars, scale bars, annotations, layer switching.

Notes
-----
- Color Schemes: Use kwargs['color_scheme'] (e.g., 'RdBu_r' for diverging); defaults
  data-specific (gravity: 'RdBu_r', magnetic: 'jet', anomalies: 'Reds').
- For 3D data: Projects surface slice (z=0).
- Dependencies: pygmt, folium, branca, matplotlib.
- Performance: PyGMT vectorized; Folium client-side for interactivity.
- Limitations: UTM requires zone specification via kwargs['utm_zone'].
"""

from __future__ import annotations

import logging
from typing import Union, Any, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import folium
from folium import LayerControl
from folium.plugins import HeatMap
import pygmt

from gam.visualization.base import Visualizer
from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

logger = logging.getLogger(__name__)


class StaticMapGenerator(Visualizer):
    """
    Generator for static 2D maps using PyGMT.

    Creates high-quality cartographic maps with contours for grids, points for
    anomalies, colorbars, scale bars, and annotations. Supports Mercator, UTM,
    geographic projections.

    Parameters
    ----------
    dpi : int, optional
        Figure DPI (default: 300).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> Figure
        Generate static map Figure.

    Notes
    -----
    - Projections: 'Mercator' (default), 'UTM' (specify utm_zone), 'geographic'.
    - For grids: grdimage with contour overlays.
    - For anomalies: plot symbols scaled by confidence.
    - Color schemes: Matplotlib colormaps; gravity 'RdBu_r', magnetic 'jet',
      seismic 'coolwarm', anomalies 'Reds'.
    - Annotations: Auto-title from data; kwargs for custom text.
    - Scale bar: Auto-placed bottom-right in map units.

    Examples
    --------
    >>> generator = StaticMapGenerator()
    >>> fig = generator.generate(
    ...     processed_grid,
    ...     projection='Mercator',
    ...     color_scheme='RdBu_r',
    ...     title='Gravity Anomaly Map'
    ... )
    >>> fig.savefig('static_map.png', dpi=300, bbox_inches='tight')
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        **kwargs: Any
    ) -> Figure:
        """
        Generate static 2D map from data.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data for mapping.
        **kwargs : dict, optional
            - 'projection': str ('Mercator', 'UTM', 'geographic'), default 'Mercator'.
            - 'color_scheme': str, colormap name (default data-specific).
            - 'title': str, map title (default auto).
            - 'utm_zone': int, for UTM projection.
            - 'region': Tuple[float, ...], (W, E, S, N) bounds (default from data).
            - 'frame': bool, add frame (default True).

        Returns
        -------
        Figure
            Matplotlib Figure with PyGMT plot.

        Raises
        ------
        ValueError
            If invalid projection or data lacks coords.
        """
        projection = kwargs.get('projection', 'Mercator').lower()
        if projection == 'mercator':
            proj = 'M15c'  # 15cm width
        elif projection == 'utm':
            zone = kwargs.get('utm_zone', 10)
            proj = f'U{zone}/10c'
        elif projection == 'geographic':
            proj = 'G10c'
        else:
            raise ValueError(f"Unsupported projection: {projection}")

        # Extract region (W, E, S, N)
        if isinstance(data, (ProcessedGrid, InversionResults)):
            coords = _extract_coords(data)  # Reuse from exporters if imported, or redefine
            region = (coords['lons'].min(), coords['lons'].max(), coords['lats'].min(), coords['lats'].max())
        elif isinstance(data, AnomalyOutput):
            region = (data['lon'].min(), data['lon'].max(), data['lat'].min(), data['lat'].max())
        else:
            raise ValueError("Unsupported data type")

        region = kwargs.get('region', region)

        fig = pygmt.Figure()
        fig.basemap(region=region, projection=proj, frame=True)

        # Color scheme
        if isinstance(data, ProcessedGrid) or isinstance(data, InversionResults):
            color_scheme = kwargs.get('color_scheme', 'RdBu_r')  # Default diverging
            if 'gravity' in str(type(data)).lower():
                color_scheme = 'RdBu_r'
            elif 'magnetic' in str(type(data)).lower():
                color_scheme = 'jet'
            elif 'seismic' in str(type(data)).lower():
                color_scheme = 'coolwarm'

            # For grids: grdimage
            if isinstance(data, ProcessedGrid):
                # Temp grid file for PyGMT
                temp_grid = 'temp_grid.nc'
                data.to_netcdf(temp_grid)
                fig.grdimage(grid=temp_grid, cmap=color_scheme, projection=proj)
                fig.colorbar(frame=["a", f"Units: {data.units}"])
                os.remove(temp_grid)
            else:  # InversionResults, slice z=0
                slice_data = data.model[:, :, 0]
                # Create temp grid (simplified; use xarray in production)
                lons, lats = np.meshgrid(coords['lons'], coords['lats'])
                temp_ds = xr.Dataset({'data': (['lat', 'lon'], slice_data)}, coords={'lat': coords['lats'], 'lon': coords['lons']})
                temp_grid = 'temp_model.nc'
                temp_ds.to_netcdf(temp_grid)
                fig.grdimage(grid=temp_grid, cmap=color_scheme, projection=proj)
                fig.colorbar(frame=["a", f"Model: {data.metadata.get('units', 'unknown')}"])
                os.remove(temp_grid)

        elif isinstance(data, AnomalyOutput):
            color_scheme = kwargs.get('color_scheme', 'Reds')
            # Plot points
            sizes = data['confidence'] * 20 + 5  # Scale size by confidence
            fig.plot(
                x=data['lon'], y=data['lat'],
                style='c0.5p',  # Circle symbol
                fill=color_scheme,  # Color by scheme
                sizes=sizes,
                projection=proj
            )
            fig.colorbar(frame=["a", "Confidence (0-1)"])

        # Scale bar
        fig.basemapJ(region=region, projection=proj, J="x1c")  # Scale bar

        # Title and annotations
        title = kwargs.get('title', f"Geophysical Map - {type(data).__name__}")
        fig.text(pos=(0.5, 0.95), text=title, font="20p,Helvetica-Bold", justify="center CB")

        # Annotations if provided
        if 'annotations' in kwargs:
            for ann in kwargs['annotations']:
                fig.text(pos=ann['pos'], text=ann['text'], font="12p")

        logger.info(f"Generated static map with projection {projection}")
        return fig.get()  # Return underlying matplotlib Figure


class InteractiveMapGenerator(Visualizer):
    """
    Generator for interactive 2D web maps using Folium.

    Creates Leaflet-based maps with markers for anomalies, raster overlays for
    grids, layer controls, zoom, and popups. Primarily geographic projection.

    Parameters
    ----------
    zoom_start : int, optional
        Initial zoom level (default: 10).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> folium.Map
        Generate interactive map object (save with map.save('map.html')).

    Notes
    -----
    - Projections: Geographic (WGS84); for others, use folium plugins (not implemented).
    - Layers: Switch between data types (grid overlay, anomaly points).
    - Popups: Show confidence, type, coords for points.
    - For grids: Convert to PNG via matplotlib, overlay as Image.
    - Controls: Built-in zoom, layer control; coordinate display via custom popup.
    - Color schemes: Folium colormaps via LinearColormap.

    Examples
    --------
    >>> generator = InteractiveMapGenerator()
    >>> m = generator.generate(anomalies, layer_name='Anomalies')
    >>> m.save('interactive_map.html')
    """

    def __init__(self, zoom_start: int = 10):
        self.zoom_start = zoom_start

    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        **kwargs: Any
    ) -> folium.Map:
        """
        Generate interactive 2D map from data.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data for mapping.
        **kwargs : dict, optional
            - 'center': Tuple[float, float], (lat, lon) center (default from data mean).
            - 'layer_name': str, layer label (default 'Data Layer').
            - 'color_scheme': str, for heatmaps (default 'Reds').
            - 'popup_template': str, custom popup HTML.

        Returns
        -------
        folium.Map
            Interactive map object.

        Raises
        ------
        ValueError
            If data lacks lat/lon.
        """
        if isinstance(data, (ProcessedGrid, InversionResults)):
            lats = _extract_coords(data)['lats']
            lons = _extract_coords(data)['lons']
            center = [np.mean(lats), np.mean(lons)]
        elif isinstance(data, AnomalyOutput):
            center = [data['lat'].mean(), data['lon'].mean()]
        else:
            raise ValueError("Unsupported data type")

        center = kwargs.get('center', center)
        m = folium.Map(location=center, zoom_start=self.zoom_start, tiles='OpenStreetMap')

        layer_name = kwargs.get('layer_name', 'Data Layer')

        if isinstance(data, AnomalyOutput):
            # Add markers for anomalies
            for idx, row in data.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=row['confidence'] * 10 + 5,  # Scale by confidence
                    popup=kwargs.get('popup_template', f"Anomaly Type: {row['anomaly_type']}<br>Confidence: {row['confidence']:.2f}<br>Depth: {row['depth']}m"),
                    color='red' if row['confidence'] > 0.7 else 'orange',
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)

            # HeatMap for density
            heat_data = [[row['lat'], row['lon'], row['confidence']] for _, row in data.iterrows()]
            HeatMap(heat_data, name=layer_name, overlay=False).add_to(m)

        elif isinstance(data, (ProcessedGrid, InversionResults)):
            # For grids: Create image overlay (convert to PNG)
            # Simplified: Use matplotlib to plot grid and save to BytesIO
            fig, ax = plt.subplots(figsize=(10, 10))
            if isinstance(data, ProcessedGrid):
                im = ax.contourf(data.ds['lon'], data.ds['lat'], data.ds['data'], cmap=kwargs.get('color_scheme', 'RdBu_r'))
            else:
                # Slice
                slice_data = data.model[:, :, 0]
                lons_grid, lats_grid = np.meshgrid(_extract_coords(data)['lons'], _extract_coords(data)['lats'])
                im = ax.contourf(lons_grid, lats_grid, slice_data, cmap=kwargs.get('color_scheme', 'RdBu_r'))
            plt.colorbar(im, ax=ax, label='Value')
            ax.set_title(kwargs.get('title', 'Grid Map'))

            # Save to BytesIO for overlay
            from io import BytesIO
            img = BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)

            # Bounds for overlay: [[min_lat, min_lon], [max_lat, max_lon]]
            bounds = [[np.min(center[0] - 1), np.min(center[1] - 1)], [np.max(center[0] + 1), np.max(center[1] + 1)]]

            folium.raster_layers.ImageOverlay(
                image=img.getvalue(),
                bounds=bounds,
                name=layer_name,
                opacity=0.6
            ).add_to(m)
            plt.close(fig)

        # Layer control
        LayerControl().add_to(m)

        # Coordinate display popup on click (simple)
        m.add_child(folium.LatLngPopup())

        logger.info(f"Generated interactive map centered at {center}")
        return m