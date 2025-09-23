"""3D visualization functionality for GAM visualization module.

This module implements volume rendering and cross-section generation for 3D
geophysical models using PyVista. VolumeRenderer creates interactive 3D volumes
with isosurfaces, slices, and rendering modes. CrossSectionGenerator extracts
2D slices and 1D profiles for detailed analysis.

Both classes subclass Visualizer and implement generate() for integration.
VolumeRenderer returns a PyVista Plotter (interactive); for static, use .show(return_img=True).
CrossSectionGenerator returns matplotlib Figure.

Supported features:
- Volume rendering with opacity/color mapping.
- Isosurface extraction at thresholds.
- Slice planes (orthogonal/arbitrary).
- Rendering modes: volume, surface, wireframe, points.
- Cross-sections: vertical/horizontal slices, line profiles.
- Overlays: Multiple data on same plot/view.

Notes
-----
- Data: Primarily InversionResults (3D model array); ProcessedGrid if 3D.
- Coordinates: Derived from data; regular grid assumed for spacing.
- Dependencies: pyvista, matplotlib, numpy.
- Performance: GPU-accelerated via VTK; downsample large models if needed.
- Limitations: Assumes Cartesian coords; curvilinear grids require mesh import.
- Interactivity: Plotter for VolumeRenderer; save screenshots for reports.
"""

from __future__ import annotations

import logging
from typing import Union, Any, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pyvista as pv

from gam.visualization.base import Visualizer
from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput
from gam.visualization.exporters import _extract_coords  # Reuse helper

logger = logging.getLogger(__name__)


class VolumeRenderer(Visualizer):
    """
    Renderer for 3D volume visualizations using PyVista.

    Creates volumetric renders of subsurface models with isosurfaces, slice planes,
    opacity controls, and multiple rendering modes. Supports InversionResults models
    and 3D ProcessedGrid data.

    Parameters
    ----------
    theme : str, optional
        PyVista theme ('document', 'default'; default: 'document' for publication).

    Methods
    -------
    generate(data: Union[InversionResults, ProcessedGrid], **kwargs) -> pv.Plotter
        Generate 3D volume plotter (use .show() for display).

    Notes
    -----
    - Rendering Modes: 'volume' (default, add_volume), 'surface' (add_mesh),
      'wireframe' (wireframe=True), 'points' (add_points).
    - Isosurfaces: Extract at kwargs['threshold'] (default 0.5); multiple via list.
    - Slices: Add plane at kwargs['slice_origin'] and normal (e.g., 'x', 'y', 'z').
    - Color/Opacity: cmap (default 'coolwarm'), opacity (0.1-1.0, default 0.6).
    - For large models (>1M cells): Auto-downsample by factor in kwargs.
    - Output: Plotter for interactivity; .screenshot() for static image.
    - View in: ParaView or Jupyter (pv.start_xvfb() for headless).

    Examples
    --------
    >>> renderer = VolumeRenderer()
    >>> plotter = renderer.generate(
    ...     inversion_results,
    ...     mode='volume',
    ...     threshold=0.5,
    ...     cmap='viridis',
    ...     opacity=0.7
    ... )
    >>> plotter.show()  # Interactive 3D view
    >>> img = plotter.screenshot('volume.png')
    """

    def __init__(self, theme: str = 'document'):
        pv.set_plot_theme(theme)
        self.theme = theme

    def generate(
        self,
        data: Union[InversionResults, ProcessedGrid],
        **kwargs: Any
    ) -> pv.Plotter:
        """
        Generate 3D volume visualization.

        Parameters
        ----------
        data : Union[InversionResults, ProcessedGrid]
            3D model or grid data.
        **kwargs : dict, optional
            - 'mode': str ('volume', 'surface', 'wireframe', 'points'), default 'volume'.
            - 'threshold': float or List[float], isosurface level(s), default 0.5.
            - 'cmap': str or List, color map (default 'coolwarm').
            - 'opacity': float, volume opacity (0.1-1.0, default 0.6).
            - 'slice_origin': Tuple[float, float, float], slice plane origin.
            - 'slice_normal': str or Tuple, plane normal ('x', 'y', 'z' or vector), default None.
            - 'downsample': int, reduce resolution (default 1, no downsample).
            - 'title': str, window title (default auto).

        Returns
        -------
        pv.Plotter
            3D plotter instance.

        Raises
        ------
        ValueError
            If non-3D data or invalid mode/threshold.
        """
        if not isinstance(data, (InversionResults, ProcessedGrid)):
            raise ValueError("VolumeRenderer supports only InversionResults or 3D ProcessedGrid")

        if isinstance(data, ProcessedGrid) and len(data.ds.dims) < 3:
            raise ValueError("ProcessedGrid must have 3D dimensions for volume rendering")

        # Extract model and coords
        if isinstance(data, InversionResults):
            model = data.model
            uncertainty = data.uncertainty
            metadata = data.metadata
        else:
            model = data.ds['data'].values
            uncertainty = data.ds.get('uncertainty', np.zeros_like(model)).values
            metadata = dict(data.ds.attrs)

        coords = _extract_coords(data)
        n_x, n_y, n_z = model.shape
        spacing = (
            (coords['lons'][-1] - coords['lons'][0]) / n_x,
            (coords['lats'][-1] - coords['lats'][0]) / n_y,
            (coords['depths'][-1] - coords['depths'][0]) / n_z
        )

        # Downsample if large
        downsample = kwargs.get('downsample', 1)
        if downsample > 1:
            model = model[::downsample, ::downsample, ::downsample]
            uncertainty = uncertainty[::downsample, ::downsample, ::downsample]
            n_x, n_y, n_z = model.shape
            spacing = (spacing[0] * downsample, spacing[1] * downsample, spacing[2] * downsample)
            logger.warning(f"Downsampled model by factor {downsample}")

        # Create grid
        grid = pv.UniformGrid(dimensions=(n_x, n_y, n_z))
        grid.origin = (coords['lons'][0], coords['lats'][0], coords['depths'][0])
        grid.spacing = spacing
        grid.cell_data['model'] = model.ravel(order='F')
        if np.any(uncertainty):
            grid.cell_data['uncertainty'] = uncertainty.ravel(order='F')

        plotter = pv.Plotter(window_size=[800, 600])
        plotter.title = kwargs.get('title', f"3D Volume - {metadata.get('units', 'unknown')}")

        mode = kwargs.get('mode', 'volume').lower()
        cmap = kwargs.get('cmap', 'coolwarm')
        opacity = kwargs.get('opacity', 0.6)

        if mode == 'volume':
            plotter.add_volume(grid, scalars='model', cmap=cmap, opacity=opacity, shade=True)
        elif mode == 'surface':
            plotter.add_mesh(grid, scalars='model', cmap=cmap, opacity=opacity)
        elif mode == 'wireframe':
            plotter.add_mesh(grid, scalars='model', cmap=cmap, style='wireframe', line_width=1)
        elif mode == 'points':
            plotter.add_points(grid.points, scalars=grid['model'], cmap=cmap, point_size=2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Isosurfaces
        thresholds = kwargs.get('threshold', [0.5])
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds]
        for thresh in thresholds:
            plotter.add_isosurface(grid, isosurfaces=thresh, scalars='model', cmap='viridis', opacity=0.8)

        # Slice plane
        slice_origin = kwargs.get('slice_origin', grid.center)
        slice_normal = kwargs.get('slice_normal', 'z')  # Default horizontal
        if isinstance(slice_normal, str):
            if slice_normal == 'x':
                normal = (1, 0, 0)
            elif slice_normal == 'y':
                normal = (0, 1, 0)
            elif slice_normal == 'z':
                normal = (0, 0, 1)
            else:
                normal = (0, 0, 1)
        else:
            normal = slice_normal
        slice_plane = grid.slice(normal=normal, origin=slice_origin)
        plotter.add_mesh(slice_plane, scalars='model', cmap=cmap, opacity=0.9, show_edges=True)

        # Add uncertainty if present
        if 'uncertainty' in grid.cell_data:
            plotter.add_mesh(grid.outline(), color='gray', label='Uncertainty Outline')

        plotter.add_legend()  # If labels added
        logger.info(f"Generated 3D volume in mode '{mode}' with {len(thresholds)} isosurfaces")
        return plotter


class CrossSectionGenerator(Visualizer):
    """
    Generator for 2D cross-sections and 1D profiles from 3D models.

    Extracts vertical/horizontal slices and generates profile plots along lines.
    Supports overlays for multiple datasets or components (e.g., model + uncertainty).

    Parameters
    ----------
    figsize : Tuple[int, int], optional
        Figure size (default: (12, 8)).

    Methods
    -------
    generate(data: Union[InversionResults, ProcessedGrid], **kwargs) -> Figure
        Generate cross-section Figure.

    Notes
    -----
    - Slices: 'vertical' (fixed lat/lon), 'horizontal' (fixed depth).
    - Profiles: Along line from start to end point; 1D extract.
    - Overlays: kwargs['overlays'] list of data or scalars (e.g., ['model', 'uncertainty']).
    - Projection: Simple Cartesian; for geospatial, use line in lat/lon/depth.
    - Units: Auto-label from metadata.

    Examples
    --------
    >>> generator = CrossSectionGenerator()
    >>> fig = generator.generate(
    ...     inversion_results,
    ...     slice_type='vertical',
    ...     fixed_lat=40.0,
    ...     overlays=['model', 'uncertainty'],
    ...     title='Vertical Cross-Section'
    ... )
    >>> fig.savefig('cross_section.png')
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def generate(
        self,
        data: Union[InversionResults, ProcessedGrid],
        **kwargs: Any
    ) -> Figure:
        """
        Generate 2D cross-section or 1D profile from 3D data.

        Parameters
        ----------
        data : Union[InversionResults, ProcessedGrid]
            3D data for slicing.
        **kwargs : dict, optional
            - 'slice_type': str ('vertical', 'horizontal', 'profile'), default 'vertical'.
            - 'fixed_lat'/'fixed_lon'/'fixed_depth': float, for slices.
            - 'line_start'/'line_end': Tuple[float, float, float], (lon, lat, depth) for profiles.
            - 'overlays': List[str], scalars to plot (default ['model']).
            - 'cmap': str, colormap (default 'viridis').
            - 'title': str, figure title (default auto).

        Returns
        -------
        Figure
            Matplotlib Figure with slice/profile plot(s).

        Raises
        ------
        ValueError
            If non-3D data or invalid slice params.
        """
        if not isinstance(data, (InversionResults, ProcessedGrid)):
            raise ValueError("CrossSectionGenerator supports only InversionResults or 3D ProcessedGrid")

        if isinstance(data, ProcessedGrid) and len(data.ds.dims) < 3:
            raise ValueError("Data must be 3D for cross-sections")

        if isinstance(data, InversionResults):
            model = data.model
            metadata = data.metadata
        else:
            model = data.ds['data'].values
            metadata = dict(data.ds.attrs)

        coords = _extract_coords(data)
        lons, lats, depths = coords['lons'], coords['lats'], coords['depths']
        lon_g, lat_g, depth_g = np.meshgrid(lons, lats, depths, indexing='ij')

        slice_type = kwargs.get('slice_type', 'vertical')
        overlays = kwargs.get('overlays', ['model'])
        cmap = kwargs.get('cmap', 'viridis')
        title = kwargs.get('title', f"{slice_type.capitalize()} Cross-Section")

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title)

        if slice_type == 'vertical':
            fixed_lat = kwargs.get('fixed_lat')
            if fixed_lat is None:
                fixed_lat = lats[len(lats)//2]  # Middle
            idx_lat = np.argmin(np.abs(lats - fixed_lat))
            slice_data = model[:, idx_lat, :]  # (lon, depth)
            ax.imshow(slice_data.T, extent=[lons[0], lons[-1], depths[-1], depths[0]], cmap=cmap, origin='lower')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Depth (m)')
            logger.info(f"Vertical slice at lat={fixed_lat}")

        elif slice_type == 'horizontal':
            fixed_depth = kwargs.get('fixed_depth', depths[0])
            idx_depth = np.argmin(np.abs(depths - fixed_depth))
            slice_data = model[:, :, idx_depth]  # (lat, lon)
            im = ax.contourf(lats, lons, slice_data, cmap=cmap)
            plt.colorbar(im, ax=ax, label='Model Value')
            ax.set_xlabel('Latitude')
            ax.set_ylabel('Longitude')
            logger.info(f"Horizontal slice at depth={fixed_depth}")

        elif slice_type == 'profile':
            line_start = kwargs.get('line_start', (lons[0], lats[0], depths[0]))
            line_end = kwargs.get('line_end', (lons[-1], lats[-1], depths[0]))
            # Simple linear interp along line (assume horizontal profile at fixed depth)
            n_points = 100
            t = np.linspace(0, 1, n_points)
            profile_lon = line_start[0] + t * (line_end[0] - line_start[0])
            profile_lat = line_start[1] + t * (line_end[1] - line_start[1])
            profile_depth = line_start[2] + t * (line_end[2] - line_start[2])  # If vertical

            # Interp model along path (simplified 1D; use scipy.interpolate for irregular)
            profile_values = []
            for i in range(n_points):
                idx_lon = np.argmin(np.abs(lons - profile_lon[i]))
                idx_lat = np.argmin(np.abs(lats - profile_lat[i]))
                idx_depth = np.argmin(np.abs(depths - profile_depth[i]))
                profile_values.append(model[idx_lat, idx_lon, idx_depth])
            distance = t * np.linalg.norm(np.array(line_end) - np.array(line_start))
            ax.plot(distance, profile_values, label='Profile')
            ax.set_xlabel('Distance along line')
            ax.set_ylabel('Model Value')
            logger.info(f"Profile from {line_start} to {line_end}")

        else:
            raise ValueError(f"Unknown slice_type: {slice_type}")

        # Overlays
        for overlay in overlays:
            if overlay == 'uncertainty' and 'uncertainty' in locals():
                # Plot uncertainty contour or line
                if slice_type == 'vertical':
                    unc_slice = uncertainty[:, idx_lat, :]
                    ax.contour(lons, depths[::-1], unc_slice.T, colors='gray', linestyles='--', alpha=0.5)
                ax.legend()

        plt.tight_layout()
        logger.info(f"Generated {slice_type} cross-section")
        return fig