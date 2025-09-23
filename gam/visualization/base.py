"""Abstract base class for visualization components in GAM."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union, Any

from matplotlib.figure import Figure

from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

logger = logging.getLogger(__name__)


class Visualizer(ABC):
    """
    Abstract base class for geophysical data visualizers in GeoAnomalyMapper (GAM).

    This class defines the interface for generating visualizations from processed
    geophysical data, inversion results, or detected anomalies. Subclasses implement
    specific visualization types (e.g., 2D maps, 3D volumes, statistical plots)
    while preserving coordinate systems, metadata, and scientific accuracy.

    All visualizations must handle geospatial coordinates (lat/lon/depth) and
    support publication-quality outputs with proper scaling, color mapping, and
    annotations. The generate method is the primary entry point, dispatching to
    modality-specific rendering logic.

    Parameters
    ----------
    None : Explicit constructor not required; subclasses may add init params
        (e.g., config: Dict[str, Any] for styling).

    Attributes
    ----------
    None : Subclasses may define (e.g., fig: Figure, config: Dict).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> Figure
        Generate a matplotlib Figure from input data.

    Notes
    -----
    - **Data Compatibility**: Input data must include spatial coordinates
      (lat/lon/depth) and metadata (units, CRS). Coordinate preservation is
      mandatory; use pyproj for transformations if needed.
    - **kwargs Handling**: Common kwargs include:
      - 'projection': str (e.g., 'Mercator', 'UTM'), default 'geographic'
      - 'color_scheme': str (e.g., 'blue-white-red'), from styling.py
      - 'dpi': int, default 300 for publication
      - 'title': str, optional
      - 'bbox': Tuple[float, float, float, float], for cropping
      Subclasses must document additional kwargs.
    - **Error Handling**: Raise ValueError for invalid data (e.g., missing coords),
      VisualizationError (custom) for rendering failures. Log warnings for
      degraded rendering (e.g., large datasets downsampled).
    - **Performance**: Support lazy evaluation for large data (xarray/Dask).
      Figures should be vector-based for scalability.
    - **Integration**: Used by VisualizationManager; outputs feed exporters
      and reports. Ensure thread-safety for parallel generation.
    - **Standards**: Follow geophysical conventions (e.g., positive depth down,
      WGS84 default CRS). All outputs publication-ready (CMYK colors, sans-serif
      fonts).

    Examples
    --------
    >>> # Abstract usage (subclass implementation)
    >>> from gam.visualization.maps_2d import StaticMapGenerator
    >>> visualizer = StaticMapGenerator()
    >>> fig = visualizer.generate(
    ...     data=anomalies,  # AnomalyOutput DataFrame
    ...     projection='Mercator',
    ...     color_scheme='red-yellow-white'
    ... )
    >>> fig.savefig('anomaly_map.png', dpi=300, bbox_inches='tight')
    >>> plt.close(fig)  # Memory management

    In manager:
    >>> viz_manager = VisualizationManager()
    >>> map_fig = viz_manager.generate_map(inversion_results, map_type='2d_static')
    """

    @abstractmethod
    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        **kwargs: Any
    ) -> Figure:
        """
        Generate a visualization Figure from geophysical data.

        Renders input data as a matplotlib Figure, applying appropriate
        projections, color schemes, and annotations based on data type and kwargs.
        Supports 2D/3D geophysical visualizations with metadata preservation.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data:
            - ProcessedGrid: Gridded observations (xarray.Dataset) for contour maps.
            - InversionResults: Subsurface models (dataclass) for volume/slices.
            - AnomalyOutput: Detected anomalies (pd.DataFrame) for point overlays.
        **kwargs : dict, optional
            Visualization parameters:
            - 'projection': str, map projection (default: 'geographic').
            - 'color_scheme': str, from ColorSchemes (default: data-specific).
            - 'dpi': int, figure resolution (default: 300).
            - 'title': str, plot title (default: auto-generated).
            - 'bbox': Tuple[float, float, float, float], spatial bounds.
            - Additional modality-specific (e.g., 'isosurface_level' for 3D).

        Returns
        -------
        Figure
            Matplotlib Figure instance ready for display/export. Includes
            axes with proper labels, colorbars, and legends.

        Raises
        ------
        ValueError
            If data lacks required spatial coordinates or metadata (e.g., no 'lat'/'lon').
        TypeError
            If data type unsupported by subclass.
        VisualizationError
            Custom error for rendering failures (e.g., invalid projection).

        Notes
        -----
        - **Coordinate Handling**: Automatically detects CRS from data.attrs;
          transforms if needed (via pyproj). Defaults to EPSG:4326 (WGS84).
        - **Metadata Integration**: Embeds units, processed_at, confidence in
          titles/legends. Preserves for exporters.
        - **Scalability**: For large data (>1M points), auto-downsample or chunk;
          log reduction factor.
        - **Thread Safety**: Method is reentrant; no shared state.

        Examples
        --------
        >>> # 2D map from processed grid
        >>> fig = visualizer.generate(
        ...     processed_grid,
        ...     projection='UTM',
        ...     color_scheme='blue-white-red'
        ... )
        >>> fig.show()

        >>> # 3D volume from inversion
        >>> fig = visualizer.generate(
        ...     inversion_results,
        ...     isosurface_level=0.5,
        ...     opacity=0.8
        ... )
        """
        pass