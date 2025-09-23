"""Annotation and overlay tools for GAM visualization module.

This module provides utilities for adding text labels, geometric shapes, and scale
indicators to maps and plots. TextAnnotator handles titles, labels, and legends.
ShapeOverlay adds bounding boxes, circles, polygons, and grid lines for regions.
ScaleIndicator creates scale bars and north arrows for cartographic reference.

These tools enhance visualizations with informative elements while maintaining
publication quality. Use with matplotlib axes or PyGMT figures (via matplotlib backend).

Supported features:
- Text: Coordinate-formatted labels, anomaly values, custom legends.
- Shapes: Vector patches for ROIs, anomaly regions, grid overlays.
- Scales: Auto-calculated bars in km/degrees, north arrows.

Notes
-----
- Compatibility: Matplotlib primary; for PyGMT, use fig.text() equivalents.
- Formatting: Degrees symbol (°) for coords, scientific notation for values.
- Positioning: Relative (0-1) or absolute coords.
- Dependencies: matplotlib.patches, matplotlib.text, numpy.
- Customization: Props dicts for font/color/line styles.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)


class TextAnnotator:
    """
    Tool for adding text elements to plots and maps.

    Supports coordinate labels with formatting, anomaly labels with values,
    title addition, and legend generation from symbols/colors.

    Parameters
    ----------
    font_family : str, optional
        Font family (default: 'sans-serif').

    Methods
    -------
    add_title(fig: Figure, title: str, **props) -> Text
        Add figure suptitle.
    add_label(ax, pos: Tuple[float, float], text: str, **props) -> Text
        Add text label at position.
    add_coord_labels(ax, coords: Dict[str, np.ndarray], **props) -> List[Text]
        Add lat/lon tick labels.
    add_anomaly_labels(ax, anomalies: AnomalyOutput, **props) -> List[Text]
        Label points with confidence/type.
    add_legend(ax, handles: List, labels: List[str], **props) -> Legend
        Add custom legend.

    Notes
    -----
    - Positions: Data coords or axes fraction (0-1).
    - Formatting: '{:.2f}°' for coords, scientific for values.
    - Props: fontsize (default 10), color ('black'), ha/va ('center').
    - For PyGMT: Use fig.text(x, y, text, ...).

    Examples
    --------
    >>> annot = TextAnnotator()
    >>> annot.add_title(fig, 'Geophysical Anomaly Map', fontsize=14)
    >>> labels = annot.add_coord_labels(ax, {'lat': lats, 'lon': lons})
    >>> legend = annot.add_legend(ax, handles, ['Void', 'Fault'])
    """

    def __init__(self, font_family: str = 'sans-serif'):
        self.font_family = font_family

    def add_title(
        self,
        fig: plt.Figure,
        title: str,
        fontsize: int = 14,
        pad: float = 20,
        **props: Any
    ) -> plt.Text:
        """
        Add title to figure.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure.
        title : str
            Title text.
        fontsize : int, optional
            Font size (default 14).
        pad : float, optional
            Padding from top (default 20).
        **props : dict, optional
            Additional text props (color, weight).

        Returns
        -------
        Text
            Title text object.
        """
        props.setdefault('fontfamily', self.font_family)
        props.setdefault('color', 'black')
        props.setdefault('weight', 'bold')
        title_obj = fig.suptitle(title, fontsize=fontsize, y=0.98, **props)
        logger.debug(f"Added title: {title}")
        return title_obj

    def add_label(
        self,
        ax: plt.Axes,
        pos: Tuple[float, float],
        text: str,
        fontsize: int = 10,
        ha: str = 'center',
        va: str = 'center',
        **props: Any
    ) -> plt.Text:
        """
        Add text label to axes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes.
        pos : Tuple[float, float]
            Position (x, y) in data coords.
        text : str
            Label text.
        fontsize : int, optional
            Size (default 10).
        ha, va : str, optional
            Horizontal/vertical alignment (default 'center').
        **props : dict, optional
            Color, rotation, etc.

        Returns
        -------
        Text
            Label text object.
        """
        props.setdefault('fontfamily', self.font_family)
        props.setdefault('color', 'black')
        label = ax.text(pos[0], pos[1], text, fontsize=fontsize, ha=ha, va=va, transform=ax.transData, **props)
        logger.debug(f"Added label '{text}' at {pos}")
        return label

    def add_coord_labels(
        self,
        ax: plt.Axes,
        coords: Dict[str, np.ndarray],
        fontsize: int = 9,
        **props: Any
    ) -> List[plt.Text]:
        """
        Add coordinate labels to axes corners.

        Parameters
        ----------
        ax : Axes
            Axes for labels.
        coords : Dict[str, np.ndarray]
            {'lat': array, 'lon': array}.
        fontsize : int, optional
            Label size (default 9).
        **props : dict, optional
            Text props.

        Returns
        -------
        List[Text]
            Label objects.
        """
        if 'lat' not in coords or 'lon' not in coords:
            raise ValueError("Coords must include 'lat' and 'lon'")

        lat_min, lat_max = coords['lat'][[0, -1]]
        lon_min, lon_max = coords['lon'][[0, -1]]

        labels = [
            self.add_label(ax, (lon_min, lat_min), f"{lat_min:.2f}°N, {lon_min:.2f}°E", fontsize=fontsize, ha='left', va='bottom', **props),
            self.add_label(ax, (lon_max, lat_min), f"{lat_min:.2f}°N, {lon_max:.2f}°E", fontsize=fontsize, ha='right', va='bottom', **props),
            self.add_label(ax, (lon_min, lat_max), f"{lat_max:.2f}°N, {lon_min:.2f}°E", fontsize=fontsize, ha='left', va='top', **props),
            self.add_label(ax, (lon_max, lat_max), f"{lat_max:.2f}°N, {lon_max:.2f}°E", fontsize=fontsize, ha='right', va='top', **props)
        ]
        logger.info("Added coordinate labels to corners")
        return labels

    def add_anomaly_labels(
        self,
        ax: plt.Axes,
        anomalies: AnomalyOutput,
        fontsize: int = 8,
        fmt: str = "{type}: {conf:.2f}",
        **props: Any
    ) -> List[plt.Text]:
        """
        Add labels to anomaly points.

        Parameters
        ----------
        ax : Axes
            Axes for labels.
        anomalies : AnomalyOutput
            DataFrame with 'lat', 'lon', 'anomaly_type', 'confidence'.
        fontsize : int, optional
            Label size (default 8).
        fmt : str, optional
            Label format string (default "{type}: {conf:.2f}").
        **props : dict, optional
            Text props (color='red', alpha=0.8).

        Returns
        -------
        List[Text]
            Label objects.
        """
        labels = []
        for _, row in anomalies.iterrows():
            text = fmt.format(type=row['anomaly_type'], conf=row['confidence'])
            label = self.add_label(ax, (row['lon'], row['lat']), text, fontsize=fontsize, **props)
            labels.append(label)
        logger.info(f"Added {len(anomalies)} anomaly labels")
        return labels

    def add_legend(
        self,
        ax: plt.Axes,
        handles: List[Line2D],
        labels: List[str],
        loc: str = 'best',
        fontsize: int = 10,
        **props: Any
    ) -> plt.legend.Legend:
        """
        Add legend to axes.

        Parameters
        ----------
        ax : Axes
            Axes for legend.
        handles : List[Line2D]
            Legend handles.
        labels : List[str]
            Legend labels.
        loc : str, optional
            Location (default 'best').
        fontsize : int, optional
            Legend font size (default 10).
        **props : dict, optional
            Legend props (ncol, frameon).

        Returns
        -------
        Legend
            Legend object.
        """
        props.setdefault('fontfamily', self.font_family)
        props.setdefault('fontsize', fontsize)
        legend = ax.legend(handles, labels, loc=loc, **props)
        logger.debug(f"Added legend with {len(labels)} items")
        return legend


class ShapeOverlay:
    """
    Tool for adding geometric overlays to plots and maps.

    Supports bounding boxes for ROIs, circles/polygons for anomaly regions,
    and grid lines for coordinate reference. Uses matplotlib patches for vector
    rendering.

    Parameters
    ----------
    alpha : float, optional
        Default fill alpha (default 0.3).

    Methods
    -------
    add_bbox(ax: Axes, bbox: Tuple[float, float, float, float], **props) -> Rectangle
        Add bounding box.
    add_circle(ax: Axes, center: Tuple[float, float], radius: float, **props) -> Circle
        Add circle.
    add_polygon(ax: Axes, vertices: np.ndarray, **props) -> Polygon
        Add polygon.
    add_grid(ax: Axes, spacing: Tuple[float, float], **props) -> LineCollection
        Add grid lines.

    Notes
    -----
    - Bbox: (min_lon, min_lat, max_lon, max_lat).
    - Vertices: (n, 2) array for polygon.
    - Grid: Major/minor lines optional via spacing tuple.
    - Props: fill (default False), edgecolor ('red'), linewidth (1).
    - For PyGMT: Use plot with style='r' for rectangles, etc.

    Examples
    --------
    >>> overlay = ShapeOverlay()
    >>> bbox_patch = overlay.add_bbox(ax, (0, 0, 10, 10), fill=True, alpha=0.2, edgecolor='blue')
    >>> circle = overlay.add_circle(ax, (5, 5), radius=2, color='red')
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def add_bbox(
        self,
        ax: plt.Axes,
        bbox: Tuple[float, float, float, float],
        fill: bool = False,
        **props: Any
    ) -> plt.patches.Rectangle:
        """
        Add bounding box overlay.

        Parameters
        ----------
        ax : Axes
            Target axes.
        bbox : Tuple[float, float, float, float]
            (x_min, y_min, width, height) or (minx, miny, maxx, maxy).
        fill : bool, optional
            Fill interior (default False).
        **props : dict, optional
            facecolor ('none'), edgecolor ('red'), linewidth (1), alpha.

        Returns
        -------
        Rectangle
            Patch object.
        """
        if len(bbox) == 4:
            x, y, width, height = bbox
        else:
            x, y = bbox[0], bbox[1]
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        props.setdefault('facecolor', 'none' if not fill else 'lightblue')
        props.setdefault('edgecolor', 'red')
        props.setdefault('linewidth', 1)
        props.setdefault('alpha', self.alpha if fill else 1)
        rect = Rectangle((x, y), width, height, **props)
        ax.add_patch(rect)
        logger.debug(f"Added bbox at ({x}, {y}) size ({width}, {height})")
        return rect

    def add_circle(
        self,
        ax: plt.Axes,
        center: Tuple[float, float],
        radius: float,
        fill: bool = False,
        **props: Any
    ) -> plt.patches.Circle:
        """
        Add circle overlay.

        Parameters
        ----------
        ax : Axes
            Target axes.
        center : Tuple[float, float]
            (x, y) center.
        radius : float
            Radius.
        fill : bool, optional
            Fill (default False).
        **props : dict, optional
            facecolor, edgecolor ('blue'), linewidth.

        Returns
        -------
        Circle
            Patch object.
        """
        props.setdefault('facecolor', 'none' if not fill else 'yellow')
        props.setdefault('edgecolor', 'blue')
        props.setdefault('linewidth', 1)
        props.setdefault('alpha', self.alpha if fill else 1)
        circle = Circle(center, radius, **props)
        ax.add_patch(circle)
        logger.debug(f"Added circle at {center} r={radius}")
        return circle

    def add_polygon(
        self,
        ax: plt.Axes,
        vertices: np.ndarray,
        fill: bool = False,
        **props: Any
    ) -> plt.patches.Polygon:
        """
        Add polygon overlay.

        Parameters
        ----------
        ax : Axes
            Target axes.
        vertices : np.ndarray
            (n_vertices, 2) array of (x, y).
        fill : bool, optional
            Fill (default False).
        **props : dict, optional
            facecolor, edgecolor ('green').

        Returns
        -------
        Polygon
            Patch object.
        """
        if vertices.shape[1] != 2:
            raise ValueError("Vertices must be (n, 2)")
        props.setdefault('facecolor', 'none' if not fill else 'lightgreen')
        props.setdefault('edgecolor', 'green')
        props.setdefault('linewidth', 1)
        props.setdefault('alpha', self.alpha if fill else 1)
        poly = Polygon(vertices, closed=True, **props)
        ax.add_patch(poly)
        logger.debug(f"Added polygon with {len(vertices)} vertices")
        return poly

    def add_grid(
        self,
        ax: plt.Axes,
        spacing: Tuple[float, float] = (1.0, 1.0),
        **props: Any
    ) -> matplotlib.collections.LineCollection:
        """
        Add grid lines overlay.

        Parameters
        ----------
        ax : Axes
            Target axes.
        spacing : Tuple[float, float], optional
            (x_spacing, y_spacing) (default 1.0).
        **props : dict, optional
            color ('gray'), alpha (0.5), linewidth (0.5).

        Returns
        -------
        LineCollection
            Grid lines.
        """
        props.setdefault('colors', 'gray')
        props.setdefault('alpha', 0.5)
        props.setdefault('linewidth', 0.5)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_lines = np.arange(xlim[0], xlim[1], spacing[0])
        y_lines = np.arange(ylim[0], ylim[1], spacing[1])

        lines = []
        for x in x_lines:
            lines.append([(x, ylim[0]), (x, ylim[1])])
        for y in y_lines:
            lines.append([(xlim[0], y), (xlim[1], y)])

        lc = LineCollection(lines, **props)
        ax.add_collection(lc)
        logger.debug(f"Added grid with spacing {spacing}")
        return lc


class ScaleIndicator:
    """
    Tool for adding scale bars and north arrows to maps.

    Creates scale bars with automatic length based on extent, supporting km/degrees.
    North arrows as simple triangles or symbols.

    Parameters
    ----------
    units : str, optional
        Default units ('km' or 'degrees'; default 'km').

    Methods
    -------
    add_scale_bar(ax: Axes, length: Optional[float] = None, pos: Tuple[float, float] = (0.05, 0.05), **props) -> Tuple[Line2D, Text]
        Add scale bar.
    add_north_arrow(ax: Axes, pos: Tuple[float, float] = (0.95, 0.05), size: float = 0.1, **props) -> Polygon
        Add north arrow.

    Notes
    -----
    - Length: Auto 1/10 of x extent if None; convert degrees to km approx (111 km/deg).
    - Pos: Axes fraction (0-1).
    - North Arrow: Simple triangle; customizable with patches.
    - For PyGMT: Use basemapJ for scale, text 'N' for arrow.
    - Accuracy: Approx for non-UTM; use pyproj for precise.

    Examples
    --------
    >>> indicator = ScaleIndicator(units='km')
    >>> bar_line, bar_text = indicator.add_scale_bar(ax, length=5, pos=(0.1, 0.1))
    >>> arrow = indicator.add_north_arrow(ax, pos=(0.9, 0.1), size=0.05, color='black')
    """

    def __init__(self, units: str = 'km'):
        self.units = units

    def add_scale_bar(
        self,
        ax: plt.Axes,
        length: Optional[float] = None,
        pos: Tuple[float, float] = (0.05, 0.05),
        text_props: Dict[str, Any] = None,
        bar_props: Dict[str, Any] = None
    ) -> Tuple[plt.Line2D, plt.Text]:
        """
        Add scale bar to axes.

        Parameters
        ----------
        ax : Axes
            Target axes.
        length : float, optional
            Bar length in units (default auto 1/10 x extent).
        pos : Tuple[float, float], optional
            Bottom-left position (default (0.05, 0.05) fraction).
        text_props : Dict, optional
            Text props for label.
        bar_props : Dict, optional
            Line props for bar.

        Returns
        -------
        Tuple[Line2D, Text]
            Bar line and label text.
        """
        xlim = ax.get_xlim()
        if length is None:
            length = (xlim[1] - xlim[0]) / 10
            if self.units == 'km':
                length *= 111  # Approx deg to km

        x0, y0 = ax.transAxes.transform(pos)
        x1 = x0 + length / (xlim[1] - xlim[0])  # Normalize to axes

        bar_props = bar_props or {'color': 'black', 'linewidth': 3, 'solid_capstyle': 'butt'}
        bar = ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, **bar_props)[0]

        text_props = text_props or {'fontsize': 10, 'color': 'black', 'ha': 'center', 'va': 'bottom'}
        text = ax.text((x0 + x1)/2, y0 - 0.01, f"{length} {self.units}", transform=ax.transAxes, **text_props)

        logger.debug(f"Added scale bar of {length} {self.units} at {pos}")
        return bar, text

    def add_north_arrow(
        self,
        ax: plt.Axes,
        pos: Tuple[float, float] = (0.95, 0.05),
        size: float = 0.1,
        **props: Any
    ) -> plt.patches.Polygon:
        """
        Add north arrow to axes.

        Parameters
        ----------
        ax : Axes
            Target axes.
        pos : Tuple[float, float], optional
            Bottom position (default (0.95, 0.05) fraction).
        size : float, optional
            Arrow size (default 0.1 fraction).
        **props : dict, optional
            facecolor ('black'), edgecolor.

        Returns
        -------
        Polygon
            Arrow patch.
        """
        props.setdefault('facecolor', 'black')
        props.setdefault('edgecolor', 'black')
        props.setdefault('alpha', 1.0)

        # Simple triangle arrow pointing up
        x0, y0 = ax.transAxes.transform(pos)
        dy = size
        dx = dy / 2
        vertices = np.array([[x0, y0], [x0 - dx, y0 + dy], [x0 + dx, y0 + dy]])
        arrow = Polygon(vertices, transform=ax.transAxes, **props)
        ax.add_patch(arrow)

        # N label
        ax.text(x0, y0 + dy + 0.01, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')

        logger.debug(f"Added north arrow at {pos} size {size}")
        return arrow