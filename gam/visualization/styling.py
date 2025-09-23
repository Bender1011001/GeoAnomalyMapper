"""Map styling utilities for GAM visualization module.

This module defines scientific color schemes, symbol styles for point data, and
layout management for consistent, publication-ready figures. ColorSchemes provides
geophysical-appropriate colormaps. SymbolStyles handles markers, sizing, and coloring
for anomalies. LayoutManager creates multi-panel layouts with shared colorbars and
uniform styling.

These utilities ensure visual consistency across maps, plots, and reports.
Import and use in visualizers: e.g., cmap = ColorSchemes.get_cmap('gravity').

Supported color schemes:
- Gravity: Blue-white-red diverging for anomalies.
- Magnetic: Rainbow for total intensity fields.
- Seismic: Blue-red for velocity contrasts.
- Anomalies: Red-yellow-white for confidence levels.

Symbol styles:
- Shapes by type (circle for voids, triangle for faults).
- Size scaled by confidence/strength.
- Colors from schemes.

Layouts:
- Flexible grids with shared axes/colorbars.
- Publication formatting (fonts, spacing, DPI).

Notes
-----
- Colormaps: Perceptually uniform using LinearSegmentedColormap; avoid rainbow for data.
- Symbols: Matplotlib markers; scalable for density.
- Layouts: Gridspec for unequal panels; rcParams for global consistency.
- Dependencies: matplotlib.colors.
- Customization: Extend dicts for new types/schemes.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import rcParams
import numpy as np

logger = logging.getLogger(__name__)


class ColorSchemes:
    """
    Registry for scientific color maps tailored to geophysical data types.

    Defines diverging/sequential colormaps for different modalities and results.
    Uses LinearSegmentedColormap for smooth gradients. Get via get_cmap(type).

    Parameters
    ----------
    None : Static class; instantiate for custom extensions.

    Methods
    -------
    get_cmap(data_type: str, reverse: bool = False) -> Colormap
        Retrieve colormap for data type.
    register_scheme(name: str, cmap: Colormap) -> None
        Add custom scheme.

    Notes
    -----
    - Schemes: 'gravity' (blue-white-red diverging), 'magnetic' (rainbow sequential),
      'seismic' (blue-red velocity), 'anomalies' (red-yellow-white confidence).
    - Reverse: For inverted scales (e.g., negative anomalies).
    - Perceptual: Designed for colorblind accessibility and print (CMYK-safe).
    - Usage: In visualizers, im = ax.imshow(data, cmap=ColorSchemes.get_cmap('gravity')).

    Examples
    --------
    >>> schemes = ColorSchemes()
    >>> cmap = schemes.get_cmap('gravity')
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(data, cmap=cmap)
    >>> plt.colorbar(im)
    """

    _schemes: Dict[str, LinearSegmentedColormap] = {}

    def __init__(self):
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in geophysical color schemes."""
        # Gravity: Blue-white-red diverging (like RdBu_r but extended)
        gravity_cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 0.5)],
            'green': [(0.0, 0.0, 0.5), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 0.7), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        self._schemes['gravity'] = LinearSegmentedColormap('gravity_div', gravity_cdict)

        # Magnetic: Rainbow for total intensity (spectral)
        self._schemes['magnetic'] = plt.cm.get_cmap('hsv')  # Rainbow-like

        # Seismic: Blue-red for velocity (coolwarm)
        seismic_cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 0.5)],
            'green': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        self._schemes['seismic'] = LinearSegmentedColormap('seismic_vel', seismic_cdict)

        # Anomalies: Red-yellow-white for confidence (hot_r inverted)
        anomaly_cdict = {
            'red': [(0.0, 1.0, 1.0), (0.5, 1.0, 0.5), (1.0, 1.0, 0.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'blue': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)]
        }
        self._schemes['anomalies'] = LinearSegmentedColormap('anomaly_conf', anomaly_cdict)

    def get_cmap(self, data_type: str, reverse: bool = False) -> LinearSegmentedColormap:
        """
        Get colormap for data type.

        Parameters
        ----------
        data_type : str
            Type: 'gravity', 'magnetic', 'seismic', 'anomalies' (case-insensitive).
        reverse : bool, optional
            Reverse colormap (default False).

        Returns
        -------
        LinearSegmentedColormap
            Matplotlib colormap.

        Raises
        ------
        ValueError
            If unknown data_type.
        """
        key = data_type.lower()
        if key not in self._schemes:
            raise ValueError(f"Unknown data_type '{data_type}'; available: {list(self._schemes.keys())}")
        cmap = self._schemes[key]
        if reverse:
            cmap = cmap.reversed()
        return cmap

    def register_scheme(self, name: str, red: List[Tuple[float, float, float]], green: List[Tuple[float, float, float]], blue: List[Tuple[float, float, float]]) -> None:
        """
        Register custom colormap.

        Parameters
        ----------
        name : str
            Scheme name.
        red, green, blue : List[Tuple[float, float, float]]
            Color dictionary segments (position, low, high).
        """
        cdict = {'red': red, 'green': green, 'blue': blue}
        self._schemes[name.lower()] = LinearSegmentedColormap(name, cdict)
        logger.info(f"Registered custom scheme '{name}'")


class SymbolStyles:
    """
    Styles for point data symbols in maps and plots.

    Defines marker shapes by anomaly type, size scaling by confidence/strength,
    and color coding via ColorSchemes. Ensures consistent, scalable symbols for
    density and publication.

    Parameters
    ----------
    base_size : float, optional
        Base marker size (default: 10).

    Methods
    -------
    get_style(anomaly_type: str, confidence: float, strength: float, **kwargs) -> Dict[str, Any]
        Get marker, size, color for point.

    Notes
    -----
    - Shapes: 'o' (void), '^' (fault), 's' (density), 'D' (other).
    - Size: base_size * (0.5 + confidence) * (1 + |strength|/max_strength).
    - Color: From ColorSchemes.anomalies, normalized by confidence.
    - Edge: Black outline for visibility.
    - Usage: In plotting, plt.scatter(..., marker=style['marker'], s=style['size'], c=style['color']).

    Examples
    --------
    >>> styles = SymbolStyles()
    >>> style = styles.get_style('void', confidence=0.8, strength=2.5)
    >>> plt.scatter(lons, lats, marker=style['marker'], s=style['size'], c=style['color'], edgecolors='black')
    """

    _shapes: Dict[str, str] = {
        'void': 'o',  # Circle
        'fault': '^',  # Triangle up
        'density': 's',  # Square
        'default': 'D'  # Diamond
    }

    def __init__(self, base_size: float = 10.0):
        self.base_size = base_size
        self.color_schemes = ColorSchemes()

    def get_style(
        self,
        anomaly_type: str,
        confidence: float,
        strength: float,
        max_strength: float = 5.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get styling dict for anomaly point.

        Parameters
        ----------
        anomaly_type : str
            Type for shape ('void', 'fault', etc.).
        confidence : float
            0-1 for size/color scaling.
        strength : float
            Magnitude for size scaling.
        max_strength : float, optional
            Normalize strength (default 5.0).
        **kwargs : dict, optional
            - 'color_scheme': str, override 'anomalies'.

        Returns
        -------
        Dict[str, Any]
            {'marker': str, 'size': float, 'color': str or Tuple, 'edgecolor': str}.

        Raises
        ------
        ValueError
            If confidence not in [0,1].
        """
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")

        marker = self._shapes.get(anomaly_type.lower(), self._shapes['default'])

        # Size scaling
        size_scale = (0.5 + confidence) * (1 + abs(strength) / max_strength)
        size = self.base_size * size_scale

        # Color from scheme
        scheme = kwargs.get('color_scheme', 'anomalies')
        cmap = self.color_schemes.get_cmap(scheme)
        norm_conf = confidence  # Already 0-1
        color = cmap(norm_conf)

        return {
            'marker': marker,
            'size': size,
            'color': color,
            'edgecolor': 'black',
            'linewidth': 0.5
        }


class LayoutManager:
    """
    Manager for multi-panel figure layouts with consistent styling.

    Creates subplots with flexible grids, shared colorbars/axes, and applies
    publication-ready formatting (fonts, spacing, DPI). Supports unequal panel
    sizes and composition.

    Parameters
    ----------
    dpi : int, optional
        Figure DPI (default: 300).
    theme : str, optional
        'publication' (sans-serif, tight) or 'presentation' (larger fonts).

    Methods
    -------
    create_layout(nrows: int, ncols: int, **kwargs) -> Tuple[Figure, ndarray]
        Create figure and axes array.
    apply_styles(axs: ndarray, theme: str = 'publication') -> None
        Apply consistent styling to axes.
    add_shared_colorbar(fig: Figure, im: Any, axs: ndarray, **kwargs) -> Colorbar
        Add horizontal/vertical shared colorbar.

    Notes
    -----
    - Gridspec: Supports width_ratios, height_ratios for unequal panels.
    - Shared: cbar for all imshow/contourf; y/x shared for subplots.
    - Publication: Sets rcParams (font='Arial', size=10, tight_layout).
    - Composition: Multi-panel with suptitle, legend placement.
    - Usage: fig, axs = manager.create_layout(2, 2); manager.apply_styles(axs).

    Examples
    --------
    >>> manager = LayoutManager(dpi=300, theme='publication')
    >>> fig, axs = manager.create_layout(1, 3, width_ratios=[1, 1, 0.1])
    >>> manager.apply_styles(axs)
    >>> im1 = axs[0].imshow(data1, cmap='RdBu_r')
    >>> cbar = manager.add_shared_colorbar(fig, im1, axs)
    >>> fig.savefig('multi_panel.png', dpi=300, bbox_inches='tight')
    """

    def __init__(self, dpi: int = 300, theme: str = 'publication'):
        self.dpi = dpi
        self.theme = theme
        self._set_rcparams()

    def _set_rcparams(self) -> None:
        """Set matplotlib rcParams for theme."""
        if self.theme == 'publication':
            rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            })
        elif self.theme == 'presentation':
            rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
        logger.debug(f"Applied theme '{self.theme}'")

    def create_layout(
        self,
        nrows: int,
        ncols: int,
        width_ratios: Optional[List[float]] = None,
        height_ratios: Optional[List[float]] = None,
        **fig_kw: Any
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create figure with subplot layout.

        Parameters
        ----------
        nrows, ncols : int
            Grid dimensions.
        width_ratios : List[float], optional
            Column widths (default equal).
        height_ratios : List[float], optional
            Row heights (default equal).
        **fig_kw : dict, optional
            Passed to plt.subplots (e.g., figsize=(12,8)).

        Returns
        -------
        Tuple[Figure, ndarray]
            fig, axs (flattened if >1).
        """
        if width_ratios is None:
            width_ratios = [1] * ncols
        if height_ratios is None:
            height_ratios = [1] * nrows

        fig, axs = plt.subplots(
            nrows, ncols,
            figsize=fig_kw.get('figsize', (ncols*4, nrows*3)),
            gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios},
            squeeze=False
        )
        axs = axs.flatten()
        logger.info(f"Created {nrows}x{ncols} layout")
        return fig, axs

    def apply_styles(self, axs: np.ndarray, theme: Optional[str] = None) -> None:
        """
        Apply consistent styling to axes.

        Parameters
        ----------
        axs : ndarray
            Axes array.
        theme : str, optional
            Override global theme.
        """
        if theme:
            self.theme = theme
            self._set_rcparams()
        for ax in axs:
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('white')
            # Tight borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

    def add_shared_colorbar(
        self,
        fig: plt.Figure,
        im: Any,
        axs: np.ndarray,
        orientation: str = 'horizontal',
        **cbar_kw: Any
    ) -> plt.colorbar.Colorbar:
        """
        Add shared colorbar for subplots.

        Parameters
        ----------
        fig : Figure
            Parent figure.
        im : Any
            Image/QuadMesh from imshow/contourf.
        axs : ndarray
            Axes for cbar position.
        orientation : str, optional
            'horizontal' (default) or 'vertical'.
        **cbar_kw : dict, optional
            Passed to fig.colorbar (e.g., label='Value', shrink=0.8).

        Returns
        -------
        Colorbar
            Shared colorbar instance.
        """
        if orientation == 'horizontal':
            cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.046, pad=0.04, **cbar_kw)
        else:
            cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.046, pad=0.04, **cbar_kw)
        return cbar