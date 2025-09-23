"""Plotting utilities for GAM visualization module.

This module provides classes for statistical analysis plots, comparative visualizations,
and 1D profile plots. StatisticalPlots generates histograms, scatters, box plots, and
Q-Q plots for data distribution analysis. ComparisonPlots creates side-by-side views,
difference maps, and uncertainty plots. ProfilePlots extracts and visualizes data
along specified transects.

All classes subclass Visualizer and return matplotlib Figures for easy integration
and export. Designed for geophysical data exploration, model validation, and
publication-ready figures.

Supported inputs:
- ProcessedGrid: Flatten 'data' for stats, slice for comparisons/profiles.
- InversionResults: Use 'model' array.
- AnomalyOutput: Use columns like 'strength', 'confidence'.

Notes
-----
- Statistical: Handles 1D/2D data; 3D flattened.
- Comparisons: Requires multiple datasets via kwargs['datasets'] = [data1, data2].
- Profiles: Linear interpolation along line; assumes regular grid.
- Dependencies: matplotlib, scipy, numpy, pandas.
- Customization: kwargs for subplots, colormaps, labels.
- Performance: Vectorized; suitable for large datasets via hist binning.
"""

from __future__ import annotations

import logging
from typing import Union, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
import pandas as pd

from gam.visualization.base import Visualizer
from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

logger = logging.getLogger(__name__)


class StatisticalPlots(Visualizer):
    """
    Generator for statistical visualizations of geophysical data.

    Creates histograms of values/anomaly strengths, scatter plots for correlations,
    box plots for summaries, and Q-Q plots for distribution analysis. Supports
    single or multiple variables.

    Parameters
    ----------
    figsize : Tuple[int, int], optional
        Figure size for subplots (default: (12, 8)).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> Figure
        Generate statistical plots Figure.

    Notes
    -----
    - Plot Types: 'hist' (default), 'scatter', 'box', 'qq'; multiple via list in kwargs['types'].
    - For AnomalyOutput: Uses 'strength', 'confidence'; specify 'var' for others.
    - Bins: Auto (sqrt(n)) or kwargs['bins'].
    - KDE: Gaussian kernel for density in hist/scatter.
    - Q-Q: Against normal distribution; labels quantiles.
    - Handles NaNs: Drops before plotting.

    Examples
    --------
    >>> plots = StatisticalPlots()
    >>> fig = plots.generate(
    ...     anomalies,
    ...     types=['hist', 'qq'],
    ...     var='strength',
    ...     bins=30,
    ...     title='Anomaly Statistics'
    ... )
    >>> fig.savefig('stats.png')
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        **kwargs: Any
    ) -> Figure:
        """
        Generate statistical plots from data.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data for statistics.
        **kwargs : dict, optional
            - 'types': List[str] ('hist', 'scatter', 'box', 'qq'), default ['hist'].
            - 'var': str, column for AnomalyOutput (default 'strength').
            - 'bins': int or str, histogram bins (default 'auto').
            - 'scatter_var2': str, second var for scatter (default 'confidence').
            - 'title': str, figure title (default 'Statistical Plots').
            - 'n_cols': int, subplot columns (default 2).

        Returns
        -------
        Figure
            Matplotlib Figure with subplots.

        Raises
        ------
        ValueError
            If invalid plot type or missing variables.
        """
        types = kwargs.get('types', ['hist'])
        if isinstance(types, str):
            types = [types]
        var = kwargs.get('var', 'strength' if isinstance(data, AnomalyOutput) else 'data')
        title = kwargs.get('title', 'Statistical Plots')
        n_cols = kwargs.get('n_cols', 2)
        n_rows = (len(types) + n_cols - 1) // n_cols

        # Extract values
        if isinstance(data, (ProcessedGrid, InversionResults)):
            values = data.ds['data'].values.flatten() if isinstance(data, ProcessedGrid) else data.model.flatten()
            if var != 'data' and var != 'model':
                raise ValueError(f"Invalid var '{var}' for grid/model data")
        elif isinstance(data, AnomalyOutput):
            values = data[var].dropna().values
            if len(values) == 0:
                raise ValueError(f"No data in column '{var}'")
        else:
            raise ValueError("Unsupported data type")

        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_rows == 1 and n_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for i, plot_type in enumerate(types):
            if i >= len(axs):
                break
            ax = axs[i]

            if plot_type.lower() == 'hist':
                bins = kwargs.get('bins', 'auto')
                ax.hist(values, bins=bins, density=True, alpha=0.7, color='skyblue')
                # KDE
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', label='KDE')
                ax.set_xlabel(var)
                ax.set_ylabel('Density')
                ax.legend()

            elif plot_type.lower() == 'scatter':
                var2 = kwargs.get('scatter_var2', 'confidence' if isinstance(data, AnomalyOutput) else None)
                if var2 is None or isinstance(data, (ProcessedGrid, InversionResults)):
                    # Self-correlation or vs index
                    x = np.arange(len(values))
                    ax.scatter(x, values, alpha=0.6)
                    ax.set_xlabel('Index')
                else:
                    values2 = data[var2].dropna().values
                    min_len = min(len(values), len(values2))
                    ax.scatter(values[:min_len], values2[:min_len], alpha=0.6)
                    ax.set_xlabel(var)
                    ax.set_ylabel(var2)
                plt.figtext(0.5, 0.02, f"Correlation: {np.corrcoef(values, values2)[0,1]:.3f}" if var2 else '', ha='center')

            elif plot_type.lower() == 'box':
                if isinstance(data, AnomalyOutput) and len(data.columns) > 1:
                    box_data = [data[col].dropna().values for col in data.select_dtypes(include=[np.number]).columns[:4]]  # First 4 numeric
                else:
                    box_data = [values]
                ax.boxplot(box_data)
                ax.set_ylabel(var)
                ax.set_xticklabels(['Data'] if len(box_data) == 1 else data.select_dtypes(include=[np.number]).columns[:4].tolist())

            elif plot_type.lower() == 'qq':
                stats.probplot(values, dist="norm", plot=ax)
                ax.set_title('Q-Q Plot vs Normal')
                ax.get_lines()[0].set_markerfacecolor('r')
                ax.get_lines()[1].set_color('b')

            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for j in range(len(types), len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()
        logger.info(f"Generated statistical plots: {types} for {len(values)} values")
        return fig


class ComparisonPlots(Visualizer):
    """
    Generator for comparative visualizations between datasets or models.

    Creates side-by-side maps for before/after, difference maps for residuals,
    and uncertainty plots with error bars. Supports two or more datasets.

    Parameters
    ----------
    n_cols : int, optional
        Subplot columns (default: 2 for side-by-side).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> Figure
        Generate comparison plots Figure.

    Notes
    -----
    - Datasets: kwargs['datasets'] = [data1, data2]; or data as tuple.
    - Types: 'side_by_side' (default), 'difference', 'uncertainty'.
    - For maps: Contourf for each; difference as (data1 - data2).
    - Uncertainty: Error bars if 'uncertainty' present; else std dev.
    - Labels: Auto from metadata (e.g., 'Observed vs Modeled').

    Examples
    --------
    >>> plots = ComparisonPlots()
    >>> fig = plots.generate(
    ...     (processed_grid, inversion_results),
    ...     type='side_by_side',
    ...     titles=['Observed', 'Modeled'],
    ...     cmap='RdBu_r'
    ... )
    >>> fig.savefig('comparison.png')
    """

    def __init__(self, n_cols: int = 2):
        self.n_cols = n_cols

    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput, Tuple],
        **kwargs: Any
    ) -> Figure:
        """
        Generate comparative plots.

        Parameters
        ----------
        data : Union[Tuple, ...]
            Single or tuple of datasets for comparison.
        **kwargs : dict, optional
            - 'datasets': List, if data single (default [data]).
            - 'type': str ('side_by_side', 'difference', 'uncertainty'), default 'side_by_side'.
            - 'titles': List[str], subplot titles (default auto).
            - 'cmap': str, colormap (default 'RdBu_r').
            - 'n_rows': int, subplot rows (default 1).

        Returns
        -------
        Figure
            Matplotlib Figure with comparison subplots.

        Raises
        ------
        ValueError
            If <2 datasets or incompatible shapes.
        """
        if not isinstance(data, (list, tuple)):
            datasets = [data]
        else:
            datasets = list(data)
        if len(datasets) < 2:
            raise ValueError("Comparison requires at least 2 datasets")

        plot_type = kwargs.get('type', 'side_by_side').lower()
        titles = kwargs.get('titles', [f"Dataset {i+1}" for i in range(len(datasets))])
        cmap = kwargs.get('cmap', 'RdBu_r')
        n_rows = kwargs.get('n_rows', 1)
        n_cols = self.n_cols if plot_type != 'side_by_side' else len(datasets)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        if n_rows * n_cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        if plot_type == 'side_by_side':
            for i, ds in enumerate(datasets):
                if i >= len(axs):
                    break
                ax = axs[i]
                if isinstance(ds, (ProcessedGrid, InversionResults)):
                    # Flatten or slice for imshow/contour
                    values = ds.ds['data'].values.flatten() if isinstance(ds, ProcessedGrid) else ds.model.flatten()
                    im = ax.imshow(values.reshape(-1, len(ds.ds['lon']) if isinstance(ds, ProcessedGrid) else ds.model.shape[1]), cmap=cmap)
                    plt.colorbar(im, ax=ax)
                elif isinstance(ds, AnomalyOutput):
                    ax.scatter(ds['lon'], ds['lat'], c=ds['strength'], cmap=cmap)
                ax.set_title(titles[i] if i < len(titles) else f"Dataset {i+1}")

        elif plot_type == 'difference':
            if len(datasets) != 2:
                raise ValueError("Difference requires exactly 2 datasets")
            ds1, ds2 = datasets
            # Assume same shape; compute diff
            if isinstance(ds1, (ProcessedGrid, InversionResults)) and isinstance(ds2, type(ds1)):
                diff = ds1.ds['data'].values - ds2.ds['data'].values if isinstance(ds1, ProcessedGrid) else ds1.model - ds2.model
                im = axs[0].imshow(diff, cmap='RdBu_r')
                plt.colorbar(im, ax=axs[0])
                axs[0].set_title('Difference (Data1 - Data2)')
            else:
                raise ValueError("Difference supports matching grid/model types")
            if len(axs) > 1:
                axs[1].hist(diff.flatten(), bins=50, alpha=0.7)
                axs[1].set_title('Difference Histogram')

        elif plot_type == 'uncertainty':
            for i, ds in enumerate(datasets):
                if i >= len(axs):
                    break
                ax = axs[i]
                if hasattr(ds, 'uncertainty') and ds.uncertainty is not None:
                    unc = ds.uncertainty.flatten()
                    # Error bars: Assume mean profile or scatter with yerr
                    x = np.arange(len(unc))
                    ax.errorbar(x, unc, yerr=np.std(unc), fmt='o-', capsize=5, label='Uncertainty')
                else:
                    # Compute std if no uncertainty
                    values = ds.ds['data'].values.flatten() if isinstance(ds, ProcessedGrid) else ds.model.flatten()
                    ax.hist(values, bins=50, alpha=0.7, label='Data Distribution')
                    ax.set_title(f"Uncertainty/Std for {titles[i] if i < len(titles) else i+1}")
                ax.legend()
                ax.grid(True)

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # Hide extra
        for j in range(len(datasets), len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(kwargs.get('title', f'Comparison Plots ({plot_type})'))
        plt.tight_layout()
        logger.info(f"Generated {plot_type} comparison for {len(datasets)} datasets")
        return fig


class ProfilePlots(Visualizer):
    """
    Generator for 1D profiles and cross-sections along specified transects.

    Extracts data along lines (transects) from 3D models or 2D grids, supporting
    multiple overlays for comparison. Generates line plots with distance on x-axis.

    Parameters
    ----------
    figsize : Tuple[int, int], optional
        Figure size (default: (10, 6)).

    Methods
    -------
    generate(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], **kwargs) -> Figure
        Generate profile plot Figure.

    Notes
    -----
    - Transects: Straight lines defined by start/end points (lon, lat, depth).
    - Extraction: Nearest-neighbor or linear interp along path.
    - Overlays: Multiple lines or datasets on same plot.
    - For AnomalyOutput: Interpolate points along line.
    - Distance: Euclidean in coord space; for geospatial, approximate great-circle.
    - Units: Label x as 'Distance (km)', y from metadata.

    Examples
    --------
    >>> plots = ProfilePlots()
    >>> fig = plots.generate(
    ...     inversion_results,
    ...     line_start=(0, 0, 0),
    ...     line_end=(10, 10, 0),
    ...     overlays=['model'],
    ...     title='Transect Profile'
    ... )
    >>> fig.savefig('profile.png')
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def generate(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        **kwargs: Any
    ) -> Figure:
        """
        Generate 1D profile plot along transect.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data for profile.
        **kwargs : dict, optional
            - 'line_start': Tuple[float, float, float], start (lon, lat, depth), default data bounds.
            - 'line_end': Tuple[float, float, float], end point.
            - 'overlays': List[str], scalars (default ['model' or 'data']).
            - 'n_points': int, interpolation points (default 100).
            - 'title': str, plot title (default 'Profile Plot').
            - 'label': str, line label (default auto).

        Returns
        -------
        Figure
            Matplotlib Figure with profile line plot.

        Raises
        ------
        ValueError
            If missing line points or non-spatial data.
        """
        line_start = kwargs.get('line_start')
        line_end = kwargs.get('line_end')
        if line_start is None or line_end is None:
            # Default transect across data
            coords = _extract_coords(data)
            line_start = (coords['lons'][0], coords['lats'][0], 0)
            line_end = (coords['lons'][-1], coords['lats'][-1], 0)
            logger.info("Using default transect across data bounds")

        n_points = kwargs.get('n_points', 100)
        t = np.linspace(0, 1, n_points)
        profile_lon = line_start[0] + t * (line_end[0] - line_start[0])
        profile_lat = line_start[1] + t * (line_end[1] - line_start[1])
        profile_depth = line_start[2] + t * (line_end[2] - line_start[2])

        # Distance (approximate Euclidean; for geo, use haversine)
        distance = t * np.sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2 + (line_end[2] - line_start[2])**2) * 111  # Approx km

        fig, ax = plt.subplots(figsize=self.figsize)
        overlays = kwargs.get('overlays', ['data' if isinstance(data, ProcessedGrid) else 'model'])
        title = kwargs.get('title', 'Transect Profile')
        ax.set_title(title)

        for overlay in overlays:
            if overlay == 'data' and isinstance(data, ProcessedGrid):
                values = data.ds['data'].values
            elif overlay == 'model' and isinstance(data, InversionResults):
                values = data.model
            else:
                values = data.ds['data'].values if isinstance(data, ProcessedGrid) else data.model

            # Extract along profile (nearest neighbor)
            profile_values = []
            for i in range(n_points):
                idx_lon = np.argmin(np.abs(_extract_coords(data)['lons'] - profile_lon[i]))
                idx_lat = np.argmin(np.abs(_extract_coords(data)['lats'] - profile_lat[i]))
                idx_depth = np.argmin(np.abs(_extract_coords(data)['depths'] - profile_depth[i]))
                profile_values.append(values[idx_lat, idx_lon, idx_depth] if len(values.shape) == 3 else values[idx_lat, idx_lon])
            profile_values = np.array(profile_values)

            label = kwargs.get('label', overlay)
            ax.plot(distance, profile_values, label=label, linewidth=2)

        ax.set_xlabel('Distance along transect (km)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        logger.info(f"Generated profile along {line_start} to {line_end} with overlays {overlays}")
        return fig