"""Main visualization manager for GAM.

The VisualizationManager coordinates all visualization tasks, dispatching to
appropriate generators and exporters based on data type and configuration.
Supports map generation, data export, report creation, and batch processing.

Integrates configuration from config.yaml for styling, formats, and themes.
Provides a unified API for the visualization module.

Key features:
- Automatic dispatching: generate_map(data, type='auto') chooses visualizer.
- Export routing: export_data(data, format='auto') selects exporter.
- Report generation: Combines visualizations into PDF/HTML reports.
- Batch support: Process multiple datasets with common tasks.
- Config validation: Loads and applies visualization parameters.

Usage pattern:
>>> manager = VisualizationManager('config.yaml')
>>> map_fig = manager.generate_map(anomalies, map_type='interactive')
>>> manager.export_data(inversion_results, 'model.tif', format='geotiff')
>>> report = manager.create_report([grid, anomalies], 'report.pdf')

Notes
-----
- Config Schema (visualization section):
  - theme: str ('publication', default 'publication')
  - default_map_type: str ('2d_static')
  - export_formats: List[str] (default ['json', 'csv'])
  - dpi: int (default 300)
  - styling: Dict (passed to LayoutManager/ColorSchemes)
- Dependencies: yaml, all submodules.
- Error Handling: Logs warnings, raises for invalid config/data.
- Extensibility: Subclass for custom dispatchers.
"""

from __future__ import annotations

import logging
import yaml
from typing import Union, Any, List, Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from gam.visualization.base import Visualizer
from gam.visualization.maps_2d import StaticMapGenerator, InteractiveMapGenerator
from gam.visualization.volume_3d import VolumeRenderer, CrossSectionGenerator
from gam.visualization.plots import StatisticalPlots, ComparisonPlots, ProfilePlots
from gam.visualization.exporters import (
    GeoTIFFExporter, VTKExporter, DatabaseExporter, JSONExporter, CSVExporter
)
from gam.visualization.styling import LayoutManager, ColorSchemes, SymbolStyles
from gam.visualization.annotations import TextAnnotator, ShapeOverlay, ScaleIndicator
from gam.visualization.reports import ReportGenerator, HTMLReportGenerator  # Forward import

logger = logging.getLogger(__name__)


class VisualizationManager:
    """
    Central manager for GAM visualizations.

    Orchestrates generators, exporters, and reports based on data type and config.
    Loads configuration for consistent styling and defaults.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml (default 'config.yaml').

    Methods
    -------
    generate_map(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], map_type: str = 'auto', **kwargs) -> Union[Figure, folium.Map, pv.Plotter]
        Generate map/visualization.
    export_data(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], path: str, format: str = 'auto', **kwargs) -> str
        Export data to file.
    create_report(datasets: List, output_path: str, format: str = 'pdf', **kwargs) -> str
        Create comprehensive report.
    batch_process(datasets: List, tasks: List[str], **kwargs) -> Dict[str, Any]
        Process multiple datasets.

    Notes
    -----
    - Dispatch Logic: 'auto' chooses based on type (AnomalyOutput -> interactive map,
      InversionResults -> 3D volume, ProcessedGrid -> 2D static).
    - Config Integration: Applies theme, dpi, formats to all operations.
    - Batch: Parallel via joblib if 'parallel=True' in kwargs.
    - Returns: File paths for exports/reports; objects for generations.

    Examples
    --------
    >>> manager = VisualizationManager()
    >>> fig = manager.generate_map(processed_grid, map_type='2d_static')
    >>> manager.export_data(anomalies, 'output.csv', format='csv')
    >>> report_path = manager.create_report([grid, model], 'analysis.pdf')
    """

    _generator_map = {
        '2d_static': StaticMapGenerator,
        'interactive': InteractiveMapGenerator,
        '3d_volume': VolumeRenderer,
        'cross_section': CrossSectionGenerator,
        'stats': StatisticalPlots,
        'comparison': ComparisonPlots,
        'profile': ProfilePlots
    }

    _exporter_map = {
        'geotiff': GeoTIFFExporter,
        'vtk': VTKExporter,
        'database': DatabaseExporter,
        'json': JSONExporter,
        'csv': CSVExporter
    }

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.layout_manager = LayoutManager(dpi=self.config.get('dpi', 300), theme=self.config.get('theme', 'publication'))
        self.color_schemes = ColorSchemes()
        self.symbol_styles = SymbolStyles()
        self.text_annotator = TextAnnotator()
        self.shape_overlay = ShapeOverlay()
        self.scale_indicator = ScaleIndicator(units=self.config.get('units', 'km'))
        logger.info("VisualizationManager initialized with config")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate config."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            viz_config = config.get('visualization', {})
            # Validate keys
            required = ['theme', 'export_formats', 'default_map_type']
            for key in required:
                if key not in viz_config:
                    logger.warning(f"Missing config key '{key}'; using default")
                    viz_config[key] = 'publication' if key == 'theme' else ['json'] if key == 'export_formats' else '2d_static'
            return viz_config
        except FileNotFoundError:
            logger.warning(f"Config '{path}' not found; using defaults")
            return {'theme': 'publication', 'export_formats': ['json', 'csv'], 'default_map_type': '2d_static', 'dpi': 300}

    def generate_map(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        map_type: str = 'auto',
        **kwargs: Any
    ) -> Union[Figure, folium.Map, pv.Plotter]:
        """
        Generate appropriate map/visualization based on data and type.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data.
        map_type : str, optional
            Type ('2d_static', 'interactive', '3d_volume', etc.; default 'auto').
        **kwargs : dict, optional
            Passed to generator (projection, color_scheme, etc.).

        Returns
        -------
        Union[Figure, folium.Map, pv.Plotter]
            Generated visualization object.

        Raises
        ------
        ValueError
            If unknown map_type or data.
        """
        if map_type == 'auto':
            if isinstance(data, AnomalyOutput):
                map_type = 'interactive'
            elif isinstance(data, InversionResults):
                map_type = '3d_volume'
            elif isinstance(data, ProcessedGrid):
                map_type = self.config['default_map_type']
            else:
                raise ValueError(f"Unknown data type: {type(data)}")

        if map_type not in self._generator_map:
            raise ValueError(f"Unknown map_type '{map_type}'; available: {list(self._generator_map.keys())}")

        generator_cls = self._generator_map[map_type]
        generator = generator_cls()
        # Apply config
        kwargs.setdefault('color_scheme', self.config.get('color_scheme', 'RdBu_r'))
        kwargs.setdefault('theme', self.config['theme'])

        result = generator.generate(data, **kwargs)
        logger.info(f"Generated {map_type} map for {type(data).__name__}")
        return result

    def export_data(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        path: str,
        format: str = 'auto',
        **kwargs: Any
    ) -> str:
        """
        Export data using appropriate exporter.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data.
        path : str
            Output path (extension may determine format).
        format : str, optional
            ('geotiff', 'vtk', 'database', 'json', 'csv'; default 'auto').
        **kwargs : dict, optional
            Passed to exporter (compress, crs, etc.).

        Returns
        -------
        str
            Exported file path.

        Raises
        ------
        ValueError
            If unknown format or data.
        """
        if format == 'auto':
            if isinstance(data, (ProcessedGrid, InversionResults)):
                format = 'geotiff'
            elif isinstance(data, AnomalyOutput):
                format = 'csv'
            else:
                format = self.config['export_formats'][0]

        if format not in self._exporter_map:
            raise ValueError(f"Unknown format '{format}'; available: {list(self._exporter_map.keys())}")

        exporter_cls = self._exporter_map[format]
        exporter = exporter_cls()
        # Apply config
        if 'compress' not in kwargs:
            kwargs['compress'] = self.config.get('compress', True)
        if 'crs' not in kwargs:
            kwargs['crs'] = _extract_coords(data)['crs']  # From exporters

        exporter.export(data, path, **kwargs)
        logger.info(f"Exported {type(data).__name__} to {path} ({format})")
        return path

    def create_report(
        self,
        datasets: List[Union[ProcessedGrid, InversionResults, AnomalyOutput]],
        output_path: str,
        format: str = 'pdf',
        **kwargs: Any
    ) -> str:
        """
        Create comprehensive analysis report from datasets.

        Parameters
        ----------
        datasets : List
            List of data objects.
        output_path : str
            Output path (.pdf or .html).
        format : str, optional
            'pdf' (default) or 'html'.
        **kwargs : dict, optional
            - 'tasks': List[str], viz tasks ('map', 'stats', 'export').
            - 'template': str, report template (default 'standard').

        Returns
        -------
        str
            Report file path.

        Raises
        ------
        ValueError
            If invalid format.
        """
        if format == 'pdf':
            report_gen = ReportGenerator()
        elif format == 'html':
            report_gen = HTMLReportGenerator()
        else:
            raise ValueError(f"Unknown report format '{format}'")

        # Generate components
        figs = []
        for ds in datasets:
            if 'tasks' in kwargs:
                for task in kwargs['tasks']:
                    if task == 'map':
                        fig = self.generate_map(ds)
                        figs.append(fig)
                    elif task == 'stats':
                        fig = StatisticalPlots().generate(ds)
                        figs.append(fig)
            else:
                # Default: map + stats
                fig_map = self.generate_map(ds)
                figs.append(fig_map)
                fig_stats = StatisticalPlots().generate(ds)
                figs.append(fig_stats)

        report_path = report_gen.generate(figs, output_path, template=kwargs.get('template', 'standard'))
        logger.info(f"Created {format} report at {report_path}")
        return report_path

    def batch_process(
        self,
        datasets: List[Union[ProcessedGrid, InversionResults, AnomalyOutput]],
        tasks: List[str],
        parallel: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process multiple datasets with specified tasks.

        Parameters
        ----------
        datasets : List
            Data objects to process.
        tasks : List[str]
            Tasks ('generate_map', 'export_data', 'stats_plot').
        parallel : bool, optional
            Use joblib for parallel (default False).
        **kwargs : dict, optional
            Task-specific (e.g., map_type, format).

        Returns
        -------
        Dict[str, Any]
            Results dict {task: list of outputs}.
        """
        from joblib import Parallel, delayed  # Optional
        results = {task: [] for task in tasks}

        def process_single(ds, task, **task_kwargs):
            if task == 'generate_map':
                out = self.generate_map(ds, **task_kwargs)
            elif task == 'export_data':
                path = kwargs.get('output_dir', '.') / f"{task}_{len(results[task])}.ext"
                out = self.export_data(ds, str(path), **task_kwargs)
            elif task == 'stats_plot':
                out = StatisticalPlots().generate(ds, **task_kwargs)
            else:
                raise ValueError(f"Unknown task '{task}'")
            return out

        if parallel:
            parallel_backend = 'threading'
            with Parallel(n_jobs=-1, backend=parallel_backend) as p:
                for ds in datasets:
                    for task in tasks:
                        task_kwargs = kwargs.get(task, {})
                        result = p(delayed(process_single)(ds, task, **task_kwargs))
                        results[task].append(result)
        else:
            for ds in datasets:
                for task in tasks:
                    task_kwargs = kwargs.get(task, {})
                    result = process_single(ds, task, **task_kwargs)
                    results[task].append(result)

        logger.info(f"Batch processed {len(datasets)} datasets with tasks {tasks}")
        return results