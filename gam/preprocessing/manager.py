"""Main preprocessing manager for the GAM preprocessing module."""

from __future__ import annotations

import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from gam.core.exceptions import PreprocessingError
from gam.ingestion.data_structures import RawData
from gam.preprocessing.base import Preprocessor
from gam.preprocessing.data_structures import ProcessedGrid
from gam.preprocessing.processors import (
    GravityPreprocessor, MagneticPreprocessor, SeismicPreprocessor, InSARPreprocessor
)
from gam.preprocessing.parallel import DaskPreprocessor
from gam.preprocessing.filters import (
    NoiseFilter, BandpassFilter, OutlierFilter, SpatialFilter
)
from gam.preprocessing.gridding import RegularGridder, CoordinateAligner
from gam.preprocessing.units import unit_converter


logger = logging.getLogger(__name__)


class PreprocessingManager:
    """
    Coordinates all preprocessing steps and selects appropriate processors.

    Loads configuration from YAML, dispatches modality-specific processors,
    supports custom processing pipelines (e.g., ['filter', 'grid', 'units']).
    Integrates parallel processing and progress reporting.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml (default: 'config.yaml').

    Attributes
    ----------
    config : Dict[str, Any]
        Loaded and validated configuration.

    Methods
    -------
    _load_config(path: str) -> Dict[str, Any]
        Load and validate YAML config.
    get_processor(modality: str, parallel: bool = False, **kwargs) -> Preprocessor
        Factory for modality processor (wrapped if parallel).
    preprocess_data(data: RawData, modality: Optional[str] = None, pipeline: Optional[List[str]] = None, **kwargs) -> ProcessedGrid
        Main entry point: process data with full pipeline or custom steps.

    Notes
    -----
    - Config schema: 'preprocessing': {'grid_resolution': float, 'filters': List[str], ...}
    - Defaults applied if config missing keys.
    - Logs progress for each step/modality.
    - Supports environment-specific overrides (e.g., 'env': 'dev' in config).

    Examples
    --------
    >>> manager = PreprocessingManager('config.yaml')
    >>> grid = manager.preprocess_data(raw_data, modality='gravity', parallel=True)
    >>> # Custom pipeline
    >>> grid = manager.preprocess_data(raw_data, pipeline=['filter', 'grid'])
    """

    MODALITY_PROCESSORS = {
        'gravity': GravityPreprocessor,
        'magnetic': MagneticPreprocessor,
        'seismic': SeismicPreprocessor,
        'insar': InSARPreprocessor
    }

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load and validate configuration from YAML.

        Parameters
        ----------
        path : str
            Config file path.

        Returns
        -------
        Dict[str, Any]
            Validated config.

        Raises
        ------
        FileNotFoundError
            If config not found (uses defaults).
        yaml.YAMLError
            If invalid YAML.
        """
        defaults = {
            'preprocessing': {
                'grid_resolution': 0.1,
                'filters': ['noise', 'outlier'],
                'parallel_workers': -1
            }
        }
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            # Merge defaults
            for key, val in defaults.items():
                if key not in config:
                    config[key] = val
                else:
                    config[key].setdefault('grid_resolution', 0.1)
                    config[key].setdefault('filters', ['noise', 'outlier'])
                    config[key].setdefault('parallel_workers', -1)
            # Basic validation
            if not isinstance(config['preprocessing']['grid_resolution'], (int, float)) or config['preprocessing']['grid_resolution'] <= 0:
                logger.warning("Invalid grid_resolution; using default 0.1")
                config['preprocessing']['grid_resolution'] = 0.1
            logger.info(f"Loaded config from {path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config {path} not found; using defaults")
            return defaults
        except yaml.YAMLError as e:
            raise PreprocessingError(f"Invalid YAML config: {e}")

    def get_processor(
        self,
        modality: str,
        parallel: bool = False,
        **kwargs
    ) -> Preprocessor:
        """
        Factory method to get modality-specific processor.

        Parameters
        ----------
        modality : str
            'gravity', 'magnetic', 'seismic', or 'insar'.
        parallel : bool, optional
            Wrap with DaskPreprocessor (default: False).
        **kwargs : dict, optional
            Passed to processor init or DaskPreprocessor (e.g., chunk_sizes).

        Returns
        -------
        Preprocessor
            Selected/wrapped processor.

        Raises
        ------
        PreprocessingError
            If unknown modality.
        """
        if modality not in self.MODALITY_PROCESSORS:
            raise PreprocessingError(f"Unknown modality '{modality}'. Supported: {list(self.MODALITY_PROCESSORS.keys())}")

        processor_class = self.MODALITY_PROCESSORS[modality]
        processor = processor_class()

        if parallel:
            n_workers = kwargs.pop('n_workers', self.config['preprocessing']['parallel_workers'])
            processor = DaskPreprocessor(
                processor,
                chunk_sizes=kwargs.pop('chunk_sizes', None),
                n_workers=n_workers,
                **kwargs
            )

        logger.info(f"Selected {modality} processor (parallel: {parallel})")
        return processor

    def preprocess_data(
        self,
        data: RawData,
        modality: Optional[str] = None,
        pipeline: Optional[List[str]] = None,
        **kwargs
    ) -> ProcessedGrid:
        """
        Process data: full pipeline or custom steps.

        Parameters
        ----------
        data : RawData
            Input raw data.
        modality : str, optional
            Modality if not in data.metadata.
        pipeline : List[str], optional
            Custom steps: 'filter', 'grid', 'units', 'align' (default: full process).
        **kwargs : dict, optional
            Passed to processor/steps (e.g., grid_resolution, parallel=True).

        Returns
        -------
        ProcessedGrid
            Processed result.

        Raises
        ------
        PreprocessingError
            If modality unknown or step fails.
        """
        data.validate()
        if modality is None:
            modality = data.metadata.get('source', '').lower()
            if 'gravity' in modality:
                modality = 'gravity'
            elif 'magnetic' in modality:
                modality = 'magnetic'
            elif 'seismic' in modality:
                modality = 'seismic'
            elif 'insar' in modality:
                modality = 'insar'
            else:
                raise PreprocessingError(f"Could not infer modality from source '{data.metadata.get('source', 'unknown')}'")

        parallel = kwargs.pop('parallel', False)
        grid_res = kwargs.pop('grid_resolution', self.config['preprocessing']['grid_resolution'])

        logger.info(f"Starting preprocessing for {modality} data (parallel: {parallel}, res: {grid_res})")

        if pipeline is None:
            # Full process
            processor = self.get_processor(modality, parallel=parallel, **kwargs)
            result = processor.process(data, grid_resolution=grid_res, **kwargs)
        else:
            # Custom pipeline
            result = data
            processor_kwargs = {'grid_resolution': grid_res, **kwargs}

            for step in pipeline:
                logger.info(f"Pipeline step: {step}")
                if step == 'filter':
                    filters = kwargs.get('filters', self.config['preprocessing']['filters'])
                    for f_type in filters:
                        if f_type == 'noise':
                            result = NoiseFilter().apply(result)
                        elif f_type == 'outlier':
                            result = OutlierFilter().apply(result)
                        elif f_type == 'spatial':
                            result = SpatialFilter().apply(result)
                        elif f_type == 'bandpass' and modality == 'seismic':
                            result = BandpassFilter().apply(result)
                        else:
                            logger.warning(f"Unknown filter '{f_type}'; skipping")
                elif step == 'grid':
                    gridder = RegularGridder(resolution=processor_kwargs['grid_resolution'], **processor_kwargs)
                    result = gridder.apply(result)
                elif step == 'units':
                    to_unit = {'gravity': 'm/s²', 'magnetic': 'nT', 'seismic': 'm/s', 'insar': 'm'}.get(modality)
                    if to_unit:
                        if isinstance(result, ProcessedGrid):
                            result = unit_converter.convert_grid(result, to_unit, modality)
                        else:
                            # Assume RawData, convert values
                            from_unit = data.metadata.get('units', 'unknown')
                            factor = unit_converter.get_factor(modality, from_unit, to_unit)
                            result.values *= factor
                            data.metadata['units'] = to_unit
                elif step == 'align':
                    aligner = CoordinateAligner(target_crs='EPSG:4326', **processor_kwargs)
                    result = aligner.apply(result)
                else:
                    logger.warning(f"Unknown pipeline step '{step}'; skipping")

            # Finalize to ProcessedGrid if not already
            if not isinstance(result, ProcessedGrid):
                result = ProcessedGrid(result)

        logger.info(f"Preprocessing complete for {modality}: output shape {result.ds['data'].shape if hasattr(result, 'ds') else 'N/A'}")
        return result