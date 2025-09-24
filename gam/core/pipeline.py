"""
Core Pipeline Module.

Orchestrates the full GAM workflow: ingestion -> preprocessing -> modeling -> visualization.
Supports parallel execution via Dask and configuration-driven execution.
Provides the high-level GAMPipeline class for end-to-end anomaly mapping.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import dask
from dask.distributed import Client

from .config import GAMConfig
from .exceptions import PipelineError
from ..ingestion.manager import IngestionManager
from ..preprocessing.manager import PreprocessingManager
from ..modeling.manager import ModelingManager
from ..visualization.manager import VisualizationManager
from .utils import validate_bbox
from .parallel import setup_dask_cluster

log = logging.getLogger(__name__)

@dataclass
class PipelineResults:
    """Data structure for pipeline outputs."""
    raw_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    inversion_results: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    visualizations: Dict[str, Path]

class GAMPipeline:
    """
    High-level pipeline orchestrator for GeoAnomalyMapper.

    Coordinates ingestion, preprocessing, modeling, and visualization stages.
    Supports bounding box or global tiled processing with optional caching and parallelism.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[GAMConfig] = None,
        use_dask: bool = True,
        n_workers: int = 4,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to config.yaml.
            config: GAMConfig object (overrides config_path).
            use_dask: Enable Dask for parallel processing.
            n_workers: Number of Dask workers.
            cache_dir: Directory for data caching.
        """
        self.config = config or GAMConfig.from_yaml(config_path)
        self.use_dask = use_dask
        self.client: Optional[Client] = None
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

        if use_dask:
            self.client = setup_dask_cluster(n_workers=n_workers)
            log.info(f"Dask client initialized with {n_workers} workers")

        # Initialize managers
        self.ingestion = IngestionManager(cache_dir=self.cache_dir)
        self.preprocessing = PreprocessingManager(config=self.config)
        self.modeling = ModelingManager(config=self.config)
        self.visualization = VisualizationManager(config=self.config)

        log.info("GAMPipeline initialized")

    def run_analysis(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        modalities: List[str] = None,
        output_dir: str = "./results",
        use_cache: bool = True,
        global_mode: bool = False,
        tiles: int = 10
    ) -> PipelineResults:
        """
        Run the full anomaly detection pipeline.

        Args:
            bbox: (min_lat, max_lat, min_lon, max_lon) for regional analysis.
            modalities: List of data types ['gravity', 'magnetic', 'seismic', 'insar'].
            output_dir: Directory to save results and visualizations.
            use_cache: Use cached data if available.
            global_mode: Process global data in tiles.
            tiles: Number of tiles for global processing.

        Returns:
            PipelineResults with all stage outputs.
        """
        if bbox:
            validate_bbox(bbox)
        if not modalities:
            modalities = self.config.modalities

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            # Stage 1: Ingestion
            log.info("Starting ingestion stage")
            raw_data = self.ingestion.fetch_multiple(
                modalities=modalities,
                bbox=bbox,
                use_cache=use_cache,
                global_mode=global_mode,
                tiles=tiles
            )

            # Stage 2: Preprocessing
            log.info("Starting preprocessing stage")
            processed_data = self.preprocessing.process(raw_data)

            # Stage 3: Modeling
            log.info("Starting modeling stage")
            inversion_results = self.modeling.invert(processed_data)

            # Stage 4: Anomaly Detection & Fusion
            log.info("Starting anomaly detection")
            anomalies = self.modeling.detect_anomalies(inversion_results)

            # Stage 5: Visualization
            log.info("Starting visualization stage")
            visualizations = self.visualization.generate(
                processed_data, inversion_results, anomalies, output_dir=output_dir
            )

            results = PipelineResults(
                raw_data=raw_data,
                processed_data=processed_data,
                inversion_results=inversion_results,
                anomalies=anomalies,
                visualizations=visualizations
            )

            log.info(f"Pipeline completed successfully. Detected {len(anomalies)} anomalies.")
            return results

        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline execution failed: {str(e)}") from e

    def close(self):
        """Close Dask client if active."""
        if self.client:
            self.client.close()
            log.info("Dask client closed")