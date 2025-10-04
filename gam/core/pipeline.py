"""
Core Pipeline Module.

Orchestrates the full GAM workflow: ingestion -> preprocessing -> modeling -> visualization.
Provides the high-level GAMPipeline class for end-to-end anomaly mapping.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .exceptions import ConfigurationError, PipelineError
from ..ingestion.data_structures import RawData
from ..preprocessing.data_structures import ProcessedGrid
from ..modeling.data_structures import InversionResults


class GAMPipeline:
    """
    High-level pipeline orchestrator for GeoAnomalyMapper.

    Coordinates ingestion, preprocessing, modeling, and visualization stages.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration dictionary containing pipeline settings.

        Raises:
            ConfigurationError: If required config sections/keys are missing.
        """
        if "pipeline" not in config:
            raise ConfigurationError("Config must contain 'pipeline' section")

        pipeline_config = config["pipeline"]
        if "output_dir" not in pipeline_config:
            raise ConfigurationError("Pipeline config must contain 'output_dir'")

        self.config = config
        self.logger = logging.getLogger("gam." + __name__)

    def run_analysis(self, bbox: Tuple[float, ...], modalities: List[str]) -> Dict[str, str]:
        """
        Run the full anomaly detection pipeline.

        Args:
            bbox: Bounding box as (min_lat, max_lat, min_lon, max_lon).
            modalities: List of data modalities to process.

        Returns:
            Dictionary of artifact paths.

        Raises:
            PipelineError: If any stage fails.
        """
        self.logger.info("Starting analysis")

        try:
            raw_data = self._ingest(bbox, modalities)
            self.logger.debug("Ingestion completed")
        except Exception as e:
            raise PipelineError(f"Ingestion failed: {e}") from e

        try:
            grids = self._preprocess(raw_data)
            self.logger.debug("Preprocessing completed")
        except Exception as e:
            raise PipelineError(f"Preprocessing failed: {e}") from e

        try:
            results = self._model(grids)
            self.logger.debug("Modeling completed")
        except Exception as e:
            raise PipelineError(f"Modeling failed: {e}") from e

        try:
            anomalies = self._detect(results)
            self.logger.debug("Detection completed")
        except Exception as e:
            raise PipelineError(f"Detection failed: {e}") from e

        try:
            artifacts = self._visualize(anomalies)
            self.logger.debug("Visualization completed")
        except Exception as e:
            raise PipelineError(f"Visualization failed: {e}") from e

        return artifacts

    def _ingest(self, bbox: Tuple[float, ...], modalities: List[str]) -> List[RawData]:
        """
        Ingest raw data for the given bbox and modalities.

        Args:
            bbox: Bounding box.
            modalities: List of modalities.

        Returns:
            List of RawData objects.
        """
        raw_data_list = []
        np.random.seed(42)  # For reproducibility

        for modality in modalities:
            # Create dummy data based on bbox
            min_lat, max_lat, min_lon, max_lon = bbox
            lat_range = np.linspace(min_lat, max_lat, 10)
            lon_range = np.linspace(min_lon, max_lon, 10)
            data = np.random.rand(10, 10) * 100  # Dummy values

            metadata = {
                "modality": modality,
                "bbox": bbox,
                "resolution": 0.1
            }

            raw_data = RawData(data=data, metadata=metadata, crs="EPSG:4326")
            raw_data.validate()
            raw_data_list.append(raw_data)

        return raw_data_list

    def _preprocess(self, raw_data: List[RawData]) -> List[ProcessedGrid]:
        """
        Preprocess raw data into grids.

        Args:
            raw_data: List of RawData objects.

        Returns:
            List of ProcessedGrid objects.
        """
        grids = []
        for rd in raw_data:
            # Create grid from raw data
            lat = np.linspace(rd.metadata["bbox"][0], rd.metadata["bbox"][1], rd.data.shape[0])
            lon = np.linspace(rd.metadata["bbox"][2], rd.metadata["bbox"][3], rd.data.shape[1])

            grid_data = {
                'data': rd.data,
                'lat': lat,
                'lon': lon,
                'units': 'mGal' if rd.metadata["modality"] == "gravity" else "nT",
                'grid_resolution': rd.metadata["resolution"],
                'processing_params': {"method": "dummy"}
            }

            grid = ProcessedGrid(grid_data)
            grid.validate()
            grids.append(grid)

        return grids

    def _model(self, grids: List[ProcessedGrid]) -> InversionResults:
        """
        Perform modeling/inversion on grids.

        Args:
            grids: List of ProcessedGrid objects.

        Returns:
            InversionResults object.
        """
        # Combine grids (simple average for demo)
        combined_data = np.mean([g.ds['data'].values for g in grids], axis=0)
        combined_uncertainty = np.std([g.ds['data'].values for g in grids], axis=0)

        # Create xarray DataArrays
        import xarray as xr
        model_da = xr.DataArray(
            combined_data,
            coords={'lat': grids[0].ds.coords['lat'], 'lon': grids[0].ds.coords['lon']},
            attrs={'units': 'kg/m³'}
        )
        uncertainty_da = xr.DataArray(
            combined_uncertainty,
            coords=model_da.coords,
            attrs={'units': 'kg/m³'}
        )

        results = InversionResults(
            model=model_da,
            uncertainty=uncertainty_da,
            metadata={"method": "dummy_inversion"}
        )
        results.validate()
        return results

    def _detect(self, results: InversionResults) -> pd.DataFrame:
        """
        Detect anomalies from inversion results.

        Args:
            results: InversionResults object.

        Returns:
            DataFrame with anomaly detections.
        """
        # Get threshold from config
        threshold = self.config.get("anomaly_threshold", 95)  # percentile

        # Calculate anomaly scores (deviation from mean)
        data = results.model.values
        mean = np.mean(data)
        std = np.std(data)
        anomaly_scores = (data - mean) / std if std > 0 else np.zeros_like(data)

        # Percentile-based detection
        percentile_threshold = np.percentile(np.abs(anomaly_scores), threshold)
        anomalies_mask = np.abs(anomaly_scores) > percentile_threshold

        # Create DataFrame
        lats, lons = np.meshgrid(results.model.coords['lat'], results.model.coords['lon'], indexing='ij')
        df = pd.DataFrame({
            'lat': lats[anomalies_mask].flatten(),
            'lon': lons[anomalies_mask].flatten(),
            'anomaly_score': anomaly_scores[anomalies_mask].flatten()
        }).astype({
            'lat': 'float64',
            'lon': 'float64',
            'anomaly_score': 'float64'
        })

        return df

    def _visualize(self, anomalies: pd.DataFrame) -> Dict[str, str]:
        """
        Save anomalies to CSV.

        Args:
            anomalies: DataFrame of anomalies.

        Returns:
            Dictionary with artifact paths.
        """
        output_dir = Path(self.config["pipeline"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "anomalies.csv"
        anomalies.to_csv(csv_path, index=False)

        return {"anomalies_csv": str(csv_path)}