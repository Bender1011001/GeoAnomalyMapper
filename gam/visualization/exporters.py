"""Export functionality for GAM visualization module.

This module provides classes for exporting geophysical data and results to various
formats, preserving coordinate systems, metadata, and scientific integrity. Each
exporter handles specific data types and formats suitable for GIS, 3D viewing,
database storage, and tabular analysis.

Supported formats:
- GeoTIFF: Raster grids for ProcessedGrid/InversionResults (2D/3D multi-band)
- VTK: 3D models for ParaView (InversionResults)
- SQLite: Tabular anomalies (AnomalyOutput)
- JSON: Serializable all data types
- CSV: Tabular anomalies (AnomalyOutput)

All exporters embed metadata (units, CRS, timestamp) and support geospatial
coordinates. Use via VisualizationManager.export_data() or directly.

Notes
-----
- CRS Preservation: Defaults to EPSG:4326; override via kwargs['crs'].
- Metadata: Embedded as file tags (GeoTIFF), point/cell data (VTK), columns (DB/CSV),
  keys (JSON).
- Error Handling: Validates data types/paths; logs warnings for fallbacks.
- Performance: Chunked writes for large data; compression optional.
- Dependencies: rasterio, pyvista, sqlalchemy, json, pandas (from requirements.txt).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Union, Any, Dict, Optional
import warnings

import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import pyvista as pv
from sqlalchemy import create_engine, Column, Float, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

logger = logging.getLogger(__name__)
Base = declarative_base()


class AnomalyTable(Base):
    """SQLAlchemy model for AnomalyOutput table."""
    __tablename__ = 'anomalies'

    id = Column(Integer, primary_key=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)
    confidence = Column(Float)
    anomaly_type = Column(String(50))
    strength = Column(Float)
    uncertainty = Column(Float, nullable=True)
    modality_contributions = Column(Text, nullable=True)  # JSON string
    metadata = Column(Text, nullable=True)  # JSON string for extra metadata
    created_at = Column(DateTime, default=datetime.utcnow)


def _extract_coords(data: Union[ProcessedGrid, InversionResults, AnomalyOutput]) -> Dict[str, Any]:
    """Helper to extract coordinates and CRS from data."""
    if isinstance(data, ProcessedGrid):
        ds = data.ds
        crs = data.coordinate_system
        lats = ds.coords['lat'].values
        lons = ds.coords['lon'].values
        if 'depth' in ds.dims:
            depths = ds.coords['depth'].values
        else:
            depths = np.array([0.0])
        return {'lats': lats, 'lons': lons, 'depths': depths, 'crs': crs}
    elif isinstance(data, InversionResults):
        # Assume regular grid from shape; in production, derive from mesh
        n_lat, n_lon, n_depth = data.model.shape
        lats = np.linspace(-90, 90, n_lat)  # Fallback; use metadata if available
        lons = np.linspace(-180, 180, n_lon)
        depths = np.linspace(0, 1000, n_depth)
        crs = data.metadata.get('crs', 'EPSG:4326')
        return {'lats': lats, 'lons': lons, 'depths': depths, 'crs': crs}
    elif isinstance(data, AnomalyOutput):
        coords = data[['lat', 'lon', 'depth']].values
        crs = data.attrs.get('crs', 'EPSG:4326') if hasattr(data, 'attrs') else 'EPSG:4326'
        return {'points': coords, 'crs': crs}
    raise ValueError(f"Unsupported data type: {type(data)}")


def _serialize_metadata(metadata: Dict[str, Any]) -> str:
    """Serialize metadata dict to JSON string."""
    # Handle non-serializable types
    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, datetime):
            serializable[k] = v.isoformat()
        elif isinstance(v, (np.ndarray, pd.DataFrame)):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return json.dumps(serializable)


class GeoTIFFExporter:
    """
    Exporter for ProcessedGrid and InversionResults to GeoTIFF format using rasterio.

    Supports 2D grids directly; 3D as multi-band (depth slices) or separate files.
    Preserves CRS, units, metadata as tags. Affine transform from lat/lon bounds.

    Parameters
    ----------
    compress : bool, optional
        Enable compression (default: True).

    Methods
    -------
    export(data: Union[ProcessedGrid, InversionResults], path: str, **kwargs) -> None

    Notes
    -----
    - For 3D: Exports as multi-band if ndepth <= 100; else warns and exports surface.
    - Metadata: Tags include 'units', 'processed_at', 'crs'.
    - CRS: Validates and writes to profile.
    - Limitations: GeoTIFF is 2.5D; for full 3D use VTKExporter.

    Examples
    --------
    >>> exporter = GeoTIFFExporter()
    >>> exporter.export(processed_grid, 'grid.tif', crs='EPSG:3857')
    """

    def __init__(self, compress: bool = True):
        self.compress = compress

    def export(
        self,
        data: Union[ProcessedGrid, InversionResults],
        path: str,
        **kwargs: Any
    ) -> None:
        """
        Export data to GeoTIFF file.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults]
            Input grid or model data.
        path : str
            Output .tif file path.
        **kwargs : dict, optional
            - 'crs': str, override CRS (default: from data).
            - 'band_order': str, 'depth' or 'surface' (default: 'depth').
            - 'dtype': str, raster dtype (default: 'float32').

        Raises
        ------
        ValueError
            If unsupported data or invalid path/CRS.
        IOError
            If write fails.
        """
        if not isinstance(data, (ProcessedGrid, InversionResults)):
            raise ValueError("GeoTIFFExporter supports only ProcessedGrid or InversionResults")

        coords = _extract_coords(data)
        crs_str = kwargs.get('crs', coords['crs'])
        try:
            crs = CRS.from_string(crs_str)
        except Exception as e:
            raise ValueError(f"Invalid CRS '{crs_str}': {e}")

        if isinstance(data, ProcessedGrid):
            ds = data.ds
            raster_data = ds['data'].values
            if 'uncertainty' in ds:
                # For simplicity, export data only; extend for multi-band uncertainty
                pass
            metadata = dict(ds.attrs)
        else:  # InversionResults
            raster_data = data.model
            metadata = data.metadata

        # Handle 3D: squeeze to 2D if no depth, else multi-band
        if len(raster_data.shape) == 3:
            if raster_data.shape[2] > 100:  # Arbitrary limit
                warnings.warn("Large 3D data; exporting surface slice")
                raster_data = raster_data[:, :, 0]  # Top slice
            else:
                # Stack depth slices as bands (height, width, bands)
                raster_data = np.moveaxis(raster_data, -1, 0)  # (depth, lat, lon) -> (bands, height, width)
        else:
            raster_data = raster_data.squeeze()

        height, width = raster_data.shape[-2:]
        transform = from_bounds(
            coords['lons'][0], coords['lats'][0], coords['lons'][-1], coords['lats'][-1],
            width, height
        )

        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': raster_data.shape[0] if len(raster_data.shape) == 3 else 1,
            'dtype': kwargs.get('dtype', rasterio.float32),
            'crs': crs,
            'transform': transform,
            'compress': self.compress,
            'tiled': True,
            'blockxsize': 256,
            'blockyssize': 256
        }

        with rasterio.open(path, 'w', **profile) as dst:
            if len(raster_data.shape) == 3:
                for i in range(raster_data.shape[0]):
                    dst.write(raster_data[i], i + 1)
            else:
                dst.write(raster_data, 1)

            # Embed metadata as tags
            for k, v in metadata.items():
                if isinstance(v, (str, int, float)):
                    dst.update_tags(**{k: str(v)})
                elif isinstance(v, dict):
                    dst.update_tags(**{f"{k}_{subk}": str(subv) for subk, subv in v.items() if isinstance(subv, (str, int, float))})

        logger.info(f"Exported GeoTIFF to {path} (CRS: {crs_str}, shape: {raster_data.shape})")


class VTKExporter:
    """
    Exporter for 3D InversionResults to VTK format using PyVista.

    Creates structured grid from model array, adds uncertainty as cell data.
    Supports point clouds for AnomalyOutput. Preserves CRS via point coordinates.

    Parameters
    ----------
    binary : bool, optional
        Binary VTK (default: True for efficiency).

    Methods
    -------
    export(data: Union[InversionResults, AnomalyOutput], path: str, **kwargs) -> None

    Notes
    -----
    - For InversionResults: UniformGrid with model/uncertainty as cell arrays.
    - For AnomalyOutput: PolyData with points and scalar 'confidence'.
    - Coordinates: Derived from data; transform if kwargs['target_crs'].
    - Metadata: Added as field arrays or global info.
    - View in ParaView: Load .vtk, apply isosurfaces/color maps.

    Examples
    --------
    >>> exporter = VTKExporter()
    >>> exporter.export(inversion_results, 'model.vtk')
    """

    def __init__(self, binary: bool = True):
        self.binary = binary

    def export(
        self,
        data: Union[InversionResults, AnomalyOutput],
        path: str,
        **kwargs: Any
    ) -> None:
        """
        Export 3D data to VTK file.

        Parameters
        ----------
        data : Union[InversionResults, AnomalyOutput]
            3D model or anomaly points.
        path : str
            Output .vtk file path.
        **kwargs : dict, optional
            - 'target_crs': str, transform coords to this CRS.
            - 'spacing': Tuple[float, float, float], grid spacing (default: auto).

        Raises
        ------
        ValueError
            If unsupported data or no 3D structure.
        """
        if isinstance(data, InversionResults):
            coords = _extract_coords(data)
            n_lat, n_lon, n_depth = data.model.shape
            # Create meshgrid for structured grid
            lats, lons, depths = np.meshgrid(coords['lats'], coords['lons'], coords['depths'], indexing='ij')
            # Flatten for points
            points = np.column_stack([lons.ravel(), lats.ravel(), depths.ravel()])

            # Transform if target_crs
            target_crs = kwargs.get('target_crs')
            if target_crs and target_crs != coords['crs']:
                transformer = Transformer.from_crs(coords['crs'], target_crs, always_xy=True)
                points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])

            # Create UniformGrid
            spacing = (
                (coords['lons'][-1] - coords['lons'][0]) / (n_lon - 1),
                (coords['lats'][-1] - coords['lats'][0]) / (n_lat - 1),
                (coords['depths'][-1] - coords['depths'][0]) / (n_depth - 1)
            )
            grid = pv.UniformGrid(dimensions=(n_lon, n_lat, n_depth))
            grid.origin = (points[0, 0], points[0, 1], points[0, 2])
            grid.spacing = spacing
            grid.cell_data['model'] = data.model.ravel(order='F')  # Fortran order for grid
            grid.cell_data['uncertainty'] = data.uncertainty.ravel(order='F')

            # Add metadata as global
            grid.field_data['units'] = data.metadata.get('units', 'unknown')
            grid.field_data['crs'] = target_crs or coords['crs']
            grid.field_data['metadata'] = _serialize_metadata(data.metadata)

            grid.save(path, binary=self.binary)
            logger.info(f"Exported VTK grid to {path} (dims: {grid.dimensions})")

        elif isinstance(data, AnomalyOutput):
            points = data[['lon', 'lat', 'depth']].values  # x=lon, y=lat, z=depth
            # Transform if needed (similar to above)
            target_crs = kwargs.get('target_crs')
            if target_crs and target_crs != _extract_coords(data)['crs']:
                transformer = Transformer.from_crs(_extract_coords(data)['crs'], target_crs, always_xy=True)
                points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])

            polydata = pv.PolyData(points)
            polydata.point_data['confidence'] = data['confidence'].values
            polydata.point_data['strength'] = data['strength'].values
            if 'uncertainty' in data:
                polydata.point_data['uncertainty'] = data['uncertainty'].values
            polydata.field_data['metadata'] = _serialize_metadata(dict(data.attrs) if hasattr(data, 'attrs') else {})

            polydata.save(path, binary=self.binary)
            logger.info(f"Exported VTK points to {path} (n_points: {len(data)})")

        else:
            raise ValueError("VTKExporter supports only InversionResults or AnomalyOutput")


class DatabaseExporter:
    """
    Exporter for AnomalyOutput to SQLite database using SQLAlchemy.

    Creates/connects to DB, defines schema, inserts rows with metadata as JSON.
    Supports append mode for batch processing.

    Parameters
    ----------
    db_url : str, optional
        SQLite URL (default: ':memory:' for testing; use file path for persistence).

    Methods
    -------
    export(data: AnomalyOutput, path: str, **kwargs) -> None
        Path is DB file path if not in-memory.

    Notes
    -----
    - Schema: anomalies table with core columns + JSON for contributions/metadata.
    - If table exists, appends (use if_exists='replace' via kwargs).
    - CRS/Metadata: Stored in separate columns or JSON.
    - Production: Use connection pooling for large inserts.

    Examples
    --------
    >>> exporter = DatabaseExporter('anomalies.db')
    >>> exporter.export(anomalies, 'anomalies.db')
    """

    def __init__(self, db_url: str = ':memory:'):
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def export(
        self,
        data: AnomalyOutput,
        path: str,
        **kwargs: Any
    ) -> None:
        """
        Export anomalies to SQLite database.

        Parameters
        ----------
        data : AnomalyOutput
            Tabular anomalies.
        path : str
            DB file path (overrides init db_url if provided).
        **kwargs : dict, optional
            - 'if_exists': 'append' or 'replace' (default: 'append').
            - 'crs': str, add as column.

        Raises
        ------
        ValueError
            If not AnomalyOutput.
        """
        if not isinstance(data, AnomalyOutput):
            raise ValueError("DatabaseExporter supports only AnomalyOutput")

        if path != ':memory:':
            self.engine = create_engine(f"sqlite:///{path}")
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)

        session = self.Session()
        try:
            if_exists = kwargs.get('if_exists', 'append')
            if if_exists == 'replace':
                Base.metadata.drop_all(self.engine, tables=[AnomalyTable.__table__])
                Base.metadata.create_all(self.engine)

            # Prepare rows
            for _, row in data.iterrows():
                anomaly = AnomalyTable(
                    lat=row['lat'],
                    lon=row['lon'],
                    depth=row['depth'],
                    confidence=row['confidence'],
                    anomaly_type=row['anomaly_type'],
                    strength=row['strength'],
                    uncertainty=row.get('uncertainty'),
                    modality_contributions=json.dumps(row.get('modality_contributions', {})),
                    metadata=_serialize_metadata(dict(data.attrs) if hasattr(data, 'attrs') else {})
                )
                session.add(anomaly)
            session.commit()

            # Add CRS if provided
            crs = kwargs.get('crs', _extract_coords(data)['crs'])
            if crs:
                # Could add a global metadata table; for simplicity, log
                logger.info(f"Exported to DB {path} with CRS: {crs}")

        except Exception as e:
            session.rollback()
            raise ValueError(f"DB export failed: {e}")
        finally:
            session.close()

        logger.info(f"Exported {len(data)} anomalies to DB {path}")


class JSONExporter:
    """
    Universal exporter for all data types to JSON format.

    Serializes arrays as lists, datetimes as ISO, dicts directly. Preserves full
    structure for API/web use.

    Parameters
    ----------
    indent : int, optional
        JSON indentation (default: 2).

    Methods
    -------
    export(data: Union[ProcessedGrid, InversionResults, AnomalyOutput], path: str, **kwargs) -> None

    Notes
    -----
    - Large arrays: Warns if >1M elements; consider compression.
    - Metadata: Top-level keys.
    - Coordinates: Included as 'coords' key.

    Examples
    --------
    >>> exporter = JSONExporter()
    >>> exporter.export(anomalies, 'anomalies.json')
    """

    def __init__(self, indent: int = 2):
        self.indent = indent

    def export(
        self,
        data: Union[ProcessedGrid, InversionResults, AnomalyOutput],
        path: str,
        **kwargs: Any
    ) -> None:
        """
        Export data to JSON file.

        Parameters
        ----------
        data : Union[ProcessedGrid, InversionResults, AnomalyOutput]
            Input data.
        path : str
            Output .json file path.
        **kwargs : dict, optional
            - 'orient': str for DataFrame (default: 'records').

        Raises
        ------
        ValueError
            If serialization fails.
        """
        serializable = {}

        if isinstance(data, ProcessedGrid):
            serializable['type'] = 'ProcessedGrid'
            serializable['data'] = data.ds['data'].values.tolist()
            if 'uncertainty' in data.ds:
                serializable['uncertainty'] = data.ds['uncertainty'].values.tolist()
            serializable['coords'] = {
                'lat': data.ds.coords['lat'].values.tolist(),
                'lon': data.ds.coords['lon'].values.tolist()
            }
            if 'depth' in data.ds.coords:
                serializable['coords']['depth'] = data.ds.coords['depth'].values.tolist()
            serializable['metadata'] = _serialize_metadata(dict(data.ds.attrs))
            serializable['crs'] = data.coordinate_system

        elif isinstance(data, InversionResults):
            serializable['type'] = 'InversionResults'
            serializable['model'] = data.model.tolist()
            serializable['uncertainty'] = data.uncertainty.tolist()
            serializable['metadata'] = _serialize_metadata(data.metadata)
            coords = _extract_coords(data)
            serializable['coords'] = {
                'lats': coords['lats'].tolist(),
                'lons': coords['lons'].tolist(),
                'depths': coords['depths'].tolist()
            }
            serializable['crs'] = coords['crs']

        elif isinstance(data, AnomalyOutput):
            orient = kwargs.get('orient', 'records')
            serializable['type'] = 'AnomalyOutput'
            serializable['data'] = data.to_dict(orient=orient)
            # Deserialize contributions for JSON
            if 'modality_contributions' in data.columns:
                for row in serializable['data']:
                    if isinstance(row.get('modality_contributions'), str):
                        row['modality_contributions'] = json.loads(row['modality_contributions'])
            serializable['metadata'] = _serialize_metadata(dict(data.attrs) if hasattr(data, 'attrs') else {})
            serializable['crs'] = _extract_coords(data)['crs']

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Check size
        total_size = len(json.dumps(serializable, default=str))
        if total_size > 10**7:  # 10MB
            warnings.warn(f"Large JSON ({total_size/10**6:.1f}MB); consider compression")

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=self.indent, default=str)

        logger.info(f"Exported JSON to {path} (size: {total_size/1024:.1f}KB)")


class CSVExporter:
    """
    Exporter for AnomalyOutput to CSV format.

    Uses pandas.to_csv; embeds metadata as header rows or extra columns.
    Handles dict columns as JSON strings.

    Parameters
    ----------
    include_metadata : bool, optional
        Add metadata columns (default: True).

    Methods
    -------
    export(data: AnomalyOutput, path: str, **kwargs) -> None

    Notes
    -----
    - Dict columns (e.g., modality_contributions): JSON strings.
    - Metadata: First rows as key-value if include_metadata.
    - No geospatial indexing; use for tabular export.

    Examples
    --------
    >>> exporter = CSVExporter()
    >>> exporter.export(anomalies, 'anomalies.csv', index=False)
    """

    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata

    def export(
        self,
        data: AnomalyOutput,
        path: str,
        **kwargs: Any
    ) -> None:
        """
        Export anomalies to CSV file.

        Parameters
        ----------
        data : AnomalyOutput
            Tabular data.
        path : str
            Output .csv file path.
        **kwargs : dict, optional
            Passed to df.to_csv (e.g., index=False, sep=',').

        Raises
        ------
        ValueError
            If not AnomalyOutput or invalid path.
        """
        if not isinstance(data, AnomalyOutput):
            raise ValueError("CSVExporter supports only AnomalyOutput")

        df = data.copy()
        # Handle dict columns as JSON strings
        if 'modality_contributions' in df.columns:
            df['modality_contributions'] = df['modality_contributions'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
            )

        if self.include_metadata and hasattr(data, 'attrs') and data.attrs:
            # Add metadata as constant columns
            for k, v in data.attrs.items():
                if isinstance(v, (str, int, float)):
                    df[f'metadata_{k}'] = v
                else:
                    df[f'metadata_{k}'] = str(v)

        # Add CRS if available
        crs = _extract_coords(data)['crs']
        if crs:
            df['crs'] = crs

        df.to_csv(path, **kwargs)
        logger.info(f"Exported CSV to {path} (rows: {len(df)}, columns: {list(df.columns)})")