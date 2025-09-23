"""Parallel processing support for the GAM preprocessing module using Dask."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import dask.array as da
from dask import delayed
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

from gam.core.exceptions import PreprocessingError
from gam.ingestion.data_structures import RawData
from gam.preprocessing.base import Preprocessor
from gam.preprocessing.data_structures import ProcessedGrid
from obspy import Stream


logger = logging.getLogger(__name__)


class ChunkedRawData:
    """
    Wrapper for chunked RawData to support Dask arrays in values.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Original metadata.
    values : Any
        Chunked values (da.Array, xr.Dataset, or list for Stream).
    chunks : Optional[Dict[str, int]]
        Chunk sizes.
    """

    def __init__(self, metadata: Dict[str, Any], values: Any, chunks: Optional[Dict[str, int]] = None):
        self.metadata = metadata
        self.values = values
        self.chunks = chunks or {}

    def validate(self):
        """Basic validation."""
        if self.values is None:
            raise ValueError("Chunked values cannot be None")


class DaskPreprocessor(Preprocessor):
    """
    Wrapper for distributed processing using Dask arrays.

    Wraps any Preprocessor to enable chunking for memory efficiency and parallel
    execution. Supports local threads or distributed Client. Handles load balancing
    via Dask scheduler and fault tolerance with retries.

    Parameters
    ----------
    preprocessor : Preprocessor
        Wrapped preprocessor instance (e.g., GravityPreprocessor).
    chunk_sizes : Optional[Dict[str, int]], optional
        Chunk sizes for dimensions (e.g., {'lat': 100, 'lon': 100}; default: auto).
    n_workers : int, optional
        Number of workers (-1 for all cores; default: -1).
    retries : int, optional
        Retry failed tasks (default: 3).
    use_client : bool, optional
        Use distributed Client (default: True if n_workers > 0).

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid
        Process with parallel chunking.

    Notes
    -----
    - For large datasets, chunks data along spatial dims.
    - Seismic Stream: Splits traces and processes in parallel.
    - Falls back to serial if data small (<1e6 points) or non-chunkable.
    - Integrates with ProgressBar for monitoring.

    Examples
    --------
    >>> wrapped = GravityPreprocessor()
    >>> parallel_proc = DaskPreprocessor(wrapped, chunk_sizes={'lat': 50})
    >>> grid = parallel_proc.process(large_raw_data)
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        chunk_sizes: Optional[Dict[str, int]] = None,
        n_workers: int = -1,
        retries: int = 3,
        use_client: bool = True
    ):
        self.preprocessor = preprocessor
        self.chunk_sizes = chunk_sizes or {'lat': 100, 'lon': 100}
        self.n_workers = n_workers
        self.retries = retries
        self.use_client = use_client
        self.client = None
        if self.use_client and self.n_workers > 0:
            self.client = Client(n_workers=self.n_workers)

    def __del__(self):
        if self.client:
            self.client.close()

    def _chunk_data(self, data: RawData) -> ChunkedRawData:
        """Chunk RawData.values for parallel processing."""
        values = data.values
        metadata = dict(data.metadata)

        if isinstance(values, xr.Dataset):
            # Chunk xarray
            chunked_ds = values.chunk(self.chunk_sizes)
            return ChunkedRawData(metadata, chunked_ds, self.chunk_sizes)
        elif isinstance(values, np.ndarray):
            # Chunk ndarray to dask.array
            if values.size < 1e6:  # Threshold for small data
                return ChunkedRawData(metadata, values)
            shape = values.shape
            chunks = tuple(self.chunk_sizes.get(dim, min(100, s//10)) for dim, s in enumerate(shape))
            chunked = da.from_array(values, chunks=chunks)
            return ChunkedRawData(metadata, chunked, dict(zip(['dim'+str(i) for i in range(len(shape))], chunks)))
        elif isinstance(values, Stream):
            # Chunk traces
            if len(values) < 10:
                return ChunkedRawData(metadata, values)
            # Split into chunks of traces
            n_chunks = max(1, len(values) // 10)
            chunked_traces = [values[i::n_chunks] for i in range(n_chunks)]
            return ChunkedRawData(metadata, chunked_traces, {'traces': n_chunks})
        else:
            logger.warning(f"Cannot chunk {type(values)}; processing serially")
            return ChunkedRawData(metadata, values)

    def _parallel_process_chunk(self, chunk_data: Any, **kwargs) -> Any:
        """Process a single chunk."""
        # Reconstruct RawData for chunk
        if isinstance(chunk_data, ChunkedRawData):
            # For dask.array, compute chunk first
            if isinstance(chunk_data.values, da.Array):
                chunk_values = chunk_data.values.compute()
            else:
                chunk_values = chunk_data.values
            chunk_raw = RawData(chunk_data.metadata, chunk_values)
        else:
            chunk_raw = RawData(data.metadata, chunk_data)

        # Apply wrapped preprocessor
        try:
            result = self.preprocessor.process(chunk_raw, **kwargs)
            return result.ds if isinstance(result, ProcessedGrid) else result
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            raise

    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process data with parallel chunking.

        Parameters
        ----------
        data : RawData
            Input data.
        **kwargs : dict, optional
            Passed to wrapped preprocessor; add 'compute': bool (default: True).

        Returns
        -------
        ProcessedGrid
            Processed result (computed or lazy).

        Raises
        ------
        PreprocessingError
            If parallel execution fails.
        """
        data.validate()
        compute = kwargs.pop('compute', True)
        parallel = kwargs.get('parallel', True)

        if not parallel or data.values is None or (hasattr(data.values, 'size') and data.values.size < 1e6):
            logger.info("Data small or parallel disabled; processing serially")
            result = self.preprocessor.process(data, **kwargs)
            return result

        # Chunk data
        chunked_data = self._chunk_data(data)

        # Prepare delayed tasks
        if isinstance(chunked_data.values, (da.Array, xr.Dataset)):
            # For array/Dataset, use map_blocks if applicable, else delayed on chunks
            delayed_tasks = [delayed(self._parallel_process_chunk)(chunk, **kwargs) for chunk in chunked_data.values.chunks if hasattr(chunked_data.values, 'chunks')]
            if not delayed_tasks:
                delayed_tasks = [delayed(self._parallel_process_chunk)(chunked_data, **kwargs)]
        elif isinstance(chunked_data.values, list):  # Stream traces
            delayed_tasks = [delayed(self._parallel_process_chunk)(trace_chunk, **kwargs) for trace_chunk in chunked_data.values]
        else:
            delayed_tasks = [delayed(self._parallel_process_chunk)(chunked_data, **kwargs)]

        # Compute with retries
        try:
            with ProgressBar():
                results = da.compute(*delayed_tasks, retries=self.retries, scheduler='threads' if self.n_workers == -1 else self.client)[0] if len(delayed_tasks) == 1 else da.compute(*delayed_tasks, retries=self.retries, scheduler='threads' if self.n_workers == -1 else self.client)
            if isinstance(results, tuple):
                results = results[0]
        except Exception as e:
            raise PreprocessingError(f"Parallel processing failed: {e}")

        # Combine results (simplified: assume single or concat)
        if isinstance(results, xr.Dataset):
            final_ds = results
        elif isinstance(results, list):
            # For Stream, merge traces
            if all(isinstance(r, Stream) for r in results):
                final_stream = Stream(traces=sum((r.traces for r in results), []))
                final_ds = ProcessedGrid.from_stream(final_stream, data.metadata)  # Assume method
            else:
                # Concat datasets
                final_ds = xr.concat(results, dim='chunk').mean(dim='chunk')  # Simple merge
        else:
            final_ds = xr.Dataset({'data': xr.DataArray(results)})

        # Rechunk and finalize
        if compute:
            final_ds = final_ds.compute()
        grid = ProcessedGrid(final_ds)
        grid.add_metadata('parallel_processing', {'chunks': self.chunk_sizes, 'n_workers': self.n_workers})

        logger.info(f"Parallel preprocessing complete with {len(delayed_tasks)} chunks")
        return grid