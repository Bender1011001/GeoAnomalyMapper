"""Parallel processing coordination using Dask for GeoAnomalyMapper.

This module provides the DaskCoordinator for distributed computing, supporting local,
distributed, and cloud backends. Handles task submission, result collection, fault
tolerance, and resource monitoring.
"""

from typing import Any, Callable, List, Optional, Dict, Union
from pathlib import Path
import time
import logging
from functools import wraps
from concurrent.futures import as_completed
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed
from dask import delayed
from .config import config_manager
from .exceptions import ParallelProcessingError
from .utils import ResourceUtils
from .progress import get_progress_reporter

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed tasks with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Task failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            raise ParallelProcessingError(
                f"Task failed after {max_retries} retries: {last_exception}",
                task_id=getattr(last_exception, 'task_id', None)
            ) from last_exception
        return wrapper
    return decorator


class DaskCoordinator:
    """Coordinator for Dask distributed computing in GAM.

    Manages Dask Client initialization, task submission, and result collection.
    Supports different backends and adaptive scaling.
    
    Attributes:
        client (Optional[Client]): Active Dask client.
        backend (str): Configured backend ('local', 'distributed', 'cloud').
        n_workers (int): Number of workers.
        memory_limit (Optional[str]): Memory limit per worker.
    """

    _instance = None

    def __new__(cls) -> 'DaskCoordinator':
        if cls._instance is None:
            cls._instance = super(DaskCoordinator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.client: Optional[Client] = None
            self.backend: str = 'local'
            self.n_workers: int = 4
            self.memory_limit: Optional[str] = None
            self.initialized = True
            self._init_from_config()

    def _init_from_config(self) -> None:
        """Initialize from configuration."""
        cfg = config_manager.current_config.parallel
        self.backend = cfg.backend
        self.n_workers = cfg.n_workers
        self.memory_limit = cfg.memory_limit

        try:
            self._initialize_client()
            logger.info(f"Dask client initialized: backend={self.backend}, workers={self.n_workers}")
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {e}")
            raise ParallelProcessingError(f"Dask initialization failed: {e}")

    def _initialize_client(self) -> None:
        """Initialize Dask client based on backend."""
        ResourceUtils.check_resources(required_mem_gb=1.0)  # Basic check

        if self.backend == 'local':
            cluster = LocalCluster(
                n_workers=self.n_workers,
                memory_limit=self.memory_limit,
                dashboard_address=None,  # Disable for now
            )
            self.client = Client(cluster)
        elif self.backend == 'distributed':
            # Assume config has address; fallback to local if not
            address = config_manager.get('parallel.address', None)
            if address:
                self.client = Client(address)
            else:
                logger.warning("No distributed address; falling back to local")
                self._initialize_client()  # Recurse to local
        elif self.backend == 'cloud':
            # Assume config has cloud params (e.g., AWS, GCP); placeholder for Client(config)
            # For production, integrate with dask-cloudprovider or similar
            address = config_manager.get('parallel.cloud_address', None)
            if address:
                self.client = Client(address)
            else:
                raise ParallelProcessingError("Cloud backend requires address in config")
        else:
            raise ParallelProcessingError(f"Unknown backend: {self.backend}")

    def adaptive_scale(self) -> None:
        """Adaptively scale workers based on resources."""
        if self.backend != 'local':
            logger.info("Adaptive scaling only supported for local backend")
            return

        suggested_workers = ResourceUtils.adaptive_scale_workers(self.n_workers)
        if suggested_workers != self.n_workers:
            logger.info(f"Scaling to {suggested_workers} workers")
            self.client.cluster.scale(suggested_workers)
            self.n_workers = suggested_workers

    def submit_task(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit a task to Dask, with optional retries.

        Args:
            func: Function to execute.
            *args, **kwargs: Arguments.

        Returns:
            Future or result (if serial fallback).

        Raises:
            ParallelProcessingError: If submission fails.
        """
        if not self.client:
            raise ParallelProcessingError("No Dask client available")

        try:
            # Wrap with retry if not already decorated
            wrapped_func = retry_on_failure()(func)
            future = self.client.submit(wrapped_func, *args, **kwargs)
            logger.debug(f"Submitted task: {func.__name__}, future={future}")
            return future
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise ParallelProcessingError(f"Failed to submit task {func.__name__}: {e}")

    def submit_delayed_tasks(
        self,
        tasks: List[Callable],
        *args_list: List[Any],
        progress_reporter: Optional[Any] = None,
    ) -> List[Any]:
        """Submit multiple delayed tasks.

        Args:
            tasks: List of delayed functions.
            args_list: List of args for each task.
            progress_reporter: Optional reporter for progress.

        Returns:
            List of futures.
        """
        futures = []
        for task, args in zip(tasks, args_list):
            future = self.submit_task(task, *args)
            futures.append(future)

        if progress_reporter:
            self._collect_with_progress(futures, progress_reporter)

        return futures

    def collect_results(
        self,
        futures: List[Any],
        raise_errors: bool = True,
        progress_reporter: Optional[Any] = None,
    ) -> List[Any]:
        """Collect results from futures, aggregating errors.

        Args:
            futures: List of Dask futures.
            raise_errors: Raise on first error or aggregate.
            progress_reporter: Optional for progress updates.

        Returns:
            List of results.

        Raises:
            ParallelProcessingError: If errors and raise_errors=True.
        """
        results = []
        errors = []

        if progress_reporter:
            total = len(futures)
            with progress_reporter.track_progress(total, desc="Collecting results") as bar:
                completed = 0
                for future in dask_as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        error_msg = f"Task failed: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        if raise_errors:
                            raise ParallelProcessingError(error_msg, task_id=str(future))
                    completed += 1
                    bar.update(1)
        else:
            for future in dask_as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    error_msg = f"Task failed: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    if raise_errors:
                        raise ParallelProcessingError(error_msg, task_id=str(future))

        if errors and raise_errors:
            raise ParallelProcessingError(f"Aggregated {len(errors)} task errors: {errors[:5]}")  # First 5

        logger.info(f"Collected {len(results)} results, {len(errors)} errors")
        return results + [None] * len(errors) if errors else results  # Pad with None for failed

    def monitor_resources(self, interval: float = 30.0) -> None:
        """Monitor and log resources periodically."""
        def monitor_loop():
            while self.client:
                resources = ResourceUtils.get_system_resources()
                logger.info(f"Resource monitor: CPU {resources.cpu_percent}%, Mem {resources.memory_percent}%")
                time.sleep(interval)

        # Start in thread; in production, use scheduler callbacks
        import threading
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def close(self) -> None:
        """Close the Dask client and cluster."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Dask client closed")

    def __del__(self):
        self.close()


# Global coordinator instance
dask_coordinator = DaskCoordinator()

# Fallback serial execution if no Dask
def serial_fallback(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Fallback to serial execution."""
    logger.warning("Using serial fallback (no Dask)")
    return func(*args, **kwargs)


def setup_dask_cluster(n_workers: int = 4) -> Client:
    """
    Setup and return a Dask client for the pipeline.

    Initializes the global DaskCoordinator if not already done, and returns the client.

    Args:
        n_workers: Number of workers for local cluster.

    Returns:
        Dask Client instance.
    """
    global dask_coordinator
    if dask_coordinator.client is None:
        # Force init with updated n_workers if needed
        dask_coordinator.n_workers = n_workers
        dask_coordinator._init_from_config()
    return dask_coordinator.client