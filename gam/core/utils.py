"""Core utility functions and classes for GeoAnomalyMapper.

This module provides reusable utilities for bounding box manipulation, file handling,
input validation, and system resource monitoring. Designed for integration with other
core components.
"""

from typing import Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class BoundingBoxUtils:
    """Utilities for geographic bounding box validation and manipulation.

    Works with tuples (lat_min, lat_max, lon_min, lon_max) or BoundingBox objects.
    """

    @staticmethod
    def validate_bbox(bbox: Union[Tuple[float, float, float, float], Any]) -> Tuple[float, float, float, float]:
        """Validate and normalize a bounding box.

        Ensures lat_min < lat_max, lon_min < lon_max, and bounds within [-90,90] lat, [-180,180] lon.
        Normalizes if necessary (e.g., crosses antimeridian).

        Args:
            bbox: Bounding box as tuple or BoundingBox object.

        Returns:
            Normalized tuple (lat_min, lat_max, lon_min, lon_max).

        Raises:
            ValueError: If bbox is invalid.
        """
        # Handle BoundingBox objects that have a .tuple attribute
        if hasattr(bbox, 'tuple'):
            bbox = bbox.tuple

        lat_min, lat_max, lon_min, lon_max = bbox

        # Validate ranges
        if not (-90 <= lat_min <= lat_max <= 90):
            raise ValueError(f"Invalid latitude bounds: {lat_min}, {lat_max}")
        if not (-180 <= lon_min <= lon_max <= 180):
            raise ValueError(f"Invalid longitude bounds: {lon_min}, {lon_max}")

        # Normalize antimeridian crossing
        if lon_max - lon_min > 180:
            # Split into two bboxes or adjust; for simplicity, warn and take smaller arc
            logger.warning("Bounding box crosses antimeridian; using primary arc")
            if lon_min < 0:
                lon_max = 180
            else:
                lon_min = -180

        return (lat_min, lat_max, lon_min, lon_max)


# Standalone validate_bbox function for direct import
def validate_bbox(bbox: Union[Tuple[float, float, float, float], Any]) -> Tuple[float, float, float, float]:
    """Standalone wrapper for BoundingBoxUtils.validate_bbox.
    
    This function provides direct access to bbox validation for imports like:
    from .utils import validate_bbox
    
    Args:
        bbox: Bounding box as tuple or BoundingBox object.
    
    Returns:
        Normalized tuple (lat_min, lat_max, lon_min, lon_max).
    
    Raises:
        ValueError: If bbox is invalid.
    """
    return BoundingBoxUtils.validate_bbox(bbox)


import psutil
from dataclasses import dataclass


@dataclass
class ResourceInfo:
    """Dataclass for system resource usage."""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float


class ResourceUtils:
    """Utilities for system resource monitoring and validation.

    Uses psutil for CPU and memory metrics. Supports adaptive worker scaling for parallel processing.
    """
    
    @staticmethod
    def check_resources(required_mem_gb: float = 1.0) -> bool:
        """Check if system has sufficient resources (memory).

        Args:
            required_mem_gb: Minimum available memory in GB.

        Returns:
            True if sufficient, else False.

        Raises:
            ImportError: If psutil not available.
        """
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            if available_gb < required_mem_gb:
                logger.warning(f"Low memory: {available_gb:.1f}GB available, {required_mem_gb}GB required")
                return False
            return True
        except ImportError:
            logger.warning("psutil not installed; skipping resource check")
            return True  # Assume OK if not checkable

    @staticmethod
    def adaptive_scale_workers(current_workers: int) -> int:
        """Suggest optimal number of workers based on CPU cores and memory.

        Limits to 80% of logical cores, adjusts down if memory low.

        Args:
            current_workers: Current number of workers.

        Returns:
            Suggested number of workers.
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024 ** 3)
            
            max_workers = int(cpu_count * 0.8)
            if memory_gb < 8:
                max_workers = min(max_workers, 2)
            elif memory_gb < 16:
                max_workers = min(max_workers, 4)
            
            suggested = min(current_workers, max_workers)
            logger.debug(f"Adaptive scaling: {current_workers} -> {suggested} (CPU: {cpu_count}, Mem: {memory_gb:.1f}GB)")
            return suggested
        except ImportError:
            logger.warning("psutil not installed; no scaling adjustment")
            return current_workers

    @staticmethod
    def get_system_resources() -> ResourceInfo:
        """Get current system resource usage.

        Returns:
            ResourceInfo with CPU and memory percentages, available memory in GB.
        """
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            mem_percent = memory.percent
            available_gb = memory.available / (1024 ** 3)
            return ResourceInfo(cpu_percent=cpu, memory_percent=mem_percent, available_memory_gb=available_gb)
        except ImportError:
            logger.warning("psutil not installed; returning dummy resources")
            return ResourceInfo(cpu_percent=0.0, memory_percent=0.0, available_memory_gb=0.0)