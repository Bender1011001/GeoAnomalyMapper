"""
Core Exceptions Module.

Defines custom exceptions for GAM pipeline errors, ingestion failures, modeling issues, etc.
All exceptions inherit from GAMError for centralized handling and logging.
"""

import logging
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

class GAMError(Exception):
    """
    Base exception for all GeoAnomalyMapper errors.

    Provides standardized error formatting with context and logging.
    """
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize GAMError.

        Args:
            message: Error message.
            context: Additional context (e.g., {'bbox': bbox, 'modality': 'gravity'}).
            cause: Original exception if chained.
        """
        full_message = f"GAM Error: {message}"
        if context:
            full_message += f" | Context: {context}"
        super().__init__(full_message)
        
        self.context = context or {}
        self.cause = cause
        log.error(full_message, exc_info=cause is not None)

    def __str__(self):
        return self.args[0]

class PipelineError(GAMError):
    """
    Exception raised during pipeline execution (e.g., stage failures, configuration issues).
    """
    def __init__(self, message: str, stage: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize PipelineError.

        Args:
            message: Error message.
            stage: Pipeline stage (e.g., 'ingestion', 'modeling').
            context: Additional context.
            cause: Original exception.
        """
        ctx = {"stage": stage}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)

class IngestionError(GAMError):
    """Raised for data fetching or parsing errors."""
    def __init__(self, message: str, source: str = "unknown", modality: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"source": source, "modality": modality}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)

class PreprocessingError(GAMError):
    """Raised for data cleaning, gridding, or filtering errors."""
    def __init__(self, message: str, step: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"step": step}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)

class ModelingError(GAMError):
    """Raised for inversion, fusion, or anomaly detection errors."""
    def __init__(self, message: str, model_type: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"model_type": model_type}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)

class VisualizationError(GAMError):
    """Raised for plotting or export errors."""
    def __init__(self, message: str, plot_type: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"plot_type": plot_type}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)

class InversionConvergenceError(ModelingError):
    """
    Raised when inversion does not converge within tolerance or iterations.
    """
    def __init__(self, message: str, model_type: str = "inversion", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message, model_type, context, cause)


class ParallelProcessingError(GAMError):
    """Raised for parallel processing errors (e.g., Dask task failures, cluster issues)."""
    def __init__(self, message: str, task_id: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"task_id": task_id}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)


# Re-export for convenience
__all__ = [
    'GAMError', 'PipelineError', 'IngestionError',
    'PreprocessingError', 'ModelingError', 'VisualizationError',
    'ParallelProcessingError'
]
class ConfigurationError(GAMError):
    """Raised for configuration loading or validation errors."""
    def __init__(self, message: str, key: str = "unknown", context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        ctx = {"key": key}
        if context:
            ctx.update(context)
        super().__init__(message, ctx, cause)