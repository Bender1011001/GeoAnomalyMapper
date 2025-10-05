"""
Exceptions for the GAM ingestion module.
"""

from ..core.exceptions import IngestionError


class CacheError(IngestionError):
    """
    Exception raised for cache-related errors.

    Parameters
    ----------
    key : str
        The cache key associated with the error.
    operation : str
        The operation that failed (e.g., 'save', 'load').
    message : str
        The error message.
    """

    def __init__(self, key: str, operation: str, message: str):
        self.key = key
        self.operation = operation
        super().__init__(f"Cache {operation} failed for key '{key}': {message}")