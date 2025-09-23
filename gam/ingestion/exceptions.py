"""Custom exceptions and retry decorators for the GAM data ingestion module."""

from typing import Any, Callable, Optional
import time
from functools import wraps
from threading import Lock
import logging

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_log
    from tenacity._utils import get_logger
except ImportError:
    raise ImportError("tenacity library is required for retry decorators. Install with: pip install tenacity")


logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """
    Base exception for all ingestion module errors.

    This serves as the parent class for all custom exceptions in the ingestion
    module, allowing for centralized error handling and logging.

    Parameters
    ----------
    message : str
        The error message.

    Attributes
    ----------
    message : str
        The error message.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DataFetchError(IngestionError):
    r"""
    Raised when data fetching from a source fails.

    This exception is thrown for network issues, API errors, invalid responses,
    or authentication failures during data retrieval.

    Parameters
    ----------
    source : str
        The data source that failed (e.g., 'USGS Gravity').
    message : str
        Detailed error message.

    Attributes
    ----------
    source : str
        The data source.
    message : str
        The error message.

    Notes
    -----
    This error triggers retry logic in fetchers. Log with source for traceability.

    Examples
    --------
    >>> raise DataFetchError("USGS Gravity", "Invalid API response")
    DataFetchError: Fetch failed for USGS Gravity: Invalid API response
    """

    def __init__(self, source: str, message: str):
        self.source = source
        super().__init__(f"Fetch failed for {source}: {message}")


class CacheError(IngestionError):
    r"""
    Raised when cache operations fail.

    This includes errors in saving, loading, or deleting cached data, such as
    file I/O issues, corruption, or key conflicts.

    Parameters
    ----------
    key : str
        The cache key involved.
    operation : str
        The operation that failed (e.g., 'save', 'load').
    message : str, optional
        Additional details.

    Attributes
    ----------
    key : str
        The cache key.
    operation : str
        The operation.
    message : str
        The error message.

    Examples
    --------
    >>> raise CacheError("gravity_123", "save", "Disk full")
    CacheError: Cache operation 'save' failed for key 'gravity_123': Disk full
    """

    def __init__(self, key: str, operation: str, message: Optional[str] = None):
        self.key = key
        self.operation = operation
        msg = f"Cache operation '{operation}' failed for key '{key}'"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class APITimeoutError(IngestionError):
    r"""
    Raised when an API request times out.

    This specific error indicates that the server did not respond within the
    allotted time, often due to high load or network latency.

    Parameters
    ----------
    source : str
        The data source (e.g., 'IRIS Seismic').
    timeout : float
        The timeout duration in seconds.
    message : str, optional
        Additional details.

    Attributes
    ----------
    source : str
        The data source.
    timeout : float
        The timeout value.
    message : str
        The error message.

    Notes
    -----
    This error is retryable with exponential backoff to handle transient issues.

    Examples
    --------
    >>> raise APITimeoutError("IRIS Seismic", 30.0, "Server overloaded")
    APITimeoutError: API timeout for IRIS Seismic (30.0s): Server overloaded
    """

    def __init__(self, source: str, timeout: float, message: Optional[str] = None):
        self.source = source
        self.timeout = timeout
        msg = f"API timeout for {source} ({timeout}s)"
        if message:
            msg += f": {message}"
        super().__init__(msg)


def retry_fetch(max_attempts: int = 3, min_wait: float = 4.0, max_wait: float = 10.0) -> Callable:
    r"""
    Decorator for retrying data fetch operations with exponential backoff.

    This decorator applies tenacity's retry logic to functions that may raise
    DataFetchError or APITimeoutError. It logs before each retry attempt.

    Parameters
    ----------
    max_attempts : int, default 3
        Maximum number of retry attempts.
    min_wait : float, default 4.0
        Minimum wait time between retries (seconds).
    max_wait : float, default 10.0
        Maximum wait time between retries (seconds).

    Returns
    -------
    Callable
        Wrapped function with retry logic.

    Notes
    -----
    Exponential backoff: wait = min_wait * (2 ** (attempt - 1)), capped at max_wait.
    Only retries on DataFetchError and APITimeoutError; other exceptions reraise immediately.
    Use on fetch_data methods in DataSource subclasses.

    Examples
    --------
    >>> @retry_fetch()
    ... def fetch_data(self, bbox, **kwargs):
    ...     # May raise DataFetchError
    ...     pass
    """

    tenacity_logger = get_logger(__name__)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((DataFetchError, APITimeoutError)),
            before=before_log(tenacity_logger, logging.INFO),
            reraise=True
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


def rate_limit(requests_per_minute: int = 60, window: int = 60) -> Callable:
    r"""
    Decorator for rate limiting API requests.

    This enforces a maximum number of requests within a time window using a
    simple token bucket-like mechanism with thread-safety. Sleeps if limit exceeded.

    Parameters
    ----------
    requests_per_minute : int, default 60
        Maximum requests allowed per minute.
    window : int, default 60
        Time window in seconds (typically 60 for per-minute).

    Returns
    -------
    Callable
        Wrapped function with rate limiting.

    Notes
    -----
    Uses a shared Lock for thread-safety. Tracks request timestamps in a list,
    removing old ones. Sleeps the remainder of the window if over limit.
    Suitable for respecting API quotas; adjust per source if needed.

    Examples
    --------
    >>> @rate_limit(30)  # 30 requests per minute
    ... def api_call(self, url):
    ...     # Make request
    ...     pass
    """

    lock = Lock()
    request_times: list[float] = []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal request_times
            with lock:
                now = time.time()
                # Remove old requests outside window
                request_times = [t for t in request_times if now - t < window]
                # Check if over limit
                if len(request_times) >= requests_per_minute:
                    # Calculate sleep time until next window slot
                    sleep_time = window - (now - request_times[0])
                    if sleep_time > 0:
                        logger.info(f"Rate limit hit; sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    # Remove the oldest
                    request_times.pop(0)
                request_times.append(now)
            return func(*args, **kwargs)
        return wrapper

    return decorator