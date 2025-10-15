"""Robust Error Handling and Resilience Utilities for GeoAnomalyMapper.

This module provides:
- Error categorization (transient vs permanent)
- Retry decorators with exponential backoff + jitter
- DNS connectivity pre-checks
- Rate limit detection and throttling
- Token management with refresh
- File integrity validation
- Circuit breaker pattern for repeated failures

Usage:
    from utils.error_handling import RobustDownloader, retry_with_backoff, ensure_dns

    @retry_with_backoff(max_retries=3, backoff_factor=2)
    def download_with_retry(url):
        ...

    downloader = RobustDownloader(max_retries=5, base_delay=1.0)
    success = downloader.download(url, path)
"""

import functools
import logging
import random
import socket
import time
from typing import Any, Callable, Dict, Optional, Type, Union
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    HTTPError,
    ReadTimeout,
    RequestException,
    SSLError,
    Timeout,
)

logger = logging.getLogger(__name__)

# Error Categories
class RetryableError(Exception):
    """Raised for transient errors that should trigger retry."""
    pass

class PermanentError(Exception):
    """Raised for permanent errors that should not retry."""
    pass

class RateLimitError(RetryableError):
    """Specific to 429 rate limits."""
    pass

class AuthError(PermanentError):
    """Authentication failures."""
    pass

class IntegrityError(PermanentError):
    """Data integrity issues."""
    pass

# Circuit Breaker States
CLOSED = "CLOSED"  # Normal operation
OPEN = "OPEN"      # Fail fast after failures
HALF_OPEN = "HALF_OPEN"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = RequestException,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._state = CLOSED
        self._last_failure_time = None
        self._failure_count = 0
        self._last_state_change = time.time()
    
    @property
    def state(self) -> str:
        return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute func with circuit breaker logic."""
        if self._state == OPEN:
            if time.time() - self._last_state_change > self.recovery_timeout:
                self._state = HALF_OPEN
                logger.info("Circuit breaker: HALF_OPEN - testing recovery")
            else:
                raise PermanentError("Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            if self._state == OPEN:
                raise PermanentError(f"Circuit breaker tripped: {e}")
            raise
    
    def _on_success(self):
        self._failure_count = 0
        if self._state == HALF_OPEN:
            self._state = CLOSED
            self._last_state_change = time.time()
            logger.info("Circuit breaker: CLOSED - recovered")
    
    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = OPEN
            self._last_state_change = time.time()
            logger.warning(f"Circuit breaker: OPEN after {self._failure_count} failures")

def is_transient_error(e: Exception) -> bool:
    """Check if error is transient (retryable)."""
    return isinstance(e, (
        ConnectionError,
        ConnectTimeout,
        ReadTimeout,
        SSLError,
        Timeout,
        socket.gaierror,  # DNS resolution
    ))

def is_rate_limit_error(e: Union[Exception, requests.Response]) -> bool:
    """Check if error/response indicates rate limiting."""
    if isinstance(e, HTTPError):
        return e.response.status_code == 429
    if isinstance(e, requests.Response):
        return e.status_code == 429
    return False

def is_auth_error(e: Union[Exception, requests.Response]) -> bool:
    """Check if error/response indicates auth failure."""
    if isinstance(e, HTTPError):
        return e.response.status_code in (401, 403)
    if isinstance(e, requests.Response):
        return e.status_code in (401, 403)
    return False

def ensure_dns(hosts: list[str], timeout: int = 5, max_retries: int = 3) -> None:
    """Preflight DNS resolution check for hosts."""
    for host in hosts:
        for attempt in range(max_retries):
            try:
                socket.getaddrinfo(host, 443, timeout=timeout)
                logger.info(f"✓ DNS OK: {host}")
                break
            except socket.gaierror as e:
                if attempt == max_retries - 1:
                    raise PermanentError(f"DNS resolution failed for {host}: {e}")
                wait = (2 ** attempt) + random.uniform(0, 1)  # Backoff + jitter
                logger.warning(f"DNS retry {attempt+1} for {host} in {wait:.1f}s")
                time.sleep(wait)

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    raise_on_permanent: bool = True,
    expected_exceptions: tuple = (RequestException,),
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Callable:
    """Decorator for exponential backoff retry with jitter and error categorization."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    return func(*args, **kwargs)
                except tuple(expected_exceptions) as e:
                    last_exception = e
                    if attempt == max_retries:
                        if raise_on_permanent and isinstance(e, PermanentError):
                            raise
                        elif is_auth_error(e):
                            raise AuthError(f"Permanent auth error: {e}")
                        elif is_transient_error(e):
                            raise RetryableError(f"Max retries exhausted: {e}")
                        else:
                            raise PermanentError(f"Permanent error after retries: {e}")
                    
                    # Categorize and handle
                    if is_rate_limit_error(e):
                        delay = base_delay * (backoff_factor ** attempt) * 2  # Extra for rate limit
                        raise_type = RateLimitError
                    elif is_auth_error(e):
                        raise AuthError(f"Auth failed: {e}")
                    elif is_transient_error(e):
                        delay = base_delay * (backoff_factor ** attempt)
                        raise_type = RetryableError
                    else:
                        raise PermanentError(f"Non-retryable: {e}")
                    
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)  # 10% jitter
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed ({raise_type.__name__}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            
            raise RuntimeError("Should not reach here")
        return wrapper
    return decorator

class TokenManager:
    """Manages token refresh for services like Copernicus/NASA."""
    
    def __init__(self, auth_url: str, client_id: str = None, username: str = None, password: str = None):
        self.auth_url = auth_url
        self.client_id = client_id
        self.username = username
        self.password = password
        self.token = None
        self.expiry = 0
        self.session = requests.Session()
    
    @retry_with_backoff(max_retries=3, expected_exceptions=(RequestException, RetryableError))
    def refresh(self) -> str:
        """Refresh token with retry on transient errors."""
        if self.token and time.time() < self.expiry - 60:  # Refresh 1min early
            return self.token
        
        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }
        if self.client_id:
            payload["client_id"] = self.client_id
        
        try:
            resp = self.session.post(self.auth_url, data=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self.token = data["access_token"]
            self.expiry = time.time() + data.get("expires_in", 3600)
            logger.info("✓ Token refreshed successfully")
            return self.token
        except HTTPError as e:
            if is_auth_error(e):
                raise AuthError(f"Invalid credentials: {e}")
            raise
        except Exception as e:
            if is_transient_error(e):
                raise RetryableError(f"Token refresh transient: {e}")
            raise PermanentError(f"Token refresh failed: {e}")
    
    def get_header(self) -> Dict[str, str]:
        """Get Authorization header with fresh token."""
        token = self.refresh()
        return {"Authorization": f"Bearer {token}"}

class RobustDownloader:
    """Robust downloader with resilience features."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: tuple = (10, 30),  # connect, read
        max_connections: int = 10,
        pool_size: int = 5,
        circuit_breaker: bool = True,
        bandwidth_throttle: float = None,  # bytes/sec, None=unlimited
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.bandwidth_throttle = bandwidth_throttle
        self.circuit = CircuitBreaker() if circuit_breaker else None
        
        # Session with pooling and retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=base_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=max_connections,
            pool_maxsize=pool_size,
            max_retries=retry_strategy,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": "GeoAnomalyMapper/2.0"})
        
        # Token managers (service-specific)
        self.tokens: Dict[str, TokenManager] = {}
    
    def add_token_manager(
        self,
        service: str,
        auth_url: str,
        username: str,
        password: str = None,
        client_id: str = None,
    ):
        """Add token manager for a service."""
        self.tokens[service] = TokenManager(
            auth_url, client_id, username, password
        )
    
    def _get_headers(self, auth_service: Optional[str] = None) -> Dict[str, str]:
        """Get headers with auth if needed."""
        headers = {}
        if auth_service and auth_service in self.tokens:
            headers.update(self.tokens[auth_service].get_header())
        return headers
    
    @retry_with_backoff(
        max_retries=lambda self: self.max_retries,
        base_delay=lambda self: self.base_delay,
        circuit_breaker=lambda self: self.circuit,
    )
    def download_with_retry(
        self,
        url: str,
        output_path: Path,
        desc: str = None,
        auth_service: Optional[str] = None,
        chunk_size: int = 8192,
        expected_size: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> bool:
        """Download with full resilience features."""
        try:
            headers = self._get_headers(auth_service)
            resp = self.session.get(
                url,
                headers=headers,
                stream=True,
                timeout=self.timeout,
                allow_redirects=True,
            )
            resp.raise_for_status()
            
            total_size = int(resp.headers.get("content-length", 0))
            if expected_size and total_size != expected_size:
                logger.warning(f"Size mismatch: expected {expected_size}, got {total_size}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                logger.info(f"File exists, skipping: {output_path}")
                return self._validate_file(output_path, checksum)
            
            start_time = time.time()
            downloaded = 0
            last_throttle_time = start_time
            throttle_interval = chunk_size / self.bandwidth_throttle if self.bandwidth_throttle else 0
            
            with open(output_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=desc or urlparse(url).netloc,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        
                        # Bandwidth throttling
                        if self.bandwidth_throttle:
                            elapsed = time.time() - last_throttle_time
                            if elapsed < throttle_interval:
                                time.sleep(throttle_interval - elapsed)
                            last_throttle_time = time.time()
            
            logger.info(f"✓ Downloaded: {output_path} ({downloaded} bytes)")
            return self._validate_file(output_path, checksum, expected_size or total_size)
        
        except HTTPError as e:
            if is_rate_limit_error(e):
                delay = resp.headers.get("Retry-After", self.max_delay)
                logger.warning(f"Rate limited. Backing off {delay}s")
                time.sleep(float(delay))
                raise RateLimitError(f"Rate limited: {e}")
            elif is_auth_error(e):
                raise AuthError(f"Auth failed: {e}")
            raise RetryableError(f"HTTP error: {e}")
        except RequestException as e:
            if is_transient_error(e):
                raise RetryableError(f"Transient network: {e}")
            raise PermanentError(f"Permanent network: {e}")
        except Exception as e:
            raise PermanentError(f"Unexpected: {e}")
    
    def _validate_file(
        self, path: Path, checksum: Optional[str] = None, expected_size: Optional[int] = None
    ) -> bool:
        """Validate downloaded file integrity."""
        try:
            size = path.stat().st_size
            if expected_size and size != expected_size:
                raise IntegrityError(f"Size mismatch: {size} != {expected_size}")
            if size < 1024:  # Too small, likely error page
                raise IntegrityError(f"File too small: {size} bytes")
            
            if checksum:
                with open(path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != checksum:
                    raise IntegrityError(f"Checksum mismatch: {file_hash} != {checksum}")
            
            logger.debug(f"✓ Validated: {path}")
            return True
        except IntegrityError:
            path.unlink(missing_ok=True)
            raise
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            path.unlink(missing_ok=True)
            return False
    
    def close(self):
        """Close session."""
        self.session.close()

# Service-specific configurations (can be loaded from config.json)
DEFAULT_SERVICES = {
    "copernicus": {
        "auth_url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        "hosts": [
            "identity.dataspace.copernicus.eu",
            "catalogue.dataspace.copernicus.eu",
            "zipper.dataspace.copernicus.eu",
        ],
    },
    "nasa_earthdata": {
        "auth_url": "https://urs.earthdata.nasa.gov/oauth/token",  # Example, adjust
        "hosts": ["urs.earthdata.nasa.gov", "e4ftl01.cr.usgs.gov"],
    },
}

def get_downloader_for_service(
    service: str, username: str, password: Optional[str] = None, **kwargs
) -> RobustDownloader:
    """Factory for service-specific downloader."""
    config = DEFAULT_SERVICES.get(service, {})
    downloader = RobustDownloader(**kwargs)
    
    # DNS pre-check
    ensure_dns(config.get("hosts", []))
    
    # Token manager if auth needed
    if username:
        auth_url = config.get("auth_url")
        if auth_url:
            downloader.add_token_manager(
                service, auth_url, username, password
            )
    
    return downloader