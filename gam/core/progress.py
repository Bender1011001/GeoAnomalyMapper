"""Progress reporting utilities for GeoAnomalyMapper.

Provides progress bars and reporters for long-running tasks, using tqdm for terminal output.
Supports optional integration with Dask for distributed progress.
"""

from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not installed; progress bars disabled")


def get_progress_reporter(total: int = None, desc: str = "Processing") -> Optional[Any]:
    """
    Get a progress reporter instance (tqdm bar or None).

    Args:
        total: Total number of items (for bar length).
        desc: Description for the progress bar.

    Returns:
        tqdm instance or None if tqdm not available.
    """
    if not TQDM_AVAILABLE:
        logger.info(f"Progress reporter disabled: {desc}")
        return None

    try:
        bar = tqdm(total=total, desc=desc, unit="task")
        logger.debug(f"Progress bar created for '{desc}' with total {total}")
        return bar
    except Exception as e:
        logger.warning(f"Failed to create progress bar: {e}")
        return None


def update_progress(reporter: Optional[Any], increment: int = 1, desc: str = None) -> None:
    """
    Update the progress reporter.

    Args:
        reporter: Progress reporter from get_progress_reporter.
        increment: Number of steps to advance.
        desc: New description (optional).
    """
    if reporter is None:
        return

    try:
        if desc:
            reporter.set_description(desc)
        reporter.update(increment)
    except Exception as e:
        logger.debug(f"Progress update failed: {e}")


def close_progress(reporter: Optional[Any]) -> None:
    """
    Close the progress reporter.

    Args:
        reporter: Progress reporter to close.
    """
    if reporter is None:
        return

    try:
        reporter.close()
        logger.debug("Progress bar closed")
    except Exception as e:
        logger.debug(f"Progress close failed: {e}")