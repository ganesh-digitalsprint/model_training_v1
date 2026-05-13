"""
retry_handler.py
Handles transient failures (OOM, spot preemptions, network blips).
"""
import time
import functools
import logging

logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3, delay: float = 5.0, backoff: float = 2.0,
               exceptions=(Exception,)):
    """Decorator: retry on specified exceptions with exponential backoff."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
                        raise
                    logger.warning(f"Attempt {attempt} failed ({e}). Retrying in {wait}s…")
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
