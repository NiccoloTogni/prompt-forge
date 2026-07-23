"""Internal retry helper for LLM calls."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def call_with_retry(fn, max_retries: int, delay: float, backoff: float = 2.0):
    """
    Call fn(), retrying up to max_retries times with exponential backoff on failure.

    Args:
        fn: Zero-argument callable to invoke.
        max_retries: Maximum number of retries (0 = no retries, try once).
        delay: Initial wait in seconds before the first retry.
        backoff: Multiplier applied to delay after each retry (default 2.0).

    Raises:
        The last exception raised by fn if all attempts fail.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = delay * (backoff ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1, max_retries + 1, exc, wait,
            )
            time.sleep(wait)
