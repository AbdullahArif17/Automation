"""Retry helper with exponential backoff.

Only retry transient errors. Never retry invalid requests endlessly.
"""
from __future__ import annotations

import time
from typing import Callable, Type, Tuple

from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0


class RetryError(Exception):
    pass


def retry(
    func: Callable,
    *args,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
):
    """Call `func` with exponential backoff on transient errors.

    Permanent (non-retryable) errors should not be in `retry_on`.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except retry_on as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (backoff ** (attempt - 1))
            logger.warning(
                f"attempt {attempt}/{max_attempts} failed, retrying in {delay:.1f}s",
                extra={"stage": "retry", "status": "retry", "error": str(exc)},
            )
            time.sleep(delay)
    raise RetryError(f"failed after {max_attempts} attempts: {last_exc}") from last_exc
