"""Bounded-retry wrapper for outbound HTTP fetches.

Replaces the three near-identical ``fetch_with_retry`` retry loops that lived
in the logger / backfill / archive-ingest scripts. This module owns only the
retry / backoff / logging skeleton; the caller supplies a zero-argument
callable and keeps its own result conventions.
"""

from __future__ import annotations

import logging
import time
import urllib.error
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Transient network errors worth retrying. ``HTTPError`` is a subclass of
# ``URLError`` so it is retried too; a caller that needs to treat a specific
# status (e.g. 404) as a definite answer should catch it inside its own
# callable and return a sentinel rather than raising.
RETRYABLE: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
)


def fetch_with_retry(
    call: Callable[[], T],
    *,
    retries: int,
    backoff_sec: int,
    retryable: tuple[type[BaseException], ...] = RETRYABLE,
    label: str = "fetch",
    logger: Optional[logging.Logger] = None,
) -> T:
    """Call ``call()`` with bounded linear backoff on transient errors.

    Retries up to ``retries`` times, sleeping ``backoff_sec * attempt`` seconds
    between attempts. Re-raises the last transient error once attempts are
    exhausted; any exception not in ``retryable`` propagates immediately.
    """
    log = logger or logging.getLogger(__name__)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            return call()
        except retryable as exc:
            last_exc = exc
            wait = backoff_sec * attempt
            log.warning(
                "%s attempt %d/%d failed: %s; sleeping %ds",
                label, attempt, retries, exc, wait,
            )
            time.sleep(wait)
    raise last_exc if last_exc is not None else RuntimeError(
        f"{label}: retries must be >= 1"
    )
