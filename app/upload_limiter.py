"""Caps how many chunk uploads may be held in memory at once.

Every other expensive resource in this service has a fixed ceiling:
MAX_CHUNK_BYTES bounds one request, WORKER_COUNT bounds concurrent embedding.
Accepting chunk uploads had no equivalent -- nothing stopped an unbounded
number of PUT /chunk requests from each holding their own chunk in memory
simultaneously. A burst of concurrent uploads (a broken client retry loop, or
deliberate load) could add up past the container's memory limit with nothing
in the code pushing back, and a container OOM-kill takes down every in-flight
request, not just the excess ones.

A semaphore makes upload-acceptance behave the same way as embedding already
does: a fixed, known number of slots. The (MAX_CONCURRENT_UPLOADS + 1)th
concurrent chunk request gets a 503 naming a retry delay instead of adding to
memory pressure -- a real, retryable answer rather than an unbounded queue or
a silent slowdown.
"""

import threading

from . import config

_semaphore = threading.Semaphore(config.MAX_CONCURRENT_UPLOADS)

# Sent as the Retry-After header so a well-behaved client backs off instead of
# retrying immediately into the same full set of slots.
RETRY_AFTER_SECONDS = 2


class TooManyConcurrentUploads(Exception):
    """Raised when no upload slot is free. Callers map this to HTTP 503."""


class UploadSlot:
    """A held slot, released automatically via `with upload_limiter.acquire():`."""

    def __enter__(self) -> "UploadSlot":
        if not _semaphore.acquire(blocking=False):
            raise TooManyConcurrentUploads(
                f"server is handling {config.MAX_CONCURRENT_UPLOADS} uploads "
                "already; retry shortly"
            )
        return self

    def __exit__(self, *exc_info) -> None:
        _semaphore.release()


def acquire() -> UploadSlot:
    """Reserve one of the fixed upload slots for the duration of a `with` block."""
    return UploadSlot()
