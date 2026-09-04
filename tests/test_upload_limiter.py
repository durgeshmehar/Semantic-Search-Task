"""The concurrent-upload cap.

Every other expensive resource here has a fixed ceiling (MAX_CHUNK_BYTES per
request, WORKER_COUNT for embedding); accepting chunk uploads had none, so an
unbounded burst of concurrent PUTs could add up past the container's memory
limit with nothing pushing back. These tests exercise the semaphore directly,
since a real HTTP round-trip is too fast to reliably collide in a test without
an artificial delay.
"""

import contextlib
import threading
import time

import pytest

from app import upload_limiter


@pytest.fixture(autouse=True)
def small_limit(monkeypatch):
    """A tiny cap makes the tests fast and the collision deterministic."""
    monkeypatch.setattr(upload_limiter.config, "MAX_CONCURRENT_UPLOADS", 2)
    monkeypatch.setattr(upload_limiter, "_semaphore", threading.Semaphore(2))


def test_acquire_succeeds_under_the_limit():
    with upload_limiter.acquire():
        pass  # no exception


def test_third_concurrent_acquire_is_rejected():
    """Slots 1 and 2 succeed and are held open; the 3rd is turned away."""
    with contextlib.ExitStack() as held:
        held.enter_context(upload_limiter.acquire())
        held.enter_context(upload_limiter.acquire())

        with pytest.raises(upload_limiter.TooManyConcurrentUploads):
            upload_limiter.acquire().__enter__()


def test_releasing_frees_a_slot_for_the_next_caller():
    with upload_limiter.acquire():
        with upload_limiter.acquire():
            # Both slots held; a third must fail.
            with pytest.raises(upload_limiter.TooManyConcurrentUploads):
                upload_limiter.acquire().__enter__()
        # Outer `with` released one slot on exit from the inner block.
        with upload_limiter.acquire():
            pass  # succeeds now that a slot is free


def test_a_released_slot_can_be_reacquired_repeatedly():
    for _ in range(5):  # far more than the limit of 2, one at a time
        with upload_limiter.acquire():
            pass


def test_concurrent_threads_never_exceed_the_limit():
    """The real property that matters: peak concurrent holders never exceeds the cap."""
    active = 0
    peak = 0
    rejected = 0
    lock = threading.Lock()
    started = threading.Barrier(6)  # line every thread up before any proceeds

    def worker():
        nonlocal active, peak, rejected
        started.wait()
        try:
            with upload_limiter.acquire():
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
        except upload_limiter.TooManyConcurrentUploads:
            with lock:
                rejected += 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= 2
    # With 6 threads racing for 2 slots held 50ms each, at least some must
    # have been rejected rather than silently let through.
    assert rejected > 0
