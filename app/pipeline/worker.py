"""Background workers that turn queued passages into searchable vectors.

This is the expensive half of the pipeline, deliberately kept off the request
path: the upload handler only appends bytes and inserts rows, while these
threads do the embedding. That separation is what lets an upload run at disk
speed regardless of how far behind indexing is.

Worker count is intentionally small (2 by default). The model is CPU-bound, so
more threads add memory without adding much throughput.
"""

import logging
import threading

from .. import config, db, storage, vector_store
from . import embeddings, job_queue

logger = logging.getLogger(__name__)


class IndexingWorker(threading.Thread):
    """Claims batches of passages, embeds them, adds them to the index."""

    def __init__(self, name: str, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self._stop = stop_event

    def run(self) -> None:
        logger.info("worker %s started", self.name)
        while not self._stop.is_set():
            try:
                processed = self._process_one_batch()
            except Exception:
                # A worker must never die: the queue would silently stop
                # draining and uploads would appear to hang forever.
                logger.exception("worker %s hit an unexpected error", self.name)
                processed = False

            if not processed:
                # Nothing queued -- wait, but stay responsive to shutdown.
                self._stop.wait(config.WORKER_POLL_SECONDS)

        db.close_connection()
        logger.info("worker %s stopped", self.name)

    def _process_one_batch(self) -> bool:
        """Handle one claimed batch. Returns False when the queue was empty."""
        jobs = job_queue.claim_batch()
        if not jobs:
            return False

        try:
            texts = [
                storage.read_range(job.file_id, job.start_byte, job.end_byte)
                for job in jobs
            ]

            # A truly empty read (file missing, or the range came back with
            # zero bytes) means the file was deleted mid-flight or the range
            # is bogus -- nothing to embed. This is deliberately narrower than
            # "falsy after strip()": a passage of blank lines between log
            # entries is legitimate content whose absence from search results
            # would be a silent, hard-to-notice gap, not a crash.
            keep = [(job, text) for job, text in zip(jobs, texts) if text]
            empty = [job for job, text in zip(jobs, texts) if not text]

            if empty:
                # Nothing to index, but they mustn't stay queued forever.
                job_queue.complete_batch(empty)

            if not keep:
                return True

            kept_jobs = [job for job, _ in keep]
            vectors = embeddings.embed_texts([text for _, text in keep])
            # Point IDs are derived from (file_id, sequence), so re-upserting
            # this batch after a crash mid-way overwrites the same points
            # rather than creating duplicates.
            vector_store.add_vectors(
                kept_jobs[0].file_id,
                vectors,
                sequences=[job.sequence for job in kept_jobs],
                start_bytes=[job.start_byte for job in kept_jobs],
                end_bytes=[job.end_byte for job in kept_jobs],
            )
            job_queue.complete_batch(kept_jobs)

        except Exception as exc:
            logger.exception("failed to index batch of %d chunks", len(jobs))
            job_queue.fail_batch(jobs, f"{type(exc).__name__}: {exc}")

        return True


class WorkerPool:
    """Owns the worker threads for the application's lifetime."""

    def __init__(self, count: int | None = None) -> None:
        self._count = count if count is not None else config.WORKER_COUNT
        self._stop = threading.Event()
        self._workers: list[IndexingWorker] = []

    def start(self) -> None:
        if self._workers:
            return
        self._stop.clear()
        for i in range(self._count):
            worker = IndexingWorker(f"indexer-{i}", self._stop)
            worker.start()
            self._workers.append(worker)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal shutdown and wait for workers to finish their current batch.

        In-flight rows stay in `processing`; startup recovery returns them to
        `pending`, so nothing is lost even on a hard kill.
        """
        self._stop.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()

    @property
    def running(self) -> bool:
        return any(worker.is_alive() for worker in self._workers)


_pool: WorkerPool | None = None


def start_workers() -> WorkerPool:
    global _pool
    if _pool is None:
        _pool = WorkerPool()
    _pool.start()
    return _pool


def stop_workers() -> None:
    global _pool
    if _pool is not None:
        _pool.stop()
        _pool = None


def drain_once() -> int:
    """Process the queue to empty on the calling thread.

    Used by tests, which need indexing to be deterministic rather than
    racing a background pool.
    """
    worker = IndexingWorker("drain", threading.Event())
    batches = 0
    while worker._process_one_batch():
        batches += 1
    return batches
