"""Durable queue semantics.

These are the guarantees the indexing pipeline relies on: work is claimed
exactly once, failures retry a bounded number of times, and a crash mid-batch
loses nothing.
"""

import time

from app import db
from app.pipeline import job_queue
from app.pipeline.line_buffer import PassageRange


def make_file(file_id: str = "f1", total_size: int = 1000, completed: bool = False):
    now = time.time()
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO files
                (file_id, filename, total_size, bytes_received, upload_status,
                 processing_status, created_at, updated_at)
            VALUES (?, 'test.log', ?, ?, ?, 'pending', ?, ?)
            """,
            (
                file_id,
                total_size,
                total_size if completed else 0,
                "completed" if completed else "uploading",
                now,
                now,
            ),
        )
    return file_id


def add_jobs(file_id: str, count: int, start: int = 0) -> None:
    passages = [PassageRange(i * 100, (i + 1) * 100) for i in range(count)]
    with db.transaction() as conn:
        job_queue.enqueue(conn, file_id, passages, start)
        conn.execute(
            "UPDATE files SET chunks_total = chunks_total + ? WHERE file_id = ?",
            (count, file_id),
        )


def test_enqueue_then_claim(isolated_env):
    file_id = make_file()
    add_jobs(file_id, 5)

    jobs = job_queue.claim_batch(limit=3)
    assert len(jobs) == 3
    # Ordered by sequence so vector positions stay monotonic.
    assert [j.sequence for j in jobs] == [0, 1, 2]
    assert job_queue.pending_count(file_id) == 5  # claimed, not yet done


def test_claim_is_exclusive(isolated_env):
    """Two claims must never return the same row -- double embedding."""
    file_id = make_file()
    add_jobs(file_id, 10)

    first = job_queue.claim_batch(limit=4)
    second = job_queue.claim_batch(limit=4)

    first_ids = {j.chunk_id for j in first}
    second_ids = {j.chunk_id for j in second}
    assert first_ids and second_ids
    assert first_ids.isdisjoint(second_ids)


def test_empty_queue_returns_nothing(isolated_env):
    assert job_queue.claim_batch() == []


def test_complete_marks_indexed_and_advances_watermark(isolated_env):
    file_id = make_file()
    add_jobs(file_id, 3)

    jobs = job_queue.claim_batch()
    job_queue.complete_batch(jobs, [0, 1, 2])

    conn = db.get_connection()
    row = conn.execute(
        "SELECT chunks_indexed, indexed_watermark FROM files WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    assert row["chunks_indexed"] == 3
    assert row["indexed_watermark"] == 300
    assert job_queue.pending_count(file_id) == 0


def test_failure_retries_then_gives_up(isolated_env, monkeypatch):
    """A bad chunk retries up to the limit, then fails without blocking others."""
    from app import config

    monkeypatch.setattr(config, "MAX_RETRIES", 3)

    file_id = make_file()
    add_jobs(file_id, 1)

    for attempt in range(2):
        jobs = job_queue.claim_batch()
        assert len(jobs) == 1, f"should still be retryable on attempt {attempt}"
        job_queue.fail_batch(jobs, "boom")

    # Third failure exhausts the retry budget.
    jobs = job_queue.claim_batch()
    assert len(jobs) == 1
    job_queue.fail_batch(jobs, "boom")

    assert job_queue.claim_batch() == []  # no longer retried

    conn = db.get_connection()
    row = conn.execute(
        "SELECT status, retry_count FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["retry_count"] == 3


def test_recover_resets_rows_left_in_flight(isolated_env):
    """A crash mid-batch must not strand work in 'processing'."""
    file_id = make_file()
    add_jobs(file_id, 4)

    claimed = job_queue.claim_batch()
    assert len(claimed) == 4

    # Simulate the process dying here -- rows are 'processing', nobody owns them.
    recovered = job_queue.recover_stuck_jobs()
    assert recovered == 4

    # They go back on the queue rather than being lost.
    again = job_queue.claim_batch()
    assert {j.chunk_id for j in again} == {j.chunk_id for j in claimed}


def test_recover_marks_live_uploads_interrupted(isolated_env):
    file_id = make_file()
    job_queue.recover_stuck_jobs()

    conn = db.get_connection()
    row = conn.execute(
        "SELECT upload_status FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()
    # The client needs to know to ask for a resume offset.
    assert row["upload_status"] == "interrupted"


def test_processing_completes_only_after_upload_finishes(isolated_env):
    """An empty queue mid-upload means caught up, not done."""
    file_id = make_file(completed=False)
    add_jobs(file_id, 2)

    jobs = job_queue.claim_batch()
    job_queue.complete_batch(jobs, [0, 1])

    conn = db.get_connection()
    status = conn.execute(
        "SELECT processing_status FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()["processing_status"]
    assert status == "processing"

    # Once the upload is done and the queue drains, it's genuinely complete.
    with db.transaction() as c:
        c.execute(
            "UPDATE files SET upload_status = 'completed' WHERE file_id = ?", (file_id,)
        )
    add_jobs(file_id, 1, start=2)
    jobs = job_queue.claim_batch()
    job_queue.complete_batch(jobs, [2])

    status = conn.execute(
        "SELECT processing_status FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()["processing_status"]
    assert status == "completed"


def test_enqueue_is_idempotent_on_sequence(isolated_env):
    """A retried chunk upload must not double-enqueue its passages."""
    file_id = make_file()
    passages = [PassageRange(0, 100), PassageRange(100, 200)]

    with db.transaction() as conn:
        job_queue.enqueue(conn, file_id, passages, 0)
    with db.transaction() as conn:
        job_queue.enqueue(conn, file_id, passages, 0)

    conn = db.get_connection()
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchone()["n"]
    assert count == 2


def test_jobs_from_multiple_files_are_batched_separately(isolated_env):
    """A batch shares one FAISS index, so it must not mix files."""
    make_file("fa")
    make_file("fb")
    add_jobs("fa", 3)
    add_jobs("fb", 3)

    jobs = job_queue.claim_batch(limit=10)
    assert len({j.file_id for j in jobs}) == 1
