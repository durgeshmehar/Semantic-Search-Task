r"""Durable job queue backed by the `chunks` table.

Why SQLite rather than an in-memory queue: work already enqueued must survive a
process crash. A `queue.Queue` loses it. Why not SQS/Redis: they'd add a
service dependency to something that has to run locally, and SQLite is already
here holding the metadata.

The lifecycle of a chunk row:

    pending -> processing -> indexed
                         \-> pending (retry, under MAX_RETRIES)
                         \-> failed  (retries exhausted)

Rows left in `processing` by a crash are reset to `pending` on startup, so the
only cost of an unclean shutdown is re-embedding a handful of passages.

The interface here is deliberately small -- claim / complete / fail / recover --
so swapping in SQS or Redis at scale is a contained change.
"""

import time
from dataclasses import dataclass

from .. import config, db


@dataclass(frozen=True)
class ChunkJob:
    """One passage awaiting embedding."""

    chunk_id: int
    file_id: str
    sequence: int
    start_byte: int
    end_byte: int
    retry_count: int


def enqueue(
    conn,
    file_id: str,
    passages: list,
    start_sequence: int,
) -> int:
    """Insert passage ranges as pending jobs. Returns the next free sequence.

    Called from the upload path, so it must stay cheap: one executemany, no
    text, no embedding.
    """
    if not passages:
        return start_sequence

    now = time.time()
    rows = [
        (
            file_id,
            start_sequence + offset,
            passage.start_byte,
            passage.end_byte,
            now,
            now,
        )
        for offset, passage in enumerate(passages)
    ]
    conn.executemany(
        """
        INSERT INTO chunks
            (file_id, sequence, start_byte, end_byte, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id, sequence) DO NOTHING
        """,
        rows,
    )
    return start_sequence + len(passages)


def claim_batch(limit: int | None = None) -> list[ChunkJob]:
    """Atomically take up to `limit` pending jobs for this worker.

    Grouped by file so a batch shares one FAISS index, and ordered by sequence
    so vector positions stay monotonic within a file.

    The SELECT and UPDATE run inside one IMMEDIATE transaction; without that,
    two workers could read the same rows and embed them twice.
    """
    limit = limit or config.CLAIM_BATCH_SIZE

    with db.transaction() as conn:
        # Pick the file with the oldest pending work, then take a run of its
        # chunks -- keeping a batch to a single index.
        target = conn.execute(
            """
            SELECT file_id
              FROM chunks
             WHERE status = 'pending'
             ORDER BY chunk_id
             LIMIT 1
            """
        ).fetchone()
        if target is None:
            return []

        rows = conn.execute(
            """
            SELECT chunk_id, file_id, sequence, start_byte, end_byte, retry_count
              FROM chunks
             WHERE status = 'pending' AND file_id = ?
             ORDER BY sequence
             LIMIT ?
            """,
            (target["file_id"], limit),
        ).fetchall()
        if not rows:
            return []

        ids = [row["chunk_id"] for row in rows]
        placeholders = ",".join("?" * len(ids))
        now = time.time()
        conn.execute(
            f"""
            UPDATE chunks
               SET status = 'processing', updated_at = ?
             WHERE chunk_id IN ({placeholders})
            """,
            [now, *ids],
        )

        conn.execute(
            """
            UPDATE files
               SET processing_status = CASE
                       WHEN processing_status = 'pending' THEN 'processing'
                       ELSE processing_status
                   END,
                   updated_at = ?
             WHERE file_id = ?
            """,
            (now, target["file_id"]),
        )

        return [
            ChunkJob(
                chunk_id=row["chunk_id"],
                file_id=row["file_id"],
                sequence=row["sequence"],
                start_byte=row["start_byte"],
                end_byte=row["end_byte"],
                retry_count=row["retry_count"],
            )
            for row in rows
        ]


def complete_batch(jobs: list[ChunkJob]) -> None:
    """Mark jobs indexed.

    Byte ranges are looked up by (file_id, sequence) rather than a stored
    vector position: Qdrant carries start_byte/end_byte as point payload, so a
    search result maps directly to a byte range without a join back here.
    """
    if not jobs:
        return

    now = time.time()
    with db.transaction() as conn:
        conn.executemany(
            """
            UPDATE chunks
               SET status = 'indexed', updated_at = ?
             WHERE chunk_id = ?
            """,
            [(now, job.chunk_id) for job in jobs],
        )

        file_id = jobs[0].file_id
        watermark = max(job.end_byte for job in jobs)
        conn.execute(
            """
            UPDATE files
               SET chunks_indexed = chunks_indexed + ?,
                   indexed_watermark = MAX(indexed_watermark, ?),
                   updated_at = ?
             WHERE file_id = ?
            """,
            (len(jobs), watermark, now, file_id),
        )
        _refresh_processing_status(conn, file_id, now)


def fail_batch(jobs: list[ChunkJob], error: str) -> None:
    """Return jobs to the queue, or mark them failed once retries run out.

    A permanently failed passage doesn't fail the file: the rest stays
    searchable, and the error is recorded for the status endpoint.
    """
    if not jobs:
        return

    now = time.time()
    truncated = error[:500]
    retryable = [j for j in jobs if j.retry_count + 1 < config.MAX_RETRIES]
    exhausted = [j for j in jobs if j.retry_count + 1 >= config.MAX_RETRIES]

    with db.transaction() as conn:
        if retryable:
            conn.executemany(
                """
                UPDATE chunks
                   SET status = 'pending',
                       retry_count = retry_count + 1,
                       error_message = ?,
                       updated_at = ?
                 WHERE chunk_id = ?
                """,
                [(truncated, now, job.chunk_id) for job in retryable],
            )
        if exhausted:
            conn.executemany(
                """
                UPDATE chunks
                   SET status = 'failed',
                       retry_count = retry_count + 1,
                       error_message = ?,
                       updated_at = ?
                 WHERE chunk_id = ?
                """,
                [(truncated, now, job.chunk_id) for job in exhausted],
            )

        file_id = jobs[0].file_id
        if exhausted:
            conn.execute(
                """
                UPDATE files
                   SET chunks_failed = chunks_failed + ?,
                       error_message = ?,
                       updated_at = ?
                 WHERE file_id = ?
                """,
                (len(exhausted), truncated, now, file_id),
            )
        _refresh_processing_status(conn, file_id, now)


def _refresh_processing_status(conn, file_id: str, now: float) -> None:
    """Move a file to a terminal processing state once nothing is outstanding.

    Only meaningful after the upload itself finishes -- while bytes are still
    arriving, an empty queue just means indexing has caught up.
    """
    row = conn.execute(
        """
        SELECT upload_status, chunks_total, chunks_indexed, chunks_failed
          FROM files
         WHERE file_id = ?
        """,
        (file_id,),
    ).fetchone()
    if row is None or row["upload_status"] != "completed":
        return

    settled = row["chunks_indexed"] + row["chunks_failed"]
    if settled < row["chunks_total"]:
        return

    status = "completed" if row["chunks_failed"] == 0 else "failed"
    conn.execute(
        "UPDATE files SET processing_status = ?, updated_at = ? WHERE file_id = ?",
        (status, now, file_id),
    )


def recover_stuck_jobs() -> int:
    """Reset rows and files abandoned mid-flight by a crash. Called on startup.

    Safe to run unconditionally here specifically because it runs during
    FastAPI's lifespan startup, before the server accepts any connections (see
    app/main.py) -- so "upload_status = 'uploading'" at this exact moment
    cannot mean a live client is mid-request, only that the previous process
    died holding that state. That precondition would not hold if this were
    ever called from a live request path or a multi-replica deployment sharing
    this database; at that scale, ownership of "is this upload actually still
    active" needs a heartbeat or lease rather than inference from a status
    string (see README section 6).
    """
    now = time.time()
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE chunks
               SET status = 'pending', updated_at = ?
             WHERE status = 'processing'
            """,
            (now,),
        )
        recovered = cursor.rowcount

        # An upload interrupted mid-flight is no longer being written to by any
        # live connection; mark it so the client knows to resume.
        conn.execute(
            """
            UPDATE files
               SET upload_status = 'interrupted', updated_at = ?
             WHERE upload_status = 'uploading'
            """,
            (now,),
        )

        # A crash between marking 'finalizing' and completing the rename
        # leaves the file mid-way through POST /complete. Since chunk PUTs
        # already refuse a 'finalizing' file, the only way forward is to
        # finish what complete() was doing, so put it back in 'uploading' and
        # let the client call /complete again -- it's idempotent past the
        # rename (storage.finalize() is a no-op if .dat already exists).
        conn.execute(
            """
            UPDATE files
               SET upload_status = 'uploading', updated_at = ?
             WHERE upload_status = 'finalizing'
            """,
            (now,),
        )
    return recovered


def pending_count(file_id: str | None = None) -> int:
    """Jobs still queued, for tests and the status endpoint."""
    conn = db.get_connection()
    if file_id is None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE status IN ('pending', 'processing')"
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
              FROM chunks
             WHERE status IN ('pending', 'processing') AND file_id = ?
            """,
            (file_id,),
        ).fetchone()
    return row["n"]
