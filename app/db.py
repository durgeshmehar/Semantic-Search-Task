"""SQLite access layer.

SQLite plays two roles here: it stores file/chunk metadata, and the `chunks`
table doubles as the durable job queue that feeds the indexing workers (see
pipeline/job_queue.py). WAL mode is what makes that work -- the upload path can
write while workers read, without blocking each other.

Note what `chunks` does NOT store: the passage text. Chunks hold byte offsets
into the uploaded file, which is already on disk and immutable once written.
Storing the text too would duplicate the entire corpus.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from . import config

# SQLite connections can't be shared across threads, and this service has
# several (the request path plus a worker pool). Each thread gets its own.
_local = threading.local()


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id            TEXT PRIMARY KEY,
    -- Client-supplied via X-User-Id (see app/identity.py). Not authenticated --
    -- there is no login -- but it scopes listing/search/delete so one user
    -- can't act on another's files, and it is what a real auth layer would
    -- slot into later without changing this schema.
    owner_id           TEXT NOT NULL,
    filename           TEXT NOT NULL,
    total_size         INTEGER NOT NULL,
    bytes_received     INTEGER NOT NULL DEFAULT 0,
    upload_status      TEXT NOT NULL DEFAULT 'pending',
    processing_status  TEXT NOT NULL DEFAULT 'pending',
    chunks_total       INTEGER NOT NULL DEFAULT 0,
    chunks_indexed     INTEGER NOT NULL DEFAULT 0,
    chunks_failed      INTEGER NOT NULL DEFAULT 0,
    -- How far indexing has progressed, in bytes. Lets indexing resume from an
    -- exact position after a crash rather than guessing from enqueued rows.
    indexed_watermark  INTEGER NOT NULL DEFAULT 0,
    -- Partial trailing line held back by the chunker, persisted so an upload
    -- resumed after a restart doesn't lose or duplicate the split line.
    pending_tail       BLOB NOT NULL DEFAULT x'',
    next_sequence      INTEGER NOT NULL DEFAULT 0,
    error_message      TEXT,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         TEXT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    sequence        INTEGER NOT NULL,
    start_byte      INTEGER NOT NULL,
    end_byte        INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

-- The queue's hot path: "give me the oldest pending work".
CREATE INDEX IF NOT EXISTS idx_chunks_queue
    ON chunks(status, file_id, sequence);

-- GET /files scopes its listing to the caller.
CREATE INDEX IF NOT EXISTS idx_files_owner
    ON files(owner_id, created_at);

-- Point IDs in Qdrant are derived from (file_id, sequence), so this is also
-- the uniqueness the vector store relies on to make re-indexing idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_sequence
    ON chunks(file_id, sequence);
"""


def get_connection() -> sqlite3.Connection:
    """Return this thread's connection, creating it on first use."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            config.DB_PATH,
            timeout=30.0,  # wait rather than fail when a writer holds the lock
            isolation_level=None,  # explicit transactions, no implicit BEGIN
        )
        conn.row_factory = sqlite3.Row
        # WAL lets readers and a writer proceed concurrently, which is the
        # whole reason uploads don't stall behind indexing workers.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Run a block inside an IMMEDIATE transaction.

    IMMEDIATE takes the write lock up front, so two workers claiming jobs
    can't interleave into a lost update.
    """
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init_db() -> None:
    """Create the schema. Safe to call on every startup."""
    conn = get_connection()
    conn.executescript(SCHEMA)


def close_connection() -> None:
    """Close this thread's connection, if it opened one."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
