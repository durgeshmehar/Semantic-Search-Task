"""Upload endpoints: create, send chunks, complete, check status, list.

The chunk handler is the hot path and does only bounded work -- append bytes,
split complete lines, insert queue rows -- then returns. Embedding happens in
background workers, so the client is never made to wait on CPU-bound work.

Resume needs no special endpoint: GET /files/{id}/status reports
`bytes_received`, and the client continues from there.

Every route requires an X-User-Id header (see app/identity.py) and scopes
reads/writes to that caller -- one user cannot see, search, or delete another
user's files.
"""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from . import config, db, models, storage, vector_store
from .identity import get_user_id
from .pipeline import job_queue
from .pipeline.line_buffer import LineBuffer

router = APIRouter(tags=["uploads"])


def _row_to_status(row) -> models.FileStatus:
    total_size = row["total_size"]
    chunks_total = row["chunks_total"]
    settled = row["chunks_indexed"] + row["chunks_failed"]

    return models.FileStatus(
        file_id=row["file_id"],
        owner_id=row["owner_id"],
        filename=row["filename"],
        total_size=total_size,
        bytes_received=row["bytes_received"],
        upload_status=row["upload_status"],
        # Clamped: total_size is a client-supplied hint that actual bytes can
        # exceed (see complete_upload), and progress is reported as a
        # fraction in [0, 1].
        upload_progress=min(row["bytes_received"] / total_size, 1.0) if total_size else 1.0,
        processing_status=row["processing_status"],
        # Measured against passages discovered so far, so it climbs during the
        # upload instead of sitting at zero until the last byte lands.
        processing_progress=(settled / chunks_total) if chunks_total else 0.0,
        chunks_total=chunks_total,
        chunks_indexed=row["chunks_indexed"],
        chunks_failed=row["chunks_failed"],
        searchable=row["chunks_indexed"] > 0,
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _fetch_owned_file(file_id: str, user_id: str):
    """Look up a file, enforcing that the caller owns it.

    A 404 (rather than 403) for a file that exists but belongs to someone else
    is deliberate: it doesn't confirm to a guessing client that the file_id is
    valid, which is the same reasoning most APIs use for object-level access
    control.
    """
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()
    if row is None or row["owner_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}")
    return row


@router.post(
    "/files",
    response_model=models.CreateUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new upload",
)
def create_upload(
    payload: models.CreateUploadRequest,
    user_id: str = Depends(get_user_id),
) -> models.CreateUploadResponse:
    """Reserve a file_id. Send the bytes with PUT /files/{file_id}/chunk.

    `total_size` is the client's stated size, used to size passages and report
    progress -- it is a hint, not what decides completion. The upload is only
    marked complete when the client calls POST /files/{file_id}/complete,
    since a client that mis-declares total_size must not leave the file stuck
    forever waiting for bytes that will never arrive.
    """
    file_id = uuid.uuid4().hex
    now = time.time()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO files
                (file_id, owner_id, filename, total_size, upload_status,
                 processing_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', 'pending', ?, ?)
            """,
            (file_id, user_id, payload.filename, payload.total_size, now, now),
        )

    return models.CreateUploadResponse(
        file_id=file_id,
        filename=payload.filename,
        total_size=payload.total_size,
        chunk_size=config.MAX_CHUNK_BYTES,
        upload_status="pending",
    )


@router.put(
    "/files/{file_id}/chunk",
    response_model=models.ChunkUploadResponse,
    summary="Upload one chunk at a byte offset",
    responses={
        404: {"model": models.ErrorResponse, "description": "Unknown file"},
        409: {"model": models.ErrorResponse, "description": "Offset mismatch"},
        413: {"model": models.ErrorResponse, "description": "Chunk too large"},
    },
)
async def upload_chunk(
    file_id: str,
    request: Request,
    offset: int = Query(
        ...,
        ge=0,
        description="Byte offset this chunk starts at. Must equal the server's "
        "current bytes_received.",
    ),
    user_id: str = Depends(get_user_id),
) -> models.ChunkUploadResponse:
    """Append one chunk.

    The offset must match what the server already holds. That check makes
    retries idempotent (re-sending a chunk whose response was lost is rejected
    rather than duplicated) and stops out-of-order chunks from corrupting the
    file.

    The read-check-append-write sequence runs inside one SQLite transaction so
    two concurrent requests for the same file (a client retry racing the
    original attempt, for instance) can't both pass the offset check against
    the same starting value: SQLite's BEGIN IMMEDIATE takes the database's
    write lock up front, so the second request blocks until the first commits,
    then sees the advanced bytes_received and correctly fails its own check
    rather than double-appending.
    """
    body = await request.body()
    if len(body) > config.MAX_CHUNK_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"chunk of {len(body)} bytes exceeds limit of {config.MAX_CHUNK_BYTES}",
        )

    with db.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if row is None or row["owner_id"] != user_id:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}"
            )

        if row["upload_status"] in ("completed", "finalizing"):
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"upload already {row['upload_status']}"
            )

        expected = row["bytes_received"]
        if offset != expected:
            # Tell the client exactly where to resume rather than just refusing.
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"offset mismatch: expected {expected}, got {offset}. "
                f"Resume from {expected}.",
            )

        # total_size is a hint (see complete_upload's docstring), not a hard
        # ceiling -- a client may legitimately exceed its own earlier
        # estimate. MAX_FILE_BYTES is the real, non-negotiable cap.
        if expected + len(body) > config.MAX_FILE_BYTES:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"upload would exceed the maximum allowed size of "
                f"{config.MAX_FILE_BYTES} bytes",
            )

        # The file on disk is the source of truth for how much we really have.
        # If a previous request died between the write and the commit, the
        # file may be ahead of the database; trim it back so appends stay
        # aligned. Still inside the transaction: another request cannot have
        # advanced bytes_received without first taking this same write lock.
        on_disk = storage.current_size(file_id)
        if on_disk != expected:
            storage.truncate_to(file_id, expected)

        storage.append_chunk(file_id, body)
        new_size = expected + len(body)

        # Split into passages from the bytes already in memory -- the uploaded
        # file is never re-read to build the index. Passage size is chosen
        # from the client's declared total_size purely to pick a sensible
        # target; it has no bearing on when the upload is considered done.
        target, maximum, overlap = config.passage_size_for(row["total_size"])
        buffer = LineBuffer(
            start_offset=expected,
            pending_tail=bytes(row["pending_tail"]),
            target_bytes=target,
            max_bytes=maximum,
            overlap_bytes=overlap,
        )
        passages = buffer.feed(body)

        now = time.time()
        next_sequence = job_queue.enqueue(
            conn, file_id, passages, row["next_sequence"]
        )
        conn.execute(
            """
            UPDATE files
               SET bytes_received = ?,
                   upload_status = 'uploading',
                   chunks_total = chunks_total + ?,
                   next_sequence = ?,
                   pending_tail = ?,
                   updated_at = ?
             WHERE file_id = ?
            """,
            (new_size, len(passages), next_sequence, buffer.pending_tail, now, file_id),
        )

    return models.ChunkUploadResponse(
        file_id=file_id,
        bytes_received=new_size,
        total_size=row["total_size"],
        upload_status="uploading",
        chunks_enqueued=next_sequence,
    )


@router.post(
    "/files/{file_id}/complete",
    response_model=models.FileStatus,
    summary="Mark an upload finished",
    responses={
        404: {"model": models.ErrorResponse, "description": "Unknown file"},
        409: {"model": models.ErrorResponse, "description": "No bytes received yet"},
    },
)
def complete_upload(
    file_id: str, user_id: str = Depends(get_user_id)
) -> models.FileStatus:
    """Finalize the upload once the client has sent every chunk.

    Completion is an explicit client action rather than inferred from
    `bytes_received >= total_size`: `total_size` is client-declared and
    unverified, so trusting it as the sole completion signal means a client
    that over-states its file's size leaves the upload stuck in `uploading`
    forever, with no bytes left to send and no way to finish. This endpoint is
    the "I am done sending bytes" signal the offset protocol alone can't give.

    The `finalizing` status closes the same race the chunk handler guards
    against: it is set inside this transaction before the rename, so a
    concurrent chunk PUT sees it and is rejected rather than appending to a
    file this request is mid-way through renaming.
    """
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if row is None or row["owner_id"] != user_id:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}"
            )

        if row["upload_status"] == "completed":
            return _row_to_status(row)

        if row["bytes_received"] == 0:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "no bytes received yet"
            )

        # Flush whatever's left in the line buffer -- the final line usually
        # has no trailing newline, so it would otherwise never be emitted.
        buffer = LineBuffer(
            start_offset=row["bytes_received"],
            pending_tail=bytes(row["pending_tail"]),
        )
        passages = buffer.flush()
        next_sequence = job_queue.enqueue(
            conn, file_id, passages, row["next_sequence"]
        )

        now = time.time()
        conn.execute(
            """
            UPDATE files
               SET upload_status = 'finalizing',
                   chunks_total = chunks_total + ?,
                   next_sequence = ?,
                   pending_tail = x'',
                   total_size = ?,
                   updated_at = ?
             WHERE file_id = ?
            """,
            (len(passages), next_sequence, row["bytes_received"], now, file_id),
        )

    # The rename happens outside the transaction (it's a filesystem call, not
    # a database one) but after upload_status is already 'finalizing', so a
    # chunk PUT racing this request sees that status and is rejected before it
    # can touch a file this call is in the middle of renaming.
    storage.finalize(file_id)

    with db.transaction() as conn:
        conn.execute(
            "UPDATE files SET upload_status = 'completed', updated_at = ? WHERE file_id = ?",
            (time.time(), file_id),
        )
        row = conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()

    return _row_to_status(row)


@router.get(
    "/files/{file_id}/status",
    response_model=models.FileStatus,
    summary="Upload and processing progress",
)
def get_status(file_id: str, user_id: str = Depends(get_user_id)) -> models.FileStatus:
    """Report both states.

    This doubles as the resume endpoint: send the next chunk from
    `bytes_received`.
    """
    return _row_to_status(_fetch_owned_file(file_id, user_id))


@router.get(
    "/files",
    response_model=models.FileListResponse,
    summary="List your uploads",
)
def list_files(
    limit: int = Query(default=100, ge=1, le=1000),
    user_id: str = Depends(get_user_id),
) -> models.FileListResponse:
    """Files owned by the caller, not every file on the service."""
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM files WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return models.FileListResponse(files=[_row_to_status(row) for row in rows])


@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file and its index",
)
def delete_file(file_id: str, user_id: str = Depends(get_user_id)) -> None:
    _fetch_owned_file(file_id, user_id)
    with db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    storage.delete_all(file_id)
    vector_store.drop(file_id)
