"""Upload endpoints: create, send chunks, check status, list.

The chunk handler is the hot path and does only bounded work -- append bytes,
split complete lines, insert queue rows -- then returns. Embedding happens in
background workers, so the client is never made to wait on CPU-bound work.

Resume needs no special endpoint: GET /files/{id}/status reports
`bytes_received`, and the client continues from there.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status

from . import config, db, models, storage, vector_store
from .pipeline import job_queue
from .pipeline.line_buffer import LineBuffer

router = APIRouter(tags=["uploads"])


def _row_to_status(row) -> models.FileStatus:
    total_size = row["total_size"]
    chunks_total = row["chunks_total"]
    settled = row["chunks_indexed"] + row["chunks_failed"]

    return models.FileStatus(
        file_id=row["file_id"],
        filename=row["filename"],
        total_size=total_size,
        bytes_received=row["bytes_received"],
        upload_status=row["upload_status"],
        upload_progress=(row["bytes_received"] / total_size) if total_size else 1.0,
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


def _fetch_file(file_id: str):
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}")
    return row


@router.post(
    "/files",
    response_model=models.CreateUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new upload",
)
def create_upload(payload: models.CreateUploadRequest) -> models.CreateUploadResponse:
    """Reserve a file_id. Send the bytes with PUT /files/{file_id}/chunk."""
    file_id = uuid.uuid4().hex
    now = time.time()

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO files
                (file_id, filename, total_size, upload_status, processing_status,
                 created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 'pending', ?, ?)
            """,
            (file_id, payload.filename, payload.total_size, now, now),
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
) -> models.ChunkUploadResponse:
    """Append one chunk.

    The offset must match what the server already holds. That check makes
    retries idempotent (re-sending a chunk whose response was lost is rejected
    rather than duplicated) and stops out-of-order chunks from corrupting the
    file.
    """
    row = _fetch_file(file_id)

    if row["upload_status"] == "completed":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "upload already completed"
        )

    body = await request.body()
    if len(body) > config.MAX_CHUNK_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"chunk of {len(body)} bytes exceeds limit of {config.MAX_CHUNK_BYTES}",
        )

    expected = row["bytes_received"]
    if offset != expected:
        # Tell the client exactly where to resume rather than just refusing.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"offset mismatch: expected {expected}, got {offset}. "
            f"Resume from {expected}.",
        )

    if expected + len(body) > row["total_size"]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"chunk would exceed declared total_size of {row['total_size']}",
        )

    # The file on disk is the source of truth for how much we really have. If a
    # previous request died between the write and the commit, the file may be
    # ahead of the database; trim it back so appends stay aligned.
    on_disk = storage.current_size(file_id)
    if on_disk != expected:
        storage.truncate_to(file_id, expected)

    storage.append_chunk(file_id, body)
    new_size = expected + len(body)
    is_final = new_size >= row["total_size"]

    # Split into passages from the bytes already in memory -- the uploaded file
    # is never re-read to build the index.
    #
    # Passage size is chosen from the declared file size so a very large upload
    # doesn't produce more vectors than the memory budget allows. See
    # config.passage_size_for.
    target, maximum, overlap = config.passage_size_for(row["total_size"])
    buffer = LineBuffer(
        start_offset=expected,
        pending_tail=bytes(row["pending_tail"]),
        target_bytes=target,
        max_bytes=maximum,
        overlap_bytes=overlap,
    )
    passages = buffer.feed(body)
    if is_final:
        passages.extend(buffer.flush())

    now = time.time()
    with db.transaction() as conn:
        next_sequence = job_queue.enqueue(
            conn, file_id, passages, row["next_sequence"]
        )
        conn.execute(
            """
            UPDATE files
               SET bytes_received = ?,
                   upload_status = ?,
                   chunks_total = chunks_total + ?,
                   next_sequence = ?,
                   pending_tail = ?,
                   updated_at = ?
             WHERE file_id = ?
            """,
            (
                new_size,
                "completed" if is_final else "uploading",
                len(passages),
                next_sequence,
                b"" if is_final else buffer.pending_tail,
                now,
                file_id,
            ),
        )

    if is_final:
        storage.finalize(file_id)

    return models.ChunkUploadResponse(
        file_id=file_id,
        bytes_received=new_size,
        total_size=row["total_size"],
        upload_status="completed" if is_final else "uploading",
        chunks_enqueued=next_sequence,
    )


@router.get(
    "/files/{file_id}/status",
    response_model=models.FileStatus,
    summary="Upload and processing progress",
)
def get_status(file_id: str) -> models.FileStatus:
    """Report both states.

    This doubles as the resume endpoint: send the next chunk from
    `bytes_received`.
    """
    return _row_to_status(_fetch_file(file_id))


@router.get(
    "/files",
    response_model=models.FileListResponse,
    summary="List uploads",
)
def list_files(limit: int = Query(default=100, ge=1, le=1000)) -> models.FileListResponse:
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM files ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return models.FileListResponse(files=[_row_to_status(row) for row in rows])


@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file and its index",
)
def delete_file(file_id: str) -> None:
    _fetch_file(file_id)
    with db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    storage.delete_all(file_id)
    vector_store.drop(file_id)
