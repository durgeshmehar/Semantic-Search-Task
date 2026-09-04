"""On-disk layout and byte-range reads.

An upload lands in `{file_id}.partial` and is renamed to `{file_id}.dat` when
complete. The rename is atomic, so the final file either exists whole or not at
all -- a reader can never observe a half-written file under the final name.

That final file is also the text store: chunk rows hold byte offsets into it,
and search reads passages back with seek()+read rather than keeping a second
copy of the corpus in the database.

Vector storage lives in Qdrant (see vector_store.py), not on this filesystem.
"""

from pathlib import Path

from . import config


def partial_path(file_id: str) -> Path:
    """Where an in-progress upload accumulates."""
    return config.UPLOAD_DIR / f"{file_id}.partial"


def final_path(file_id: str) -> Path:
    """Where a completed upload lives."""
    return config.UPLOAD_DIR / f"{file_id}.dat"


def readable_path(file_id: str) -> Path | None:
    """The file to read passages from, whichever stage it's at.

    Passages become searchable during the upload, so search has to work against
    the .partial file too.
    """
    final = final_path(file_id)
    if final.exists():
        return final
    partial = partial_path(file_id)
    if partial.exists():
        return partial
    return None


def append_chunk(file_id: str, data: bytes) -> int:
    """Append bytes to the partial upload, returning the new size.

    Opened per call rather than held open: an upload may be resumed by a
    different worker process, and a dropped connection shouldn't leak a handle.
    """
    path = partial_path(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(data)
        return fh.tell()


def current_size(file_id: str) -> int:
    """Bytes on disk for this upload -- the source of truth for resuming."""
    path = partial_path(file_id)
    if path.exists():
        return path.stat().st_size
    final = final_path(file_id)
    if final.exists():
        return final.stat().st_size
    return 0


def finalize(file_id: str) -> Path:
    """Atomically promote the completed upload to its final name."""
    partial = partial_path(file_id)
    final = final_path(file_id)
    if final.exists():
        return final
    partial.rename(final)
    return final


def truncate_to(file_id: str, size: int) -> None:
    """Trim a partial upload back to `size`.

    Used when the recorded byte count and the file on disk disagree, which can
    happen if the process died between the write and the metadata commit. The
    database is authoritative; extra bytes on disk are discarded.
    """
    path = partial_path(file_id)
    if path.exists() and path.stat().st_size > size:
        with open(path, "r+b") as fh:
            fh.truncate(size)


def read_range(file_id: str, start: int, end: int) -> str:
    """Read a passage back as text.

    A range may begin or end mid-character when it came from a long-line split,
    so decoding replaces malformed bytes rather than raising.
    """
    path = readable_path(file_id)
    if path is None:
        return ""
    with open(path, "rb") as fh:
        fh.seek(start)
        raw = fh.read(max(0, end - start))
    return raw.decode("utf-8", errors="replace")


def delete_all(file_id: str) -> None:
    """Remove this file's on-disk artifacts.

    Does not touch Qdrant -- callers drop the vector collection separately via
    vector_store.drop(), since that's a network call rather than a local one.
    """
    for path in (partial_path(file_id), final_path(file_id)):
        path.unlink(missing_ok=True)
