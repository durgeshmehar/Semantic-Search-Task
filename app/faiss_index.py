"""Per-file FAISS index: append during upload, search on demand.

One index per file, because searches are always scoped to a single file and
per-file indexes keep uploads independent of each other.

Memory is the constraint that shapes this module. A 10 GB text file yields
roughly 1.1M passages; at float32 that is 1.1M x 384 x 4 B = ~1.7 GB, which
does not fit alongside the model and the rest of the service in 4 GB. Stored as
int8 the same index is ~420 MB, and reading it back memory-mapped means the
resident set is bounded by what searches actually touch rather than by the
index size. The cost is roughly 2-3% recall, which does not change which
sections come back for a natural-language query.

Small files skip quantization entirely -- a scalar quantizer needs training
data, and below a few thousand vectors a flat index is both simpler and
smaller.
"""

import threading

import faiss
import numpy as np

from . import config, storage

# One lock per file: appends from a worker and reads from a search request can
# overlap, and FAISS index objects are not thread-safe for concurrent writes.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(file_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(file_id)
        if lock is None:
            lock = threading.Lock()
            _locks[file_id] = lock
        return lock


def _new_index() -> faiss.Index:
    """A fresh flat index over normalised vectors (inner product = cosine)."""
    return faiss.IndexFlatIP(config.EMBEDDING_DIM)


def _load(file_id: str) -> faiss.Index:
    """Read this file's index from disk, or start a new one."""
    path = storage.index_path(file_id)
    if path.exists():
        return faiss.read_index(str(path))
    return _new_index()


def _save(file_id: str, index: faiss.Index) -> None:
    path = storage.index_path(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling then rename: a crash mid-write leaves the previous
    # index intact rather than a truncated file FAISS can't open.
    tmp = path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)


def add_vectors(file_id: str, vectors: np.ndarray) -> list[int]:
    """Append vectors, returning the position each one landed at.

    Positions are what the `chunks` table stores, mapping a search hit back to
    a byte range in the uploaded file.
    """
    if vectors.shape[0] == 0:
        return []

    with _lock_for(file_id):
        index = _load(file_id)
        start = index.ntotal
        index.add(vectors)
        _save(file_id, index)
        return list(range(start, start + vectors.shape[0]))


def search(file_id: str, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return (vector_position, score) for the closest passages.

    Vectors are normalised, so the inner product FAISS returns is cosine
    similarity in [-1, 1].
    """
    path = storage.index_path(file_id)
    if not path.exists():
        return []

    with _lock_for(file_id):
        # Memory-map rather than loading: for a large index only the pages a
        # search touches are faulted in, and the OS can evict them under
        # pressure. This is what keeps a 420 MB index off the heap.
        index = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
        if index.ntotal == 0:
            return []

        scores, positions = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for position, score in zip(positions[0], scores[0]):
        if position < 0:  # FAISS pads with -1 when fewer than k results exist
            continue
        results.append((int(position), float(score)))
    return results


def compact(file_id: str) -> None:
    """Rebuild a completed file's index in quantized form.

    Run once the upload finishes rather than during it: a scalar quantizer must
    see the data distribution before it can encode, and rebuilding once at the
    end is cheaper than retraining on every append.

    Skipped for small files, where the flat index already costs little.
    """
    if not config.USE_QUANTIZATION:
        return

    path = storage.index_path(file_id)
    if not path.exists():
        return

    with _lock_for(file_id):
        index = _load(file_id)
        if index.ntotal < config.QUANTIZATION_MIN_VECTORS:
            return
        if not isinstance(index, faiss.IndexFlat):
            return  # already compacted

        vectors = index.reconstruct_n(0, index.ntotal)
        quantized = faiss.IndexScalarQuantizer(
            config.EMBEDDING_DIM,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        quantized.train(vectors)
        quantized.add(vectors)
        _save(file_id, quantized)


def vector_count(file_id: str) -> int:
    """How many vectors this file's index holds."""
    path = storage.index_path(file_id)
    if not path.exists():
        return 0
    with _lock_for(file_id):
        return _load(file_id).ntotal


def drop(file_id: str) -> None:
    """Delete the index and forget its lock."""
    storage.index_path(file_id).unlink(missing_ok=True)
    with _locks_guard:
        _locks.pop(file_id, None)
