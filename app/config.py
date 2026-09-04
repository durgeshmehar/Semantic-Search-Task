"""Runtime configuration, all overridable via environment variables.

Defaults are tuned for the assignment's constraint: a 4 GB machine ingesting
files up to 10 GB. See README for the memory budget these numbers produce.
"""

import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# --- Storage layout -------------------------------------------------------

DATA_DIR = Path(os.environ.get("DATA_DIR", "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "metadata.db"

# --- Upload ---------------------------------------------------------------

# Largest single chunk body we will accept. The client picks its own chunk
# size; this is the ceiling that bounds per-request memory.
MAX_CHUNK_BYTES = _int_env("MAX_CHUNK_BYTES", 16 * 1024 * 1024)

# Ceiling on a single upload, per the assignment.
MAX_FILE_BYTES = _int_env("MAX_FILE_BYTES", 10 * 1024 * 1024 * 1024)

# --- Passage chunking -----------------------------------------------------

# Target passage size in bytes. ~600 B is a handful of log lines or a short
# paragraph.
#
# This is the main precision/cost dial. A passage is embedded as a single
# vector, so its embedding is an average of everything inside it: make
# passages too large and one relevant line gets diluted by its neighbours
# until a query about it no longer matches. Make them too small and the vector
# count -- and so index size and embedding time -- rises proportionally.
PASSAGE_TARGET_BYTES = _int_env("PASSAGE_TARGET_BYTES", 600)

# A passage is emitted once it reaches the target, but we never split a line,
# so this caps how far past the target a single very long line may push us.
PASSAGE_MAX_BYTES = _int_env("PASSAGE_MAX_BYTES", 2048)

# Trailing bytes of each passage repeated at the head of the next, so a match
# spanning a boundary is still retrievable from one side of it.
PASSAGE_OVERLAP_BYTES = _int_env("PASSAGE_OVERLAP_BYTES", 120)

# How much RAM the vector index for one file may occupy. Each int8 vector costs
# EMBEDDING_DIM bytes, so this is what caps the vector count -- and therefore
# how small passages are allowed to be.
INDEX_MEMORY_BUDGET_BYTES = _int_env(
    "INDEX_MEMORY_BUDGET_BYTES", 1024 * 1024 * 1024
)

# Ceiling for the adaptive sizing below.
PASSAGE_MAX_TARGET_BYTES = _int_env("PASSAGE_MAX_TARGET_BYTES", 6000)


def passage_size_for(total_size: int) -> tuple[int, int, int]:
    """Choose (target, max, overlap) passage sizes for a file of this size.

    Passage size is the central precision/scale tradeoff, and the right answer
    depends on the file:

    - Small passages embed precisely. One relevant line is a large share of the
      passage, so it survives being averaged into a single vector.
    - Small passages mean more vectors. A 10 GB file at the 600 B default would
      produce ~22M vectors -- 8 GB even at int8, which does not fit in 4 GB.

    So rather than one size that is either imprecise for small files or
    unaffordable for large ones, the target scales with the declared file size
    to keep each file's index inside INDEX_MEMORY_BUDGET_BYTES. Typical uploads
    keep the precise default; only genuinely huge ones are coarsened, and the
    README documents that recall softens as a result.
    """
    target = PASSAGE_TARGET_BYTES
    overlap = PASSAGE_OVERLAP_BYTES

    if total_size > 0:
        max_vectors = INDEX_MEMORY_BUDGET_BYTES / EMBEDDING_DIM
        # Overlap repeats bytes, so each passage only advances (target-overlap).
        needed_stride = total_size / max_vectors
        if needed_stride > (target - overlap):
            target = min(int(needed_stride * 1.25), PASSAGE_MAX_TARGET_BYTES)
            overlap = max(int(target * 0.2), PASSAGE_OVERLAP_BYTES)

    maximum = max(target * 3, PASSAGE_MAX_BYTES)
    return target, maximum, overlap

# --- Embedding / index ----------------------------------------------------

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = _int_env("EMBEDDING_DIM", 384)

# Passages embedded per forward pass. Larger batches are faster but hold more
# activations in memory; 32 keeps a worker around ~150 MB.
EMBED_BATCH_SIZE = _int_env("EMBED_BATCH_SIZE", 32)

# Background threads consuming the indexing queue. Kept low deliberately: the
# embedding model is CPU-bound, so more threads means more memory without
# proportional throughput.
WORKER_COUNT = _int_env("WORKER_COUNT", 2)

# --- Vector store (Qdrant) -------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Keep the int8-quantized vectors resident for search speed, while raw
# float32 vectors and the HNSW graph stay on disk (see vector_store.py). This
# is the actual RAM lever now: quantized vectors are 384 B each, so the
# resident set for even a 10 GB file's ~2.8M passages is under 1.1 GB -- see
# README section 1 for the sizing table this replaced.
QUANTIZATION_ALWAYS_RAM = _bool_env("QUANTIZATION_ALWAYS_RAM", True)

# --- Job queue ------------------------------------------------------------

# How many pending chunks a worker claims per round trip to SQLite.
CLAIM_BATCH_SIZE = _int_env("CLAIM_BATCH_SIZE", 32)

# Attempts before a chunk is marked permanently failed.
MAX_RETRIES = _int_env("MAX_RETRIES", 3)

# Idle sleep between polls when the queue is empty.
WORKER_POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "0.5"))

# --- Search ---------------------------------------------------------------

DEFAULT_TOP_K = _int_env("DEFAULT_TOP_K", 10)
MAX_TOP_K = _int_env("MAX_TOP_K", 100)


def ensure_dirs() -> None:
    """Create the on-disk layout. Safe to call repeatedly."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
