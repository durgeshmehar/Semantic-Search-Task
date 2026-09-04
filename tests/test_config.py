"""Adaptive passage sizing.

Passage size is the precision/scale tradeoff: small passages embed precisely
but produce more vectors than can stay resident for a 10 GB file. These tests
pin both ends of that -- typical files keep the precise default, and a 10 GB
file's quantized vectors stay within the resident-memory budget.

This budget applies to Qdrant's quantized vectors (see vector_store.py,
QUANTIZATION_ALWAYS_RAM) rather than an in-process index file; the sizing math
itself -- passages per byte, bytes per vector -- is unchanged by that choice.
"""

from app import config


def index_bytes(total_size: int) -> int:
    """Resident RAM the quantized vectors would need for a file this size."""
    target, _, overlap = config.passage_size_for(total_size)
    stride = target - overlap
    return int(total_size / stride) * config.EMBEDDING_DIM


def test_small_files_keep_the_precise_default():
    """A typical upload shouldn't be coarsened -- precision matters more."""
    for size in (1024, 1024**2, 50 * 1024**2):
        target, _, overlap = config.passage_size_for(size)
        assert target == config.PASSAGE_TARGET_BYTES
        assert overlap == config.PASSAGE_OVERLAP_BYTES


def test_ten_gigabyte_file_fits_the_memory_budget():
    """The assignment's worst case must not blow the index budget."""
    ten_gb = 10 * 1024**3
    assert index_bytes(ten_gb) <= config.INDEX_MEMORY_BUDGET_BYTES * 1.05


def test_passage_size_grows_with_file_size():
    """Coarsening should be gradual, not a cliff."""
    sizes = [100 * 1024**2, 1024**3, 5 * 1024**3, 10 * 1024**3]
    targets = [config.passage_size_for(s)[0] for s in sizes]
    assert targets == sorted(targets)
    assert targets[-1] > targets[0]


def test_target_never_exceeds_the_ceiling():
    """Even absurd sizes stay bounded -- passages must remain readable."""
    for size in (10 * 1024**3, 100 * 1024**3, 1024**4):
        target, maximum, overlap = config.passage_size_for(size)
        assert target <= config.PASSAGE_MAX_TARGET_BYTES
        assert overlap < target
        assert maximum >= target


def test_zero_size_is_safe():
    """A declared size of zero must not divide by zero."""
    target, maximum, overlap = config.passage_size_for(0)
    assert target == config.PASSAGE_TARGET_BYTES
    assert maximum > 0
