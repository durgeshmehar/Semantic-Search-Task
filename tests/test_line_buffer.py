"""Boundary correctness for the passage chunker.

The property that matters: however the network splits the byte stream, the
passages produced must cover the file completely, in order, without losing a
byte -- and must not cut UTF-8 characters in half.
"""

import random

import pytest

from app.pipeline.line_buffer import LineBuffer


def collect(data: bytes, chunk_sizes: list[int], **kwargs) -> list:
    """Feed `data` in the given chunk sizes and return all passages."""
    buf = LineBuffer(**kwargs)
    passages = []
    offset = 0
    for size in chunk_sizes:
        if offset >= len(data):
            break
        passages.extend(buf.feed(data[offset : offset + size]))
        offset += size
    if offset < len(data):
        passages.extend(buf.feed(data[offset:]))
    passages.extend(buf.flush())
    return passages


def assert_covers(passages: list, data: bytes) -> None:
    """Passages must tile the input in order, overlapping but never gapping."""
    assert passages, "expected at least one passage"
    assert passages[0].start_byte == 0
    assert passages[-1].end_byte == len(data)

    for prev, cur in zip(passages, passages[1:]):
        # Next passage starts inside or at the end of the previous one: the
        # overlap is deliberate, a gap would mean lost text.
        assert cur.start_byte <= prev.end_byte, "gap between passages"
        assert cur.start_byte >= prev.start_byte, "passages out of order"
        assert cur.end_byte > cur.start_byte, "empty passage"


def test_single_feed_covers_input():
    data = b"".join(b"line %d with some content\n" % i for i in range(500))
    passages = collect(data, [len(data)])
    assert_covers(passages, data)


def test_split_mid_line_is_reassembled():
    """A chunk boundary inside a line must not produce a passage boundary."""
    data = b"alpha beta gamma\ndelta epsilon zeta\n" * 200
    # Deliberately awkward sizes, none aligned to a line.
    passages = collect(data, [7, 13, 29, 3, 101, 5])
    assert_covers(passages, data)


def test_split_mid_multibyte_character():
    """UTF-8 characters split across chunks must survive intact."""
    # 'é' is 2 bytes, '中' is 3, '𝄞' is 4 -- cover each width.
    line = "café 中文 𝄞 music\n".encode("utf-8")
    data = line * 300

    # One byte at a time is the worst case: every multi-byte character is split.
    passages = collect(data, [1] * len(data))
    assert_covers(passages, data)

    # Every passage must decode cleanly, which fails if we cut a character.
    for p in passages:
        data[p.start_byte : p.end_byte].decode("utf-8")


def test_long_line_exceeding_max_is_split_safely():
    """A single line longer than max_bytes can't be buffered forever."""
    data = ("x" * 20_000 + "\n").encode("utf-8")
    passages = collect(data, [512] * 50, target_bytes=1000, max_bytes=2000)
    assert_covers(passages, data)
    for p in passages:
        assert p.size <= 2000


def test_no_trailing_newline():
    """The last line often has no newline; it must still be emitted."""
    data = b"first line\nsecond line\nthird without newline"
    passages = collect(data, [len(data)])
    assert_covers(passages, data)


def test_empty_input_yields_nothing():
    buf = LineBuffer()
    assert buf.feed(b"") == []
    assert buf.flush() == []


def test_resume_from_persisted_tail():
    """A resumed upload continues mid-line using the persisted tail."""
    data = b"alpha beta\ngamma delta\nepsilon zeta\n" * 100
    split_at = 137  # lands mid-line

    first = LineBuffer(target_bytes=200)
    passages = first.feed(data[:split_at])
    tail = first.pending_tail

    # Simulate a restart: rebuild from the persisted tail and byte offset.
    second = LineBuffer(start_offset=split_at, pending_tail=tail, target_bytes=200)
    passages += second.feed(data[split_at:])
    passages += second.flush()

    assert_covers(passages, data)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_chunk_sizes_preserve_coverage(seed):
    """Fuzz: arbitrary chunk boundaries must never lose or reorder bytes."""
    rng = random.Random(seed)
    lines = []
    for i in range(400):
        width = rng.randint(0, 120)
        lines.append(("%d " % i + "word " * (width // 5)).strip().encode() + b"\n")
    data = b"".join(lines)

    sizes = [rng.randint(1, 3000) for _ in range(200)]
    passages = collect(data, sizes)
    assert_covers(passages, data)


def test_passages_stay_near_target_size():
    """Passages must not stretch to max_bytes.

    Regression: an earlier version took the *last* line break in the window
    rather than the first past the target, so every passage grew to max_bytes.
    Oversized passages embed as an average of their contents, which buries a
    single relevant line inside a block of unrelated text and makes it
    unfindable -- semantic search silently degrades.
    """
    data = b"a short log line of roughly forty bytes\n" * 500
    passages = collect(data, [8192], target_bytes=1500, max_bytes=4096)

    assert_covers(passages, data)
    assert len(passages) > 5

    # Ignore the final passage, which is just whatever remains.
    sizes = [p.size for p in passages[:-1]]
    average = sum(sizes) / len(sizes)

    assert average < 2200, f"passages averaging {average:.0f}B, expected near 1500B"
    assert max(sizes) <= 4096


def test_flush_does_not_emit_shrinking_fragments():
    """The final flush must not re-emit its own overlap tail.

    Regression: flush() looped until the buffer was empty, so after emitting
    the last real passage it kept re-emitting the retained overlap as
    ever-shorter fragments (a 57-byte passage, then 28, then 14). Very short
    passages then dominate search results, because a fragment containing a
    couple of the query's words has high cosine similarity to it.
    """
    data = b"a log line with a reasonable amount of text in it\n" * 60
    passages = collect(data, [4096], target_bytes=600, max_bytes=2048, overlap_bytes=120)

    assert_covers(passages, data)

    # No passage may be a sliver. The last one is whatever remains, but even it
    # should be a meaningful chunk rather than a fragment of the overlap.
    sizes = [p.size for p in passages]
    assert min(sizes) > 120, f"found fragment passages: {sorted(sizes)[:5]}"

    # Every passage must have a distinct end -- repeated tails share one.
    ends = [p.end_byte for p in passages]
    assert len(ends) == len(set(ends)), "duplicate passage endings"


def test_overlap_is_bounded_and_backwards():
    """Overlap should exist but never rewind past the previous start."""
    data = b"some reasonably long line of text here\n" * 300
    passages = collect(data, [4096], target_bytes=500, overlap_bytes=100)
    assert len(passages) > 1
    for prev, cur in zip(passages, passages[1:]):
        overlap = prev.end_byte - cur.start_byte
        assert 0 <= overlap <= 100
