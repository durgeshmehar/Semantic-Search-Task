"""Turn a stream of arbitrary byte chunks into passage byte-ranges.

The problem this solves: an upload chunk boundary falls wherever the network
put it -- mid-word, mid-line, even mid-UTF-8-character. Passages, though, need
to end on line boundaries, or search results come back with sentences sliced in
half.

So this holds back the trailing incomplete line from each chunk and prepends it
to the next one. Only complete lines are grouped into passages.

Passages are emitted as (start_byte, end_byte) into the *uploaded file* -- the
text itself is never returned or stored, since the file on disk already has it.
"""

from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class PassageRange:
    """A passage's byte span in the uploaded file, end-exclusive."""

    start_byte: int
    end_byte: int

    @property
    def size(self) -> int:
        return self.end_byte - self.start_byte


class LineBuffer:
    """Accumulates bytes, emitting passage ranges aligned to line boundaries.

    Stateful and not thread-safe: one instance per in-flight upload, used only
    from that upload's request path.

    The state that must survive a crash (`pending_tail`, `absolute_offset`) is
    persisted to the files table after each chunk, so a resumed upload picks up
    mid-line without losing or duplicating text.
    """

    def __init__(
        self,
        start_offset: int = 0,
        pending_tail: bytes = b"",
        target_bytes: int | None = None,
        max_bytes: int | None = None,
        overlap_bytes: int | None = None,
    ) -> None:
        # Byte offset in the file where `pending_tail` begins.
        self.absolute_offset = start_offset - len(pending_tail)
        self._buffer = bytearray(pending_tail)
        self.target_bytes = target_bytes or config.PASSAGE_TARGET_BYTES
        self.max_bytes = max_bytes or config.PASSAGE_MAX_BYTES
        self.overlap_bytes = overlap_bytes or config.PASSAGE_OVERLAP_BYTES

    @property
    def pending_tail(self) -> bytes:
        """Bytes held back awaiting more input. Persist this across restarts."""
        return bytes(self._buffer)

    def feed(self, data: bytes) -> list[PassageRange]:
        """Add a chunk of uploaded bytes, returning any complete passages."""
        self._buffer.extend(data)
        return self._drain(final=False)

    def flush(self) -> list[PassageRange]:
        """Emit whatever remains at end of upload.

        The final line usually has no trailing newline, so it would otherwise
        sit in the buffer forever.
        """
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[PassageRange]:
        passages: list[PassageRange] = []

        while True:
            cut = self._find_cut(final)
            if cut is None:
                break

            start = self.absolute_offset
            end = start + cut
            passages.append(PassageRange(start_byte=start, end_byte=end))

            # Carry a little of the tail into the next passage so a match that
            # straddles the boundary is still findable from one side.
            overlap = self._overlap_size(cut)
            del self._buffer[: cut - overlap]
            self.absolute_offset = end - overlap

            if final and len(self._buffer) <= overlap:
                # Everything left is the overlap tail, already covered by the
                # passage just emitted. Without this the final flush would
                # re-emit it as a succession of ever-shorter fragments, and a
                # very short passage scores misleadingly high against any
                # query whose words it happens to contain.
                self._buffer.clear()
                self.absolute_offset = end
                break

        return passages

    def _find_cut(self, final: bool) -> int | None:
        """Where to end the next passage, as an offset into the buffer.

        Returns None when the buffer can't yield a passage yet.
        """
        size = len(self._buffer)
        if size == 0:
            return None

        if size >= self.target_bytes:
            # End at the FIRST line break at or after the target, so passages
            # stay close to the target size. Taking the last break in the
            # window instead would stretch every passage to max_bytes, and an
            # oversized passage embeds as an average of everything in it --
            # one relevant line inside 4 KB of unrelated text gets diluted
            # until it no longer matches a query about it.
            window = self._buffer[: min(size, self.max_bytes)]
            newline = window.find(b"\n", self.target_bytes - 1)
            if newline != -1:
                return newline + 1

            # No break after the target: fall back to the last one before it,
            # provided the passage is still substantial.
            newline = window.rfind(b"\n")
            if newline != -1 and newline + 1 >= self.target_bytes // 2:
                return newline + 1

            if size >= self.max_bytes:
                # A single line longer than max_bytes (minified JSON, a giant
                # log line). Split on a safe character boundary rather than
                # buffering without limit.
                return self._safe_split_point(self.max_bytes)

            # Not enough yet, and no boundary to cut on -- unless this is the
            # end of the file, in which case take what we have.
            return size if final else None

        return size if final else None

    def _safe_split_point(self, limit: int) -> int:
        """Back off from `limit` to avoid splitting a UTF-8 character.

        UTF-8 continuation bytes match 0b10xxxxxx; a character starts at the
        first byte that doesn't. Worst case a character is 4 bytes, so this
        steps back at most 3.
        """
        size = len(self._buffer)
        point = min(limit, size)
        for _ in range(4):
            # Splitting at the very end can't bisect a character.
            if point <= 0 or point >= size:
                return point
            if (self._buffer[point] & 0xC0) != 0x80:
                return point
            point -= 1
        return min(limit, size)

    def _overlap_size(self, cut: int) -> int:
        """Bytes to carry forward, clamped and aligned to a character start."""
        overlap = min(self.overlap_bytes, cut // 2)
        if overlap <= 0:
            return 0

        # Align so the retained tail begins on a character boundary.
        size = len(self._buffer)
        point = cut - overlap
        for _ in range(4):
            if point <= 0 or point >= size:
                return cut - point
            if (self._buffer[point] & 0xC0) != 0x80:
                return cut - point
            point += 1
        return overlap
