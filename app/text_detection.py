"""Reject uploads that aren't actually text.

The assignment scopes this service to large *text* files, and nothing else in
the pipeline can recover from binary input: passages decode with
errors="replace" (to tolerate a chunk boundary landing mid-UTF-8-character,
not to tolerate a whole binary file), so a PDF, image, or archive produces
passages of replacement-character noise, gets embedded as meaningless
vectors, and returns garbage from every search -- without ever raising an
error the client would notice before that point.

Checked once, against the first chunk of an upload: by then enough of the
file is available to judge it, and rejecting before any bytes are written or
queued for embedding avoids wasted disk and CPU on content that can never
search correctly.
"""

# A single null byte is close to definitive: legitimate UTF-8 text never
# contains one. Binary formats (executables, images, many archives) almost
# always do somewhere in their first few KB.
NULL_BYTE = b"\x00"

# Above this fraction of bytes that fail to decode as UTF-8, treat the sample
# as binary. Legitimate text can contain a handful of stray bytes at a chunk
# boundary (part of a still-arriving multi-byte character) without being
# binary; a file that's mostly undecodable is not text with a rough edge, it's
# a different format entirely.
MAX_INVALID_UTF8_FRACTION = 0.05

# Only the leading portion of the first chunk needs checking -- binary
# formats reveal themselves immediately (a PDF header, a PNG signature, a
# zip's local file header), so scanning more only costs CPU for no better
# signal.
SAMPLE_BYTES = 8192


def looks_like_binary(sample: bytes) -> bool:
    """True if `sample` (typically the start of an upload) is not text."""
    if not sample:
        return False

    probe = sample[:SAMPLE_BYTES]

    if NULL_BYTE in probe:
        return True

    invalid = 0
    try:
        probe.decode("utf-8")
    except UnicodeDecodeError:
        # Count byte-level failures rather than giving up at the first one,
        # so a single split multi-byte character at the very end of the
        # sample doesn't get the same verdict as a file that's mostly
        # undecodable throughout.
        pos = 0
        remaining = probe
        while remaining:
            try:
                remaining.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                invalid += exc.end - exc.start
                remaining = remaining[exc.end :]

    return (invalid / len(probe)) > MAX_INVALID_UTF8_FRACTION
