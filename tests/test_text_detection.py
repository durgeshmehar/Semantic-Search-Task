"""Binary-content detection.

The service only supports text files; without this check, uploading a PDF,
image, or archive silently indexes decode-noise (errors="replace" on invalid
UTF-8) and every search against it returns garbage with no error anywhere.
These tests pin the boundary between "legitimate text with a rough edge" (a
chunk split mid-multibyte-character, a stray control byte) and "not text at
all."
"""

from app import text_detection


def test_plain_ascii_text_is_not_binary():
    sample = b"INFO Starting server\nERROR Connection failed\n" * 50
    assert not text_detection.looks_like_binary(sample)


def test_utf8_text_with_multibyte_characters_is_not_binary():
    sample = ("café 中文 music note: \U0001D11E\n" * 50).encode("utf-8")
    assert not text_detection.looks_like_binary(sample)


def test_empty_sample_is_not_binary():
    assert not text_detection.looks_like_binary(b"")


def test_a_single_null_byte_is_binary():
    """Legitimate UTF-8 text never contains a null byte."""
    sample = b"hello world\n" * 100 + b"\x00" + b"more text\n" * 100
    assert text_detection.looks_like_binary(sample)


def test_a_png_header_is_binary():
    png_signature = b"\x89PNG\r\n\x1a\n"
    sample = png_signature + bytes(range(256)) * 30
    assert text_detection.looks_like_binary(sample)


def test_a_pdf_header_is_binary():
    sample = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + bytes(
        [i % 256 for i in range(4000)]
    )
    assert text_detection.looks_like_binary(sample)


def test_a_zip_local_file_header_is_binary():
    zip_magic = b"PK\x03\x04"
    sample = zip_magic + bytes([i % 256 for i in range(4000)])
    assert text_detection.looks_like_binary(sample)


def test_a_chunk_boundary_mid_multibyte_character_is_not_binary():
    """A tiny amount of trailing invalid UTF-8 (a split character) is normal,
    not a sign of a binary file -- it's exactly what the line buffer expects
    to handle via its own overlap/tail logic, upstream of this check."""
    whole = ("plain log line here\n" * 100 + "café").encode("utf-8")
    # Cut off mid-character: drop the last byte of the multi-byte 'é'.
    truncated = whole[:-1]
    assert not text_detection.looks_like_binary(truncated)


def test_mostly_invalid_utf8_is_binary_even_without_a_null_byte():
    # High bytes with no valid UTF-8 structure, but avoiding 0x00 specifically
    # to isolate the ratio-based path from the null-byte shortcut.
    sample = bytes([b if b != 0 else 1 for b in range(1, 256)]) * 20
    assert text_detection.looks_like_binary(sample)


def test_only_the_configured_sample_size_is_scanned():
    """A file that's garbage after the sample window still passes -- the
    check is a fast first-chunk heuristic, not a whole-file scan."""
    good_prefix = b"clean ascii text\n" * 1000  # far more than SAMPLE_BYTES
    assert len(good_prefix) > text_detection.SAMPLE_BYTES
    garbage_suffix = b"\x00" * 1000
    assert not text_detection.looks_like_binary(good_prefix + garbage_suffix)
