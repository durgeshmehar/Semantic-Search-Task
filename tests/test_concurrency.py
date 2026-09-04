"""Concurrent chunk uploads to the same file must not corrupt it.

The scenario this guards against: a client's HTTP request times out waiting
for a response that the server actually did send, so the client retries with
the same offset while the original request may still be in flight server-side.
Both requests reading `bytes_received` before either writes it back would
double-append the same bytes to disk. The fix is running the whole
read-check-append-write sequence inside one SQLite transaction, so the second
request's BEGIN IMMEDIATE blocks until the first commits and then correctly
fails its offset check instead of racing it.
"""

import threading

from tests.conftest import upload_file


def test_concurrent_puts_at_the_same_offset_do_not_duplicate_bytes(client):
    """Exactly one of two racing identical requests should succeed."""
    payload = b"x" * 5000
    response = client.post(
        "/files", json={"filename": "race.log", "total_size": len(payload)}
    )
    file_id = response.json()["file_id"]

    results = []

    def send():
        r = client.put(
            f"/files/{file_id}/chunk", params={"offset": 0}, content=payload
        )
        results.append(r.status_code)

    threads = [threading.Thread(target=send) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one request should have been accepted as the "first" append;
    # the rest must see the advanced offset and be rejected, not silently
    # duplicate the bytes.
    assert results.count(200) == 1
    assert results.count(409) == 4

    from app import storage

    # The critical assertion: the file has exactly one copy of the payload,
    # not two, three, four, or five.
    assert storage.current_size(file_id) == len(payload)


def test_chunk_upload_returns_503_when_no_upload_slot_is_free(client, monkeypatch):
    """The HTTP wiring for the concurrent-upload cap, not just the semaphore.

    Confirms upload_chunk actually surfaces TooManyConcurrentUploads as a 503
    with Retry-After, by holding every slot open before making a real request
    through the app -- rather than only testing the semaphore in isolation
    (see test_upload_limiter.py for that).
    """
    from app import upload_limiter

    monkeypatch.setattr(upload_limiter.config, "MAX_CONCURRENT_UPLOADS", 1)
    monkeypatch.setattr(upload_limiter, "_semaphore", threading.Semaphore(1))

    response = client.post(
        "/files", json={"filename": "full.log", "total_size": 10}
    )
    file_id = response.json()["file_id"]

    with upload_limiter.acquire():  # the one slot is now held
        response = client.put(
            f"/files/{file_id}/chunk", params={"offset": 0}, content=b"x" * 10
        )

    assert response.status_code == 503
    assert "Retry-After" in response.headers
    assert response.json()["detail"]  # names the reason, not a bare 503


def test_concurrent_puts_at_different_offsets_both_eventually_succeed(client):
    """Sequential (non-racing) chunks are unaffected by the transaction change."""
    data = b"a" * 3000 + b"b" * 3000
    response = client.post(
        "/files", json={"filename": "seq.log", "total_size": len(data)}
    )
    file_id = response.json()["file_id"]

    r1 = client.put(f"/files/{file_id}/chunk", params={"offset": 0}, content=data[:3000])
    assert r1.status_code == 200

    r2 = client.put(f"/files/{file_id}/chunk", params={"offset": 3000}, content=data[3000:])
    assert r2.status_code == 200

    from app import storage

    assert storage.current_size(file_id) == len(data)
