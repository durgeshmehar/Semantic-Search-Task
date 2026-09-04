"""Chunked upload, interruption, resume, completion, and ownership.

The property under test: however the upload is interrupted and resumed, the
bytes on disk must end up identical to what the client sent, and the caller
who created a file is the only one who can act on it.
"""

from app import storage
from tests.conftest import upload_file


def test_create_upload_returns_file_id(client):
    response = client.post("/files", json={"filename": "a.log", "total_size": 100})
    assert response.status_code == 201

    body = response.json()
    assert body["file_id"]
    assert body["upload_status"] == "pending"
    assert body["chunk_size"] > 0


def test_full_upload_reassembles_bytes_exactly(client, sample_text):
    file_id = upload_file(client, sample_text, chunk_size=1000)

    status = client.get(f"/files/{file_id}/status").json()
    assert status["upload_status"] == "completed"
    assert status["bytes_received"] == len(sample_text)
    assert status["upload_progress"] == 1.0

    written = storage.final_path(file_id).read_bytes()
    assert written == sample_text


def test_interrupted_upload_resumes_from_reported_offset(client, sample_text):
    """The status endpoint is the resume mechanism."""
    response = client.post(
        "/files", json={"filename": "big.log", "total_size": len(sample_text)}
    )
    file_id = response.json()["file_id"]

    # Send part of the file, then "lose the connection".
    chunk_size = 1000
    sent = 0
    for _ in range(3):
        piece = sample_text[sent : sent + chunk_size]
        client.put(f"/files/{file_id}/chunk", params={"offset": sent}, content=piece)
        sent += len(piece)

    status = client.get(f"/files/{file_id}/status").json()
    assert status["upload_status"] == "uploading"
    resume_from = status["bytes_received"]
    assert resume_from == sent

    # Resume from exactly where the server says it stopped.
    offset = resume_from
    while offset < len(sample_text):
        piece = sample_text[offset : offset + chunk_size]
        response = client.put(
            f"/files/{file_id}/chunk", params={"offset": offset}, content=piece
        )
        assert response.status_code == 200
        offset += len(piece)

    client.post(f"/files/{file_id}/complete")
    assert storage.final_path(file_id).read_bytes() == sample_text


def test_wrong_offset_is_rejected_with_resume_hint(client, sample_text):
    """A mismatched offset must not corrupt the file."""
    response = client.post(
        "/files", json={"filename": "x.log", "total_size": len(sample_text)}
    )
    file_id = response.json()["file_id"]

    client.put(f"/files/{file_id}/chunk", params={"offset": 0}, content=sample_text[:500])

    # Skipping ahead would leave a hole in the file.
    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 9999}, content=b"junk"
    )
    assert response.status_code == 409
    assert "500" in response.json()["detail"]  # tells the client where to resume

    # Re-sending an already-received chunk is refused rather than duplicated.
    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=sample_text[:500]
    )
    assert response.status_code == 409

    assert storage.current_size(file_id) == 500


def test_upload_beyond_max_file_bytes_is_rejected(client, monkeypatch):
    """total_size is a hint, but there is still a hard ceiling on real bytes."""
    from app import config

    monkeypatch.setattr(config, "MAX_FILE_BYTES", 40)

    # Declaring a small total_size doesn't matter -- MAX_FILE_BYTES is what
    # actually caps how many bytes the server will accept.
    response = client.post("/files", json={"filename": "s.log", "total_size": 10})
    file_id = response.json()["file_id"]

    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=b"x" * 50
    )
    assert response.status_code == 400


def test_binary_first_chunk_is_rejected(client):
    """A non-text upload is refused before anything is written or queued."""
    png_signature = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 30
    response = client.post(
        "/files", json={"filename": "photo.png", "total_size": len(png_signature)}
    )
    file_id = response.json()["file_id"]

    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=png_signature
    )
    assert response.status_code == 415

    from app import storage

    # Nothing should have been written for a rejected first chunk.
    assert storage.current_size(file_id) == 0


def test_binary_check_only_applies_to_the_first_chunk(client):
    """Once an upload is underway, later chunks aren't re-scanned.

    A control byte appearing deep in an otherwise-legitimate text file (rare,
    but not impossible in real logs) shouldn't retroactively fail an upload
    that already passed its initial check.
    """
    first = b"clean text content\n" * 50
    second = b"\x00\x01\x02more bytes after a stray control byte\n"
    response = client.post(
        "/files", json={"filename": "log.txt", "total_size": len(first) + len(second)}
    )
    file_id = response.json()["file_id"]

    r1 = client.put(f"/files/{file_id}/chunk", params={"offset": 0}, content=first)
    assert r1.status_code == 200

    r2 = client.put(
        f"/files/{file_id}/chunk", params={"offset": len(first)}, content=second
    )
    assert r2.status_code == 200  # not re-checked, so not rejected


def test_upload_may_exceed_its_own_declared_total_size(client):
    """total_size is a sizing hint, not a hard cap -- real bytes may exceed it."""
    response = client.post("/files", json={"filename": "s.log", "total_size": 10})
    file_id = response.json()["file_id"]

    # Declared 10 bytes, actually sending 20 -- must be accepted.
    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=b"x" * 20
    )
    assert response.status_code == 200
    assert response.json()["bytes_received"] == 20


def test_completed_upload_rejects_more_chunks(client):
    data = b"hello world\n"
    file_id = upload_file(client, data)

    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": len(data)}, content=b"more"
    )
    assert response.status_code == 409


def test_complete_requires_at_least_one_chunk(client):
    response = client.post("/files", json={"filename": "empty.log", "total_size": 10})
    file_id = response.json()["file_id"]

    response = client.post(f"/files/{file_id}/complete")
    assert response.status_code == 409


def test_complete_is_idempotent(client):
    """Calling complete twice (e.g. after a retried request) must not error."""
    data = b"hello world\n"
    file_id = upload_file(client, data)

    response = client.post(f"/files/{file_id}/complete")
    assert response.status_code == 200
    assert response.json()["upload_status"] == "completed"


def test_complete_flushes_the_final_unterminated_line(client):
    """A file with no trailing newline must still index its last line.

    Without an explicit flush on complete, a line with no trailing newline
    sits in the chunker's buffer forever, so its bytes are on disk but never
    become a searchable passage -- a silent gap, not a crash. The property
    that matters: the enqueued passage(s) must cover the whole file, not stop
    short at the last newline.
    """
    from app import db

    data = b"first line\nsecond line with no trailing newline"
    file_id = upload_file(client, data)

    conn = db.get_connection()
    max_end = conn.execute(
        "SELECT MAX(end_byte) AS m FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchone()["m"]
    assert max_end == len(data)
    assert storage.final_path(file_id).read_bytes() == data


def test_declared_total_size_is_corrected_to_actual_bytes_on_complete(client):
    """total_size reflects reality after completion, not the original guess."""
    response = client.post(
        "/files", json={"filename": "s.log", "total_size": 999999}
    )
    file_id = response.json()["file_id"]
    data = b"only twelve b"
    client.put(f"/files/{file_id}/chunk", params={"offset": 0}, content=data)
    client.post(f"/files/{file_id}/complete")

    status = client.get(f"/files/{file_id}/status").json()
    assert status["total_size"] == len(data)
    assert status["upload_progress"] == 1.0


def test_passages_are_enqueued_during_upload(client, sample_text):
    """Indexing work is queued as chunks arrive, not after the upload ends."""
    response = client.post(
        "/files", json={"filename": "stream.log", "total_size": len(sample_text)}
    )
    file_id = response.json()["file_id"]

    # After a single chunk -- with the upload far from finished -- passages
    # should already be queued.
    client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=sample_text[:4000]
    )

    status = client.get(f"/files/{file_id}/status").json()
    assert status["upload_status"] == "uploading"
    assert status["chunks_total"] > 0


def test_status_404_for_unknown_file(client):
    assert client.get("/files/does-not-exist/status").status_code == 404


def test_list_and_delete(client):
    file_id = upload_file(client, b"some content here\n")

    listing = client.get("/files").json()
    assert any(f["file_id"] == file_id for f in listing["files"])

    assert client.delete(f"/files/{file_id}").status_code == 204
    assert client.get(f"/files/{file_id}/status").status_code == 404
    assert storage.readable_path(file_id) is None


def test_partial_file_promoted_to_final_name(client):
    data = b"line one\nline two\n"
    file_id = upload_file(client, data)

    # The .partial name only exists mid-upload.
    assert not storage.partial_path(file_id).exists()
    assert storage.final_path(file_id).exists()


# --- Ownership -------------------------------------------------------------


def test_files_are_scoped_to_their_owner(client):
    """One user cannot see, resume, search, or delete another user's file."""
    file_id = upload_file(client, b"owner-only content\n")

    other = {"X-User-Id": "someone-else"}
    assert client.get(f"/files/{file_id}/status", headers=other).status_code == 404
    assert (
        client.put(
            f"/files/{file_id}/chunk", params={"offset": 0}, content=b"x", headers=other
        ).status_code
        == 404
    )
    assert (
        client.post(f"/files/{file_id}/search", json={"query": "x"}, headers=other).status_code
        == 404
    )
    assert client.delete(f"/files/{file_id}", headers=other).status_code == 404

    # The owner can still do all of this.
    assert client.get(f"/files/{file_id}/status").status_code == 200


def test_listing_only_shows_the_caller_s_own_files(client):
    mine = upload_file(client, b"mine\n")
    other_headers = {"X-User-Id": "someone-else"}
    theirs_resp = client.post(
        "/files", json={"filename": "theirs.log", "total_size": 5}, headers=other_headers
    )
    theirs = theirs_resp.json()["file_id"]

    my_listing = {f["file_id"] for f in client.get("/files").json()["files"]}
    their_listing = {
        f["file_id"] for f in client.get("/files", headers=other_headers).json()["files"]
    }

    assert mine in my_listing and theirs not in my_listing
    assert theirs in their_listing and mine not in their_listing


def test_missing_user_id_header_falls_back_to_anonymous(client):
    """Omitting X-User-Id doesn't error -- it's a shared default identity."""
    response = client.post(
        "/files",
        json={"filename": "anon.log", "total_size": 10},
        headers={"X-User-Id": ""},
    )
    assert response.status_code == 201
