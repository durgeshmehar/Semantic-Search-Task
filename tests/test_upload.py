"""Chunked upload, interruption, and resume.

The property under test: however the upload is interrupted and resumed, the
bytes on disk must end up identical to what the client sent.
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


def test_upload_beyond_declared_size_is_rejected(client):
    response = client.post("/files", json={"filename": "s.log", "total_size": 10})
    file_id = response.json()["file_id"]

    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=b"x" * 50
    )
    assert response.status_code == 400


def test_completed_upload_rejects_more_chunks(client):
    data = b"hello world\n"
    file_id = upload_file(client, data)

    response = client.put(
        f"/files/{file_id}/chunk", params={"offset": len(data)}, content=b"more"
    )
    assert response.status_code == 409


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
