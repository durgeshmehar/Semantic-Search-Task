"""Shared fixtures.

Each test gets its own data directory, so uploads and the database never leak
between tests. The config module is re-pointed before anything imports it,
since paths are read at module load.

Vector storage is a real Qdrant instance (QDRANT_URL, default localhost:6333 --
see docker-compose.yml's `tests` service for the containerized run). Tests
don't need to isolate collections from each other: every file gets a fresh
uuid4 file_id, so collection names (file_<file_id>) never collide across
tests or runs. Created collections are dropped at teardown for hygiene, not
correctness.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def isolated_env(monkeypatch):
    """Point the app at a throwaway data directory for one test."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        from app import config, db, vector_store

        monkeypatch.setattr(config, "DATA_DIR", root)
        monkeypatch.setattr(config, "UPLOAD_DIR", root / "uploads")
        monkeypatch.setattr(config, "DB_PATH", root / "metadata.db")

        # Small passages so tests produce several without large fixtures.
        monkeypatch.setattr(config, "PASSAGE_TARGET_BYTES", 200)
        monkeypatch.setattr(config, "PASSAGE_MAX_BYTES", 600)
        monkeypatch.setattr(config, "PASSAGE_OVERLAP_BYTES", 40)

        db.close_connection()
        config.ensure_dirs()
        db.init_db()

        yield root

        db.close_connection()
        _drop_test_collections()


def _drop_test_collections() -> None:
    """Best-effort cleanup of collections this test session created.

    Not required for correctness -- every file_id is a fresh uuid4, so
    collections never collide across tests -- but leaving hundreds of
    empty-ish Qdrant collections around after a full test run is untidy.
    """
    from app import vector_store

    try:
        client = vector_store.get_client()
        for name in client.get_collections().collections:
            if name.name.startswith("file_"):
                client.delete_collection(name.name)
    except Exception:
        pass  # Qdrant not reachable or already clean; not a test failure


@pytest.fixture
def client(isolated_env, monkeypatch):
    """TestClient with background workers disabled.

    Tests drain the queue explicitly via worker.drain_once() so indexing is
    deterministic rather than racing a pool.
    """
    from fastapi.testclient import TestClient

    from app import main
    from app.pipeline import worker

    monkeypatch.setattr(worker, "start_workers", lambda: None)
    monkeypatch.setattr(worker, "stop_workers", lambda: None)

    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def sample_text() -> bytes:
    """Log-like content including the assignment's own example line."""
    lines = [
        "INFO  Starting application server on port 8080",
        "INFO  Loading configuration from /etc/app/config.yaml",
        "WARN  Cache warmup skipped, no snapshot available",
        "ERROR Connection to database failed after 30 seconds.",
        "INFO  Retrying with exponential backoff",
        "INFO  User alice logged in from 10.0.0.4",
        "DEBUG Rendering dashboard template for tenant acme",
        "INFO  Scheduled nightly report generation at 02:00 UTC",
        "WARN  Disk usage on /var has reached 85 percent",
        "INFO  Payment webhook accepted for order 41182",
        "DEBUG Flushing metrics buffer, 512 samples",
        "INFO  Background job queue drained successfully",
    ]
    # Repeat so the file spans many passages, keeping the target line distinct.
    body = []
    for i in range(40):
        for line in lines:
            body.append(f"[{i:04d}] {line}")
    return ("\n".join(body) + "\n").encode("utf-8")


def upload_file(client, data: bytes, filename: str = "test.log", chunk_size: int = 4096) -> str:
    """Upload bytes in chunks and return the file_id."""
    response = client.post(
        "/files", json={"filename": filename, "total_size": len(data)}
    )
    assert response.status_code == 201, response.text
    file_id = response.json()["file_id"]

    offset = 0
    while offset < len(data):
        piece = data[offset : offset + chunk_size]
        response = client.put(
            f"/files/{file_id}/chunk",
            params={"offset": offset},
            content=piece,
        )
        assert response.status_code == 200, response.text
        offset += len(piece)

    return file_id
