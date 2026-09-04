"""End-to-end semantic search.

The headline test is the assignment's own example: a file containing
"Connection to database failed after 30 seconds." must be findable by searching
"database connectivity problems" -- overlapping in meaning, not in phrasing.

These tests load the real embedding model, so they are slower than the rest of
the suite.
"""

import pytest

from app.pipeline import worker
from tests.conftest import upload_file


def index_everything() -> None:
    """Drain the queue on this thread so indexing is deterministic."""
    worker.drain_once()


def test_semantic_search_finds_paraphrased_section(client, sample_text):
    """The assignment's example, verbatim."""
    file_id = upload_file(client, sample_text)
    index_everything()

    response = client.post(
        f"/files/{file_id}/search",
        json={"query": "database connectivity problems", "top_k": 5},
    )
    assert response.status_code == 200

    results = response.json()["results"]
    assert results, "expected at least one hit"

    # The target line shares no words with the query except "database" -- the
    # match has to come from meaning.
    combined = " ".join(hit["text"] for hit in results)
    assert "Connection to database failed" in combined


def test_results_are_ranked_by_similarity(client, sample_text):
    file_id = upload_file(client, sample_text)
    index_everything()

    response = client.post(
        f"/files/{file_id}/search",
        json={"query": "database connection failure", "top_k": 10},
    )
    scores = [hit["score"] for hit in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_hits_carry_byte_offsets_that_locate_the_text(client, sample_text):
    """Byte ranges must point at the returned text in the original file."""
    file_id = upload_file(client, sample_text)
    index_everything()

    response = client.post(
        f"/files/{file_id}/search",
        json={"query": "disk space running low", "top_k": 3},
    )
    for hit in response.json()["results"]:
        assert hit["end_byte"] > hit["start_byte"]
        assert hit["end_byte"] <= len(sample_text)
        # The text the API returned is exactly what those offsets contain.
        expected = sample_text[hit["start_byte"] : hit["end_byte"]].decode(
            "utf-8", errors="replace"
        )
        assert hit["text"] == expected


def test_unrelated_query_scores_lower_than_related(client, sample_text):
    """Sanity check that scores track meaning rather than being noise."""
    file_id = upload_file(client, sample_text)
    index_everything()

    related = client.post(
        f"/files/{file_id}/search",
        json={"query": "database connectivity problems", "top_k": 1},
    ).json()["results"][0]["score"]

    unrelated = client.post(
        f"/files/{file_id}/search",
        json={"query": "baking a chocolate cake recipe", "top_k": 1},
    ).json()["results"][0]["score"]

    assert related > unrelated


def test_search_before_indexing_is_refused(client, sample_text):
    """Nothing indexed yet is a distinct state from 'no matches'."""
    file_id = upload_file(client, sample_text)
    # Deliberately skip draining the queue.

    response = client.post(
        f"/files/{file_id}/search", json={"query": "anything"}
    )
    assert response.status_code == 409
    assert "searchable" in response.json()["detail"]


def test_search_works_during_an_in_progress_upload(client, sample_text):
    """Passages indexed so far are queryable before the upload finishes."""
    response = client.post(
        "/files", json={"filename": "live.log", "total_size": len(sample_text)}
    )
    file_id = response.json()["file_id"]

    # Send only the first part of the file.
    half = len(sample_text) // 2
    client.put(
        f"/files/{file_id}/chunk", params={"offset": 0}, content=sample_text[:half]
    )
    index_everything()

    status = client.get(f"/files/{file_id}/status").json()
    assert status["upload_status"] == "uploading"
    assert status["searchable"] is True

    response = client.post(
        f"/files/{file_id}/search",
        json={"query": "application server startup", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["results"]


def test_search_unknown_file_is_404(client):
    response = client.post("/files/nope/search", json={"query": "x"})
    assert response.status_code == 404


def test_top_k_is_respected(client, sample_text):
    file_id = upload_file(client, sample_text)
    index_everything()

    response = client.post(
        f"/files/{file_id}/search", json={"query": "logging", "top_k": 2}
    )
    assert len(response.json()["results"]) <= 2


def test_empty_query_is_rejected(client, sample_text):
    file_id = upload_file(client, sample_text)
    response = client.post(f"/files/{file_id}/search", json={"query": ""})
    assert response.status_code == 422  # pydantic min_length


def test_unicode_content_is_searchable(client):
    """Multi-byte text must survive chunking and come back intact."""
    lines = []
    for i in range(60):
        lines.append(f"[{i}] El servidor de base de datos no responde")
        lines.append(f"[{i}] 数据库连接超时，请稍后重试")
        lines.append(f"[{i}] Le café est prêt pour les développeurs")
    data = ("\n".join(lines) + "\n").encode("utf-8")

    file_id = upload_file(client, data, chunk_size=137)  # awkward, splits chars
    index_everything()

    response = client.post(
        f"/files/{file_id}/search",
        json={"query": "database server not responding", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["results"]
