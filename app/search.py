"""Semantic search over an uploaded file.

The query is embedded with the same model used at index time, so a natural
language phrase lands near passages that *mean* the same thing, whether or not
they share words. "database connectivity problems" scores highly against
"Connection to database failed after 30 seconds" because the sentence
embeddings are close, not because the terms overlap.

Qdrant carries each passage's byte range as point payload, so a search result
maps directly to a location in the uploaded file with no lookup back to
SQLite -- the matching text itself is still read from the file by that byte
range, since chunk rows (and now point payload) store coordinates rather than
a duplicate copy of the corpus.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from . import db, models, storage, vector_store
from .identity import get_user_id
from .pipeline import embeddings

router = APIRouter(tags=["search"])


@router.post(
    "/files/{file_id}/search",
    response_model=models.SearchResponse,
    summary="Search a file in natural language",
    responses={
        404: {"model": models.ErrorResponse, "description": "Unknown file"},
        409: {"model": models.ErrorResponse, "description": "Nothing indexed yet"},
    },
)
def search_file(
    file_id: str,
    payload: models.SearchRequest,
    user_id: str = Depends(get_user_id),
) -> models.SearchResponse:
    """Return the sections whose meaning is closest to the query.

    Searching is allowed while the upload is still in progress -- whatever has
    been indexed so far is queryable.
    """
    conn = db.get_connection()
    file_row = conn.execute(
        "SELECT file_id, owner_id, chunks_indexed FROM files WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    if file_row is None or file_row["owner_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}")

    if file_row["chunks_indexed"] == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "no passages indexed yet; poll /status until searchable is true",
        )

    query_vector = embeddings.embed_query(payload.query)
    hits = vector_store.search(file_id, query_vector, payload.top_k)

    results = []
    for hit in hits:
        text = storage.read_range(file_id, hit["start_byte"], hit["end_byte"])
        if not text:
            # A genuinely empty read only happens if the file's gone; a
            # whitespace-only passage is legitimate content and must not be
            # dropped from results just because it strips to nothing.
            continue
        results.append(
            models.SearchHit(
                text=text,
                start_byte=hit["start_byte"],
                end_byte=hit["end_byte"],
                score=hit["score"],
                sequence=hit["sequence"],
            )
        )

    return models.SearchResponse(
        file_id=file_id,
        query=payload.query,
        total_hits=len(results),
        results=results,
    )
