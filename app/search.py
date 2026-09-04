"""Semantic search over an uploaded file.

The query is embedded with the same model used at index time, so a natural
language phrase lands near passages that *mean* the same thing, whether or not
they share words. "database connectivity problems" scores highly against
"Connection to database failed after 30 seconds" because the sentence
embeddings are close, not because the terms overlap.

Matching text is read back from the uploaded file by byte offset, since the
chunk rows store coordinates rather than a duplicate copy of the corpus.
"""

from fastapi import APIRouter, HTTPException, status

from . import db, faiss_index, models, storage
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
def search_file(file_id: str, payload: models.SearchRequest) -> models.SearchResponse:
    """Return the sections whose meaning is closest to the query.

    Searching is allowed while the upload is still in progress -- whatever has
    been indexed so far is queryable.
    """
    conn = db.get_connection()
    file_row = conn.execute(
        "SELECT file_id, chunks_indexed FROM files WHERE file_id = ?", (file_id,)
    ).fetchone()
    if file_row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown file_id: {file_id}")

    if file_row["chunks_indexed"] == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "no passages indexed yet; poll /status until searchable is true",
        )

    query_vector = embeddings.embed_query(payload.query)

    # Over-fetch: some vector positions may belong to rows deleted or failed
    # since indexing, and we still want a full page of results.
    raw_hits = faiss_index.search(file_id, query_vector, payload.top_k * 2)
    if not raw_hits:
        return models.SearchResponse(
            file_id=file_id, query=payload.query, total_hits=0, results=[]
        )

    positions = [position for position, _ in raw_hits]
    scores = {position: score for position, score in raw_hits}

    placeholders = ",".join("?" * len(positions))
    rows = conn.execute(
        f"""
        SELECT sequence, start_byte, end_byte, vector_position
          FROM chunks
         WHERE file_id = ?
           AND status = 'indexed'
           AND vector_position IN ({placeholders})
        """,
        [file_id, *positions],
    ).fetchall()

    results = []
    for row in rows:
        text = storage.read_range(file_id, row["start_byte"], row["end_byte"])
        if not text.strip():
            continue
        results.append(
            models.SearchHit(
                text=text,
                start_byte=row["start_byte"],
                end_byte=row["end_byte"],
                score=scores.get(row["vector_position"], 0.0),
                sequence=row["sequence"],
            )
        )

    # SQL returned rows in table order; restore similarity order.
    results.sort(key=lambda hit: hit.score, reverse=True)
    results = results[: payload.top_k]

    return models.SearchResponse(
        file_id=file_id,
        query=payload.query,
        total_hits=len(results),
        results=results,
    )
