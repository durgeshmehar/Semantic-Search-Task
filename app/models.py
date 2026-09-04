"""Request and response schemas.

These double as the API documentation: FastAPI renders them into the OpenAPI
spec served at /docs.
"""

from pydantic import BaseModel, Field

from . import config


class CreateUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=512, examples=["server.log"])
    total_size: int = Field(
        ...,
        ge=0,
        le=config.MAX_FILE_BYTES,
        description="Full size of the file in bytes, known up front so the "
        "server can validate completion and report progress.",
        examples=[1048576],
    )


class CreateUploadResponse(BaseModel):
    file_id: str
    filename: str
    total_size: int
    chunk_size: int = Field(
        ..., description="Largest chunk body the server will accept, in bytes."
    )
    upload_status: str


class ChunkUploadResponse(BaseModel):
    file_id: str
    bytes_received: int = Field(
        ..., description="Send the next chunk from this offset."
    )
    total_size: int
    upload_status: str
    chunks_enqueued: int = Field(
        ..., description="Passages queued for indexing so far."
    )


class FileStatus(BaseModel):
    """Upload and processing progress, reported independently.

    They are separate states on purpose: bytes can be fully received while
    indexing is still catching up, and the client may want to act on either.
    """

    file_id: str
    filename: str
    total_size: int
    bytes_received: int
    upload_status: str = Field(
        ..., description="pending | uploading | interrupted | completed | failed"
    )
    upload_progress: float = Field(..., ge=0.0, le=1.0)
    processing_status: str = Field(
        ..., description="pending | processing | completed | failed"
    )
    processing_progress: float = Field(..., ge=0.0, le=1.0)
    chunks_total: int
    chunks_indexed: int
    chunks_failed: int
    searchable: bool = Field(
        ..., description="True once at least one passage is indexed."
    )
    error_message: str | None = None
    created_at: float
    updated_at: float


class FileListResponse(BaseModel):
    files: list[FileStatus]


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language query. Matching is by meaning, not "
        "keyword overlap.",
        examples=["database connectivity problems"],
    )
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1,
        le=config.MAX_TOP_K,
        description="How many sections to return.",
    )


class SearchHit(BaseModel):
    text: str = Field(..., description="The matching section, read from the file.")
    start_byte: int
    end_byte: int
    score: float = Field(
        ..., description="Cosine similarity in [-1, 1]; higher is closer."
    )
    sequence: int = Field(..., description="Position of this passage in the file.")


class SearchResponse(BaseModel):
    file_id: str
    query: str
    total_hits: int
    results: list[SearchHit]


class ErrorResponse(BaseModel):
    detail: str
