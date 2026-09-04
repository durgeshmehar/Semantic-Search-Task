"""Vector storage and search via Qdrant, one collection per file.

Replaces an earlier hand-rolled FAISS index. The reason for the switch: real
disk-backed, quantized, approximate search needs an index that can be built
incrementally, because this pipeline embeds passages continuously as an
upload streams in rather than all at once from a complete file.

FAISS's approximate index (IndexIVFPQ) can't do that -- both its clustering
(IVF) and its compression codebook (PQ) are learned by *training* on a
representative sample before anything can be inserted, which conflicts with
indexing-while-uploading. Qdrant's default index, HNSW, is a graph built one
insertion at a time with no training phase, so it fits this pipeline as-is.
Point-in-time compression (scalar quantization) and on-disk storage are then
just collection configuration, not code we have to write and validate
ourselves -- which is what the earlier IO_FLAG_MMAP call attempted and did
not actually achieve (see git history / README for that correction).

Point IDs are deterministic (uuid5 of file_id + sequence), so re-embedding a
chunk after a retry or a worker restart is a safe upsert rather than a
duplicate -- the job queue no longer needs to be the only thing preventing
double-indexing.

Byte ranges are stored as point payload rather than only in SQLite, so a
search is a single Qdrant call: no join back to the chunks table is needed to
turn a hit into a byte range.
"""

import logging
import uuid
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from . import config

logger = logging.getLogger(__name__)

# Point IDs are deterministic within this namespace, so the same
# (file_id, sequence) always maps to the same point regardless of which
# worker or retry produced it.
_POINT_NAMESPACE = uuid.UUID("5f2f5a4e-8b8b-4a8e-9f0a-1a2b3c4d5e6f")


def _point_id(file_id: str, sequence: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{file_id}:{sequence}"))


def _collection_name(file_id: str) -> str:
    # One collection per file: searches are always scoped to a single file,
    # and per-file collections keep uploads independent -- deleting a file is
    # dropping one collection, not a filtered delete across a shared one.
    return f"file_{file_id}"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """A single shared client for the process's lifetime."""
    return QdrantClient(url=config.QDRANT_URL, timeout=30.0)


def _ensure_collection(client: QdrantClient, file_id: str) -> str:
    """Create this file's collection on first use.

    Scalar quantization and on-disk storage are declared here as collection
    config -- this is the entire replacement for the FAISS IndexScalarQuantizer
    training/rebuild step, and it applies to the collection from its first
    point rather than needing a post-upload compaction pass.
    """
    name = _collection_name(file_id)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=config.EMBEDDING_DIM,
                distance=qm.Distance.COSINE,
                # Keep raw vectors on disk; only the HNSW graph and quantized
                # vectors need to stay resident for search.
                on_disk=True,
            ),
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=config.QUANTIZATION_ALWAYS_RAM,
                )
            ),
            hnsw_config=qm.HnswConfigDiff(
                on_disk=True,
            ),
        )
    return name


def add_vectors(
    file_id: str,
    vectors: np.ndarray,
    sequences: list[int],
    start_bytes: list[int],
    end_bytes: list[int],
) -> None:
    """Upsert a batch of embedded passages.

    Idempotent by construction: re-upserting the same (file_id, sequence)
    overwrites the same point rather than creating a duplicate, so a worker
    retrying a batch after a crash cannot double-count a passage.
    """
    if vectors.shape[0] == 0:
        return

    client = get_client()
    collection = _ensure_collection(client, file_id)

    points = [
        qm.PointStruct(
            id=_point_id(file_id, sequence),
            vector=vector.tolist(),
            payload={
                "file_id": file_id,
                "sequence": sequence,
                "start_byte": start_byte,
                "end_byte": end_byte,
            },
        )
        for vector, sequence, start_byte, end_byte in zip(
            vectors, sequences, start_bytes, end_bytes
        )
    ]
    client.upsert(collection_name=collection, points=points, wait=True)


def search(file_id: str, query_vector: np.ndarray, top_k: int) -> list[dict]:
    """Return the closest passages as dicts with byte ranges and scores.

    Vectors are cosine-configured on the collection, so `score` is already
    cosine similarity -- no separate normalisation step needed here.
    """
    client = get_client()
    collection = _collection_name(file_id)
    if not client.collection_exists(collection):
        return []

    hits = client.query_points(
        collection_name=collection,
        query=query_vector.reshape(-1).tolist(),
        limit=top_k,
    ).points

    return [
        {
            "sequence": hit.payload["sequence"],
            "start_byte": hit.payload["start_byte"],
            "end_byte": hit.payload["end_byte"],
            "score": float(hit.score),
        }
        for hit in hits
    ]


def vector_count(file_id: str) -> int:
    """How many passages this file has indexed."""
    client = get_client()
    collection = _collection_name(file_id)
    if not client.collection_exists(collection):
        return 0
    return client.count(collection_name=collection, exact=True).count


def drop(file_id: str) -> None:
    """Delete this file's collection."""
    client = get_client()
    collection = _collection_name(file_id)
    if client.collection_exists(collection):
        client.delete_collection(collection)
