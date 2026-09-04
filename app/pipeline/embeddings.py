"""Sentence-transformer wrapper.

The model is loaded once and shared by every worker: it is ~90 MB of weights
and re-loading per request would dominate both memory and latency.

Vectors are L2-normalised at encode time so cosine similarity reduces to an
inner product, which is what the FAISS index is built for.
"""

import threading

import numpy as np

from .. import config

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the model on first use, then reuse it.

    Deferred rather than imported at module load so that tests and the
    line-buffer path don't pay for torch unless they actually embed.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed passages, returning normalised float32 vectors of shape (n, dim)."""
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a search query as a (1, dim) array ready for FAISS."""
    return embed_texts([query])


def warm_up() -> None:
    """Pull the model into memory at startup so the first search isn't slow."""
    embed_texts(["warm up"])
