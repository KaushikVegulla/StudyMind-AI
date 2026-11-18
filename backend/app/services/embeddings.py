"""Embeddings interface.

Default model: all-MiniLM-L6-v2, run locally with sentence-transformers
(384 dimensions, no API key). The model loads lazily so the scaffold API
runs fine before the optional dependency is installed.
"""
from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed yet. "
                "Install the optional dependency in backend/requirements.txt "
                "to enable vector embeddings."
            ) from exc
        _model = SentenceTransformer(MODEL_NAME)
    vectors = _model.encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]
