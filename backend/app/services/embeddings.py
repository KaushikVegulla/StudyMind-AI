"""Embeddings interface.

Default model: all-MiniLM-L6-v2, run locally with sentence-transformers
(384 dimensions, no API key). The model loads lazily so the API runs fine
before the optional dependency is installed; callers check
``embeddings_available()`` and fall back to keyword retrieval when it is
not — retrieval mode is always labelled in API responses, never hidden.
"""
from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None
_load_failed = False


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def embed_texts(texts: list[str]) -> list[list[float]]:
    global _model, _load_failed
    if _load_failed:
        raise RuntimeError("sentence-transformers is unavailable in this run")
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            _load_failed = True
            raise RuntimeError(
                "sentence-transformers is not installed yet. "
                "Install the optional dependency in backend/requirements.txt "
                "to enable vector embeddings."
            ) from exc
        _model = SentenceTransformer(MODEL_NAME)
    vectors = _model.encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
