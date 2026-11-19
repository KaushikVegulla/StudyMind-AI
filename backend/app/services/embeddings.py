"""Embeddings interface.

Default model: BAAI/bge-small-en-v1.5, run locally with
sentence-transformers (384 dimensions, no API key). It replaces
all-MiniLM-L6-v2, which truncates beyond 256 word pieces and scores
far lower on retrieval benchmarks (MTEB retrieval about 41.9 vs 51.7).
Same 384 dimensions, so stored pgvector columns need no migration,
but existing chunks must be re-embedded after an upload or re-ingest
because the vector space changed.

bge models expect an instruction prefix on the QUERY side only:
``Represent this sentence for searching relevant passages: ``.
Passages are embedded without it. Use ``embed_query`` for questions
and ``embed_texts`` for document chunks.

The model loads lazily so the API runs fine before the optional
dependency is installed; callers check ``embeddings_available()`` and
fall back to keyword retrieval when it is not. Retrieval mode is
always labelled in API responses, never hidden.
"""
from __future__ import annotations

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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


def embed_query(text: str) -> list[float]:
    """Embed a search question with the bge query instruction prefix."""
    return embed_texts([QUERY_PREFIX + text])[0]
