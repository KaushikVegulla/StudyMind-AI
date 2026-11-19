"""Embeddings interface.

Model: BAAI/bge-small-en-v1.5, 384 dimensions, run locally with no
API key. It replaced all-MiniLM-L6-v2, which truncates beyond 256
word pieces and scores far lower on retrieval benchmarks (MTEB
retrieval about 41.9 vs 51.7). Same 384 dimensions, so stored
pgvector columns need no migration, but chunks embedded by another
model must be re-embedded after an upload or re-ingest because the
vector space changed.

Two local runtimes can serve the model, tried in this order:

1. **fastembed** — ONNX runtime build. Small install, no torch, so it
   is the one that fits a free-tier web service (512 MB RAM class).
2. **sentence-transformers** — the original runtime; heavier (torch),
   kept for machines that already have it.

If neither is importable, embeddings are unavailable and retrieval
falls back to keyword mode; the active embedder is always reported
honestly (see ``active_embedder_name`` and the /health endpoint),
never hidden. ``EMBEDDINGS_BACKEND`` can force one backend
(``fastembed`` / ``sentence-transformers``) or disable embeddings
(``none``) — CI uses ``none`` so tests never download a model.

bge models expect an instruction prefix on the QUERY side only:
``Represent this sentence for searching relevant passages: ``.
Passages are embedded without it. The prefix is applied by
``embed_query`` for both runtimes, so behaviour is identical no
matter which one is installed. Use ``embed_query`` for questions and
``embed_texts`` for document chunks.

Models load lazily on first use, so the API runs fine before the
optional dependency is installed; a backend whose model fails to
load is marked failed for the rest of the process and the next
backend in the chain is tried instead.
"""
from __future__ import annotations

import os

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_BACKENDS = ("fastembed", "sentence-transformers")

_models: dict[str, object] = {}
_load_failed: set[str] = set()


def _forced_backend() -> str | None:
    value = (os.getenv("EMBEDDINGS_BACKEND") or "").strip().lower()
    return value or None


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _backend_available(name: str) -> bool:
    forced = _forced_backend()
    if forced == "none":
        return False
    if forced is not None and forced != name:
        return False
    if name in _load_failed:
        return False
    module = "fastembed" if name == "fastembed" else "sentence_transformers"
    return _importable(module)


def active_embedder_name() -> str | None:
    """Name of the runtime that will serve embeddings, or None.

    One of ``"fastembed"``, ``"sentence-transformers"`` or None when
    no runtime is installed (keyword-only retrieval)."""
    for name in _BACKENDS:
        if _backend_available(name):
            return name
    return None


def embeddings_available() -> bool:
    return active_embedder_name() is not None


def _load_model(name: str):
    if name in _models:
        return _models[name]
    if name == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=MODEL_NAME)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
    _models[name] = model
    return model


def _encode_with(name: str, texts: list[str]) -> list[list[float]]:
    model = _load_model(name)
    if name == "fastembed":
        vectors = model.embed(texts)
    else:
        vectors = model.encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed passages. Raises RuntimeError when no runtime can serve
    (not installed, or every installed runtime failed to load) so
    callers can fall back to keyword retrieval."""
    errors = []
    for name in _BACKENDS:
        if not _backend_available(name):
            continue
        try:
            return _encode_with(name, texts)
        except Exception as exc:  # model download / load failure
            _load_failed.add(name)
            errors.append(f"{name}: {exc}")
    detail = "; ".join(errors) if errors else "no embeddings runtime installed"
    raise RuntimeError(
        f"Embeddings unavailable ({detail}). Install fastembed "
        "(see backend/requirements.txt) to enable vector embeddings.")


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a search question with the bge query instruction prefix."""
    return embed_texts([QUERY_PREFIX + text])[0]
