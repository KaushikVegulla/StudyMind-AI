"""Cross-encoder reranking of retrieved chunks.

Uses FlashRank (``ms-marco-MiniLM-L-12-v2``, a small local ONNX model,
no API key) when the optional ``flashrank`` package is installed.
When it is not installed, the input order (already fused by hybrid
RRF) is returned unchanged and the mode is labelled ``rrf`` so the
pipeline never pretends a reranker ran when it did not.
"""
from __future__ import annotations

MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

_ranker = None
_load_failed = False


def reranker_available() -> bool:
    try:
        import flashrank  # noqa: F401
        return True
    except ImportError:
        return False


def rerank(question: str, chunks: list[dict],
           top_k: int = 4) -> tuple[list[dict], str]:
    """Return (chunks, mode) where mode is 'flashrank' or 'rrf'."""
    global _ranker, _load_failed
    if not chunks or _load_failed or not reranker_available():
        return chunks[:top_k], "rrf"
    try:
        from flashrank import Ranker, RerankRequest

        if _ranker is None:
            _ranker = Ranker(model_name=MODEL_NAME)
        passages = [
            {"id": i, "text": c["text"], "meta": {}}
            for i, c in enumerate(chunks)
        ]
        results = _ranker.rerank(RerankRequest(query=question,
                                               passages=passages))
        ordered = [chunks[r["id"]] for r in results]
        for chunk, result in zip(ordered, results):
            chunk["rerank_score"] = float(result["score"])
        return ordered[:top_k], "flashrank"
    except Exception:
        _load_failed = True
        return chunks[:top_k], "rrf"
