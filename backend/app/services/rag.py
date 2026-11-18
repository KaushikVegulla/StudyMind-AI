"""Retrieval: find the chunks a question should be answered from.

Two modes, both returning the same chunk/citation shapes:

- ``vector`` — the question is embedded and the repository runs a cosine
  similarity search (pgvector ``<=>`` on Postgres, in-Python cosine on the
  in-memory backend). Used when embeddings are available and the stored
  chunks have embeddings.
- ``keyword`` — keyword-overlap ranking over the workspace's chunks. The
  fallback when embeddings are not installed, so the pipeline still works
  with zero setup. The chat response labels which mode answered.
"""
from __future__ import annotations

import re

from . import embeddings


def _score(question_terms: set[str], text: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(question_terms & words)


def retrieve(question: str, workspace_id: str | None, repo,
             top_k: int = 4) -> tuple[list[dict], str]:
    """Return (chunks, mode) where mode is 'vector' or 'keyword'."""
    if embeddings.embeddings_available():
        try:
            query_embedding = embeddings.embed_one(question)
            hits = repo.search_by_embedding(query_embedding, workspace_id,
                                            top_k=top_k)
            if hits:
                return hits, "vector"
        except RuntimeError:
            pass
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates = repo.list_chunks(workspace_id)
    ranked = sorted(candidates,
                    key=lambda c: _score(terms, c["text"]), reverse=True)
    return [c for c in ranked if _score(terms, c["text"]) > 0][:top_k], "keyword"


def citation_for(chunk: dict) -> dict:
    return {
        "document": chunk.get("filename", "unknown"),
        "page": chunk["page"],
        "chunk_index": chunk["index"],
        "text": chunk["text"][:300],
    }


def build_prompt(question: str, chunks: list[dict]) -> str:
    sep = chr(10) + chr(10)
    context = sep.join(
        f"[{c.get('filename', c['document_id'])} p.{c['page']}] {c['text']}"
        for c in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite the document and page for every claim. "
        "If the context does not contain the answer, say so."
        + sep + "Context:" + chr(10) + context
        + sep + f"Question: {question}"
    )
