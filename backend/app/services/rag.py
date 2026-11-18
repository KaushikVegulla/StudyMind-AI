"""Retrieval helpers for the scaffold.

The scaffold ranks chunks by keyword overlap so the upload -> chunk -> cite
pipeline can be verified end to end without a database or an LLM key.
This is replaced by pgvector similarity search in the persistence step;
the citation shape returned to the frontend does not change.
"""
from __future__ import annotations

import re

from .. import store


def _score(question_terms: set[str], text: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(question_terms & words)


def retrieve(question: str, workspace_id: str | None, top_k: int = 4) -> list[dict]:
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates = [
        c for c in store.chunks.values()
        if workspace_id is None or c["workspace_id"] == workspace_id
    ]
    ranked = sorted(candidates, key=lambda c: _score(terms, c["text"]), reverse=True)
    return [c for c in ranked if _score(terms, c["text"]) > 0][:top_k]


def citation_for(chunk: dict) -> dict:
    doc = store.documents.get(chunk["document_id"], {})
    return {
        "document": doc.get("filename", "unknown"),
        "page": chunk["page"],
        "chunk_index": chunk["index"],
        "text": chunk["text"][:300],
    }


def build_prompt(question: str, chunks: list[dict]) -> str:
    sep = chr(10) + chr(10)
    context = sep.join(
        f"[{c['document_id']} p.{c['page']}] {c['text']}" for c in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite the document and page for every claim. "
        "If the context does not contain the answer, say so."
        + sep + "Context:" + chr(10) + context
        + sep + f"Question: {question}"
    )
