"""Retrieval: find the chunks a question should be answered from.

Batch 1 pipeline (research report 2026-11-19):

1. **Hybrid search.** Vector similarity and keyword ranking each
   produce a candidate pool, fused with Reciprocal Rank Fusion
   (``rrf_k = 50``). Hybrid beats either signal alone because exact
   terms (names, formulas, code) are where pure vector search fails.
2. **Refusal threshold.** If the best candidate is too weak (vector
   similarity below ``MIN_SIMILARITY`` and no keyword overlap), the
   question is refused with no LLM call, instead of answering from
   unrelated notes.
3. **Reranking.** The fused pool is reranked down to ``top_k`` with a
   local cross-encoder when FlashRank is installed (see rerank.py).
4. **Code-assigned citations.** Sources are labelled ``[S1]``, ``[S2]``
   by this code, never by the LLM. The prompt instructs the model to
   cite only those labels, so a page number in an answer always came
   from stored chunk metadata.

Modes returned: ``hybrid``, ``vector`` or ``keyword``, naming the
signals that actually contributed. The chat response labels the mode.
"""
from __future__ import annotations

import re

from . import embeddings, rerank

RRF_K = 50
POOL_SIZE = 20
MIN_SIMILARITY = 0.20

_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "how", "does",
    "do", "explain", "define", "are", "was", "were", "and", "or", "to",
    "for", "with", "why", "when", "which", "who", "it", "this", "that",
}


def _terms(question: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return {w for w in words if w not in _STOPWORDS} or set(words)


def keyword_score(question_terms: set[str], text: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(question_terms & words)


def _keyword_ranked(question: str, workspace_id: str | None, repo,
                    pool: int) -> list[dict]:
    """Keyword pool: use the repository keyword search when it has one
    (Postgres tsvector), else rank the workspace chunks in Python."""
    search = getattr(repo, "search_by_keyword", None)
    if callable(search):
        try:
            hits = search(question, workspace_id, top_k=pool)
            if hits:
                return hits
        except Exception:
            pass
    terms = _terms(question)
    candidates = repo.list_chunks(workspace_id)
    ranked = sorted(candidates,
                    key=lambda c: keyword_score(terms, c["text"]),
                    reverse=True)
    hits = []
    for chunk in ranked:
        score = keyword_score(terms, chunk["text"])
        if score <= 0:
            break
        chunk = dict(chunk)
        chunk["keyword_score"] = score
        hits.append(chunk)
        if len(hits) >= pool:
            break
    return hits


def _rrf_fuse(pools: list[list[dict]]) -> list[dict]:
    """Reciprocal Rank Fusion over ranked pools, keyed by chunk id
    (falling back to document/page/index identity)."""
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for pool in pools:
        for rank, chunk in enumerate(pool):
            key = str(chunk.get("id") or
                      f"{chunk.get('document_id')}:{chunk['page']}:{chunk['index']}")
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in fused:
                fused[key] = dict(chunk)
            else:
                for field in ("similarity", "keyword_score"):
                    if chunk.get(field) is not None:
                        fused[key][field] = chunk[field]
    ordered = sorted(fused.values(),
                     key=lambda c: scores[str(c.get("id") or
                        f"{c.get('document_id')}:{c['page']}:{c['index']}")],
                     reverse=True)
    for chunk in ordered:
        key = str(chunk.get("id") or
                  f"{chunk.get('document_id')}:{chunk['page']}:{chunk['index']}")
        chunk["rrf_score"] = round(scores[key], 6)
    return ordered


def retrieve(question: str, workspace_id: str | None, repo,
             top_k: int = 4, pool_size: int = POOL_SIZE) -> tuple[list[dict], str]:
    """Return (chunks, mode). An empty list means: refuse, do not
    answer. Mode is 'hybrid', 'vector' or 'keyword'."""
    vector_hits: list[dict] = []
    if embeddings.embeddings_available():
        try:
            query_embedding = embeddings.embed_query(question)
            vector_hits = repo.search_by_embedding(
                query_embedding, workspace_id, top_k=pool_size)
        except RuntimeError:
            vector_hits = []
    keyword_hits = _keyword_ranked(question, workspace_id, repo, pool_size)

    # Refusal threshold: weak vector matches with no keyword support
    # are not evidence. Drop them before fusion.
    if vector_hits:
        best_sim = max((h.get("similarity") or 0.0) for h in vector_hits)
        if best_sim < MIN_SIMILARITY and not keyword_hits:
            return [], "below_threshold"
        vector_hits = [h for h in vector_hits
                       if (h.get("similarity") or 0.0) >= MIN_SIMILARITY
                       or keyword_hits]
    if not vector_hits and not keyword_hits:
        return [], "keyword"

    if vector_hits and keyword_hits:
        mode = "hybrid"
    elif vector_hits:
        mode = "vector"
    else:
        mode = "keyword"

    fused = _rrf_fuse([p for p in (vector_hits, keyword_hits) if p])
    reranked, rerank_mode = rerank.rerank(question, fused, top_k=top_k)
    if rerank_mode == "flashrank":
        mode = f"{mode}+rerank"
    return reranked, mode


def assign_source_ids(chunks: list[dict]) -> list[dict]:
    """Label chunks [S1], [S2], ... in retrieval order. Code assigns
    these, never the LLM, so citations cannot be invented."""
    labelled = []
    for i, chunk in enumerate(chunks, start=1):
        chunk = dict(chunk)
        chunk["source_id"] = f"S{i}"
        labelled.append(chunk)
    return labelled


def citation_for(chunk: dict) -> dict:
    return {
        "source_id": chunk.get("source_id"),
        "document": chunk.get("filename", "unknown"),
        "page": chunk["page"],
        "chunk_index": chunk["index"],
        "section": chunk.get("section", ""),
        "text": chunk["text"][:300],
    }


ANSWER_STYLES = ("concise", "standard", "exam")

_STYLE_INSTRUCTIONS = {
    "concise": "Style: concise. Answer directly in at most three short "
               "sentences. No preamble, no restating the question.",
    "standard": "",
    "exam": "Style: exam. Structure the answer for marks: start with a "
            "one-line definition, then the key points as a short list, "
            "then one example or application if the sources give one.",
}


def build_prompt(question: str, chunks: list[dict],
                 style: str = "standard") -> str:
    if style not in _STYLE_INSTRUCTIONS:
        style = "standard"
    chunks = assign_source_ids(chunks)
    sep = chr(10) + chr(10)
    context = sep.join(
        f"[{c['source_id']}] {c.get('filename', c.get('document_id', 'unknown'))}"
        f" p.{c['page']}"
        + (f" - {c['section']}" if c.get("section") else "")
        + f": {c['text']}"
        for c in chunks
    )
    style_line = _STYLE_INSTRUCTIONS[style]
    return (
        "Answer the question using ONLY the sources below. "
        "Cite every claim with its source label, like [S1]. "
        "Use ONLY the labels shown. Never invent a document name, "
        "page number or source label that is not in the list. "
        "If the sources do not contain the answer, say exactly: "
        "I could not find this in your uploaded material."
        + (sep + style_line if style_line else "")
        + sep + "Sources:" + chr(10) + context
        + sep + f"Question: {question}"
    )
