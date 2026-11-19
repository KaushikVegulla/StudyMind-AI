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

Parent-child retrieval: ingestion also stores small child chunks
linked to each parent chunk (see chunking.py). When enabled
(``PARENT_CHILD_RETRIEVAL``, default on), the search pools keep only
child chunks — plus any chunk that has no children, so older or
flag-off corpora still work. A matched child contributes its parent's
full text to the prompt as ``context_text``; the citation still names
the child's document, page and section.

HyDE (hypothetical document embeddings) is opt-in per request and
off by default: when on, an LLM writes a short hypothetical answer,
that text is embedded **instead of** the raw question for the vector
pool (the keyword pool always uses the raw question), and retrieval
proceeds as usual. Any HyDE failure — no LLM, an error, empty output —
falls back silently to normal retrieval.
"""
from __future__ import annotations

import os
import re

from . import embeddings, rerank

RRF_K = 50
POOL_SIZE = 20
MIN_SIMILARITY = 0.20


def parent_child_enabled() -> bool:
    """Parent-child retrieval flag (env PARENT_CHILD_RETRIEVAL,
    default on; "0"/"false"/"no"/"off" disable it)."""
    value = (os.getenv("PARENT_CHILD_RETRIEVAL") or "on").strip().lower()
    return value not in ("0", "false", "no", "off")


def searchable_only(chunks: list[dict]) -> list[dict]:
    """Drop parent chunks that have children: the children represent
    them in search and generation. Chunks without children (older
    corpora, flag-off ingests) are kept as-is."""
    parents_with_children = {c.get("parent_id") for c in chunks
                             if c.get("parent_id")}
    if not parents_with_children:
        return list(chunks)
    return [c for c in chunks
            if c.get("parent_id") or c.get("id") not in parents_with_children]

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


def _hypothetical_answer(question: str) -> str | None:
    """HyDE: ask the LLM for a short hypothetical answer to embed.
    None when no LLM is configured or anything goes wrong — HyDE must
    never break retrieval."""
    try:
        from . import llm

        if llm.configured_provider() is None:
            return None
        text = llm.generate_text(
            "Write a short, factual paragraph that could answer this "
            "question in a student's course notes. Do not say you are "
            "guessing; just write the paragraph.\n\n"
            f"Question: {question}")
        return text.strip() if text and text.strip() else None
    except Exception:
        return None


def retrieve(question: str, workspace_id: str | None, repo,
             top_k: int = 4, pool_size: int = POOL_SIZE,
             hyde: bool = False) -> tuple[list[dict], str]:
    """Return (chunks, mode). An empty list means: refuse, do not
    answer. Mode is 'hybrid', 'vector' or 'keyword'.

    With ``hyde=True`` the vector pool is built from the embedding of
    a hypothetical answer instead of the raw question (see the module
    docstring); the keyword pool always uses the raw question."""
    vector_hits: list[dict] = []
    if embeddings.embeddings_available():
        try:
            query_text = question
            if hyde:
                hypothetical = _hypothetical_answer(question)
                if hypothetical:
                    query_text = hypothetical
            query_embedding = embeddings.embed_query(query_text)
            vector_hits = repo.search_by_embedding(
                query_embedding, workspace_id, top_k=pool_size)
        except RuntimeError:
            vector_hits = []
    keyword_hits = _keyword_ranked(question, workspace_id, repo, pool_size)

    if parent_child_enabled():
        all_chunks = repo.list_chunks(workspace_id)
        parents_with_children = {c.get("parent_id") for c in all_chunks
                                 if c.get("parent_id")}
        if parents_with_children:
            def _is_searchable(chunk: dict) -> bool:
                return bool(chunk.get("parent_id")) or \
                    chunk.get("id") not in parents_with_children
            vector_hits = [h for h in vector_hits if _is_searchable(h)]
            keyword_hits = [h for h in keyword_hits if _is_searchable(h)]

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
    if parent_child_enabled():
        # Matched children answer with their parent's full text; the
        # citation metadata stays the child's (see citation_for).
        expanded = []
        for chunk in reranked:
            chunk = dict(chunk)
            parent_id = chunk.get("parent_id")
            if parent_id:
                parent = repo.get_chunk(parent_id)
                if parent:
                    chunk["context_text"] = parent["text"]
            expanded.append(chunk)
        reranked = expanded
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
        # Parent-child retrieval: the prompt sees the parent chunk's
        # full text; the label still names the child's location.
        + f": {c.get('context_text') or c['text']}"
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
