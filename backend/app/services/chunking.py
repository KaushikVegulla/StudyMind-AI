"""Split page text into overlapping chunks for embedding and retrieval.

Batch 1 rules (from the 2026-11-19 research report):

- A chunk never crosses a page boundary, so every citation can name an
  exact document and page.
- Chunk boundaries snap to sentence ends where possible instead of
  cutting mid-sentence at a fixed character offset.
- Each chunk records a ``section`` (the nearest heading-like line on the
  page, when one is found) and an ``embedding_text`` that prepends the
  context path ``Subject > Document > Section`` to the chunk text. The
  stored ``text`` stays the original wording, so citations quote the
  notes exactly as written.

Parent-child retrieval adds one more level: every parent chunk (the
page/section chunk above) also gets smaller ``child`` chunks of about
200 characters, built by :func:`child_chunks_for`. Retrieval matches
the children — small, focused passages rank more precisely — while
the prompt receives the parent chunk's full text as context, so the
answer still sees the whole passage. Citations keep pointing at the
child's document, page and section. Children carry ``parent_ref``
(the parent's ``ref``) so repositories can link them to the parent
row they create in the same ingest call.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADING_MAX_LEN = 80


def _looks_like_heading(line: str) -> bool:
    """Heuristic: a short line with no sentence punctuation that is not
    mostly lowercase prose is treated as a section heading."""
    line = line.strip()
    if not line or len(line) > _HEADING_MAX_LEN:
        return False
    if line[-1] in ".!?,":
        return False
    words = line.split()
    if len(words) > 10:
        return False
    if line[0].isdigit() or line.isupper() or line.istitle():
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]


def chunk_page_text(text: str, page: int, chunk_size: int = 800,
                    overlap: int = 120, subject: str = "",
                    document: str = "", section: str = "") -> list[dict]:
    """Return chunks for one page. Each chunk keeps its page number so
    answers can cite the exact page they came from."""
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    if not section and raw_lines and _looks_like_heading(raw_lines[0]):
        section = raw_lines[0].strip()
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = _split_sentences(cleaned)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                pieces.append(sentence[start:start + chunk_size])
                if start + chunk_size >= len(sentence):
                    break
                start += chunk_size - overlap
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        pieces.append(current)

    context_path = " > ".join(
        p for p in [subject.strip(), document.strip(), section.strip()] if p)
    chunks: list[dict] = []
    for index, piece in enumerate(pieces):
        embedding_text = f"{context_path}\n{piece}" if context_path else piece
        chunks.append({
            "page": page,
            "index": index,
            "text": piece,
            "section": section,
            "embedding_text": embedding_text,
            "ref": f"p{page}-{index}",
            "parent_ref": None,
        })
    return chunks


CHILD_CHUNK_SIZE = 200


def child_chunks_for(parent: dict, size: int = CHILD_CHUNK_SIZE) -> list[dict]:
    """Split one parent chunk into small child chunks for retrieval.

    Children inherit the parent's page, index and section (they live
    at the same place in the document) and carry ``parent_ref`` so the
    repository can store the link. A parent whose text already fits in
    one child still gets exactly one child, so every parent is
    reachable through the child index. Sentence boundaries are kept
    where possible, reusing the parent splitter.
    """
    text = " ".join((parent.get("text") or "").split())
    if not text:
        return []
    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                pieces.append(sentence[start:start + size])
                start += size
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)

    context_path = ""
    embedding_text = parent.get("embedding_text") or ""
    if "\n" in embedding_text:
        context_path = embedding_text.rsplit("\n", 1)[0]
    children: list[dict] = []
    for child_index, piece in enumerate(pieces):
        children.append({
            "page": parent["page"],
            "index": parent["index"],
            "child_index": child_index,
            "text": piece,
            "section": parent.get("section", ""),
            "embedding_text": f"{context_path}\n{piece}" if context_path
                              else piece,
            "ref": f"{parent['ref']}-c{child_index}",
            "parent_ref": parent["ref"],
        })
    return children


def with_children(parents: list[dict]) -> list[dict]:
    """Parents followed by their children, ready for one ingest call."""
    combined = list(parents)
    for parent in parents:
        combined.extend(child_chunks_for(parent))
    return combined
