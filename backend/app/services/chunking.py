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
        })
    return chunks
