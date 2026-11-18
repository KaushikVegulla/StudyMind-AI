"""Split page text into overlapping chunks for embedding and retrieval."""
from __future__ import annotations


def chunk_page_text(text: str, page: int, chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    """Return chunks for one page. Each chunk keeps its page number so answers
    can cite the exact page they came from."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[dict] = []
    start, index = 0, 0
    while start < len(cleaned):
        end = start + chunk_size
        piece = cleaned[start:end]
        chunks.append({"page": page, "index": index, "text": piece})
        if end >= len(cleaned):
            break
        start = end - overlap
        index += 1
    return chunks
