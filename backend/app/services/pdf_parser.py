"""Extract text from a PDF, page by page, so citations can name the page."""
from __future__ import annotations

import io


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...]. Raises ValueError on bad PDFs."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is not installed - see backend/requirements.txt") from exc
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError("Could not read this file as a PDF") from exc
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append({"page": i, "text": page.extract_text() or ""})
    return pages
