"""Extract text from a PDF, page by page, so citations can name the page.

Extractor interface: one function, two implementations.

- **pypdf** (default) — pure-Python, always installed, extracts the
  text layer page by page.
- **docling** (optional) — when the docling package is importable it
  is used instead: it understands tables and reading order, which
  plain text extraction mangles. It is deliberately NOT in
  requirements.txt: it pulls torch-scale dependencies that do not
  fit the free tier. If docling is present but fails on a file, the
  code falls back to pypdf and the document records the extractor
  that actually ran (``extract_document`` returns its name).

Scanned pages: a page whose extracted text is shorter than
``MIN_PAGE_TEXT_CHARS`` almost certainly holds a scanned image, not
text. Such pages are counted (``pages_needing_ocr``) and reported
honestly on the document — there is no OCR in this app, so those
pages contribute no chunks and the upload response says so.
"""
from __future__ import annotations

import io

MIN_PAGE_TEXT_CHARS = 20

DEFAULT_EXTRACTOR = "pypdf"


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_with_pypdf(pdf_bytes: bytes) -> list[dict]:
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


def _extract_with_docling(pdf_bytes: bytes) -> list[dict]:
    """Docling extraction. Only called when docling is importable.

    Docling works on streams; per-page text comes from the converted
    document's page items when the installed version exposes them,
    otherwise the whole-document text is returned as a single page —
    a deliberate simplification, since this path is an optional
    upgrade and never the one the free-tier deployment runs.
    """
    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter

    stream = DocumentStream(name="upload.pdf", stream=io.BytesIO(pdf_bytes))
    result = DocumentConverter().convert(stream)
    document = result.document
    pages: list[dict] = []
    page_items = getattr(document, "pages", None)
    if page_items:
        for page_no in sorted(page_items):
            item = page_items[page_no]
            export = getattr(item, "export_to_text", None)
            text = export() if callable(export) else ""
            pages.append({"page": int(page_no), "text": text or ""})
    if not pages:
        pages = [{"page": 1, "text": document.export_to_text() or ""}]
    return pages


def extract_document(pdf_bytes: bytes) -> dict:
    """Return {"pages": [{"page", "text"}, ...], "extractor": name}.

    The extractor that actually ran is named: "docling" only when
    docling was importable AND succeeded, otherwise "pypdf".
    Raises ValueError on bad PDFs (from whichever extractor ran last).
    """
    if docling_available():
        try:
            return {"pages": _extract_with_docling(pdf_bytes),
                    "extractor": "docling"}
        except Exception:
            pass  # fall through to the default extractor
    return {"pages": _extract_with_pypdf(pdf_bytes),
            "extractor": DEFAULT_EXTRACTOR}


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...]. Raises ValueError on bad PDFs."""
    return extract_document(pdf_bytes)["pages"]


def count_pages_needing_ocr(pages: list[dict]) -> int:
    """Pages with almost no extracted text: scanned images, most
    likely. They contribute nothing until real OCR exists."""
    return sum(1 for p in pages
               if len((p.get("text") or "").strip()) < MIN_PAGE_TEXT_CHARS)
