from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..repositories import get_repository
from ..services import embeddings, rag
from ..services.chunking import chunk_page_text, with_children
from ..services.pdf_parser import count_pages_needing_ocr, extract_document

router = APIRouter(prefix="/documents", tags=["documents"])

PDF_MAGIC = b"%PDF-"


def _max_upload_bytes() -> int:
    try:
        mb = float(os.getenv("MAX_UPLOAD_MB") or 25)
    except ValueError:
        mb = 25.0
    return int(mb * 1024 * 1024)


@router.get("")
def list_documents(workspace_id: str | None = None):
    return get_repository().list_documents(workspace_id)


@router.post("/upload", status_code=201)
async def upload_document(workspace_id: str, file: UploadFile = File(...)):
    repo = get_repository()
    workspace = repo.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported")
    data = await file.read()
    # Upload validation: size cap first (413), then the PDF magic
    # bytes (400) — an extension alone proves nothing about content.
    if len(data) > _max_upload_bytes():
        raise HTTPException(
            413, f"File is larger than the "
                 f"{os.getenv('MAX_UPLOAD_MB') or 25} MB upload limit")
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(400, "This file does not look like a PDF "
                                 "(missing %PDF- header)")
    try:
        extracted = extract_document(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    pages = extracted["pages"]
    pages_needing_ocr = count_pages_needing_ocr(pages)

    parents: list[dict] = []
    for page in pages:
        parents.extend(chunk_page_text(
            page["text"], page["page"],
            subject=workspace.get("name", ""),
            document=file.filename or ""))
    # Parent-child retrieval: children are the searchable units, the
    # parents stay as their context. Flag off = parents only, exactly
    # the pre-existing behaviour.
    chunks = with_children(parents) if rag.parent_child_enabled() else parents

    # Embed at ingest time when a local embeddings runtime is
    # available, so vector search can run at question time. The
    # embedded text carries the Subject > Document > Section context
    # path; the stored chunk text stays the original wording for exact
    # citations. Without a runtime, chunks are stored with no
    # embedding and retrieval falls back to keyword mode (labelled).
    # Parent chunks that have children are not embedded: their
    # children represent them in search.
    if chunks and embeddings.embeddings_available():
        parents_with_children = {c.get("parent_ref") for c in chunks
                                 if c.get("parent_ref")}
        searchable = [c for c in chunks
                      if c.get("ref") not in parents_with_children]
        try:
            vectors = embeddings.embed_texts(
                [c.get("embedding_text") or c["text"] for c in searchable])
            for chunk, vector in zip(searchable, vectors):
                chunk["embedding"] = vector
        except RuntimeError:
            pass

    document = repo.create_document(
        workspace_id, file.filename, len(pages), chunks,
        extractor=extracted["extractor"],
        pages_needing_ocr=pages_needing_ocr)
    if pages_needing_ocr:
        document = {**document, "warning":
                    f"{pages_needing_ocr} of {len(pages)} page(s) "
                    "appeared to be scanned images with no readable "
                    "text, so they contributed nothing to your notes. "
                    "OCR is not supported yet."}
    return document
