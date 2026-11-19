from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..repositories import get_repository
from ..services import embeddings
from ..services.chunking import chunk_page_text
from ..services.pdf_parser import extract_pages

router = APIRouter(prefix="/documents", tags=["documents"])


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
        raise HTTPException(400, "Only PDF uploads are supported in the MVP")
    data = await file.read()
    try:
        pages = extract_pages(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    chunks: list[dict] = []
    for page in pages:
        chunks.extend(chunk_page_text(
            page["text"], page["page"],
            subject=workspace.get("name", ""),
            document=file.filename or ""))

    # Embed at ingest time when the local model is available, so vector
    # search can run at question time. The embedded text carries the
    # Subject > Document > Section context path; the stored chunk text
    # stays the original wording for exact citations. Without the model,
    # chunks are stored with no embedding and retrieval falls back to
    # keyword mode (labelled).
    if chunks and embeddings.embeddings_available():
        try:
            vectors = embeddings.embed_texts(
                [c.get("embedding_text") or c["text"] for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk["embedding"] = vector
        except RuntimeError:
            pass

    return repo.create_document(workspace_id, file.filename, len(pages), chunks)
