from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import store
from ..services.chunking import chunk_page_text
from ..services.pdf_parser import extract_pages

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(workspace_id: str | None = None):
    docs = list(store.documents.values())
    if workspace_id:
        docs = [d for d in docs if d["workspace_id"] == workspace_id]
    return docs


@router.post("/upload", status_code=201)
async def upload_document(workspace_id: str, file: UploadFile = File(...)):
    if workspace_id not in store.workspaces:
        raise HTTPException(404, "Workspace not found")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported in the MVP")
    data = await file.read()
    try:
        pages = extract_pages(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    doc_id = store.next_id("doc")
    total_chunks = 0
    for page in pages:
        for chunk in chunk_page_text(page["text"], page["page"]):
            cid = store.next_id("chunk")
            store.chunks[cid] = {
                "id": cid, "document_id": doc_id, "workspace_id": workspace_id,
                "page": chunk["page"], "index": chunk["index"], "text": chunk["text"],
            }
            total_chunks += 1
    doc = {
        "id": doc_id, "workspace_id": workspace_id, "filename": file.filename,
        "pages": len(pages), "chunk_count": total_chunks,
    }
    store.documents[doc_id] = doc
    return doc
