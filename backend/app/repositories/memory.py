"""In-memory repository: zero-setup backend for local runs and tests."""
from __future__ import annotations

import itertools
import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryRepository:
    name = "memory"

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.workspaces: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"

    # -- lifecycle ---------------------------------------------------------
    def seed_if_empty(self) -> None:
        if self.workspaces:
            return
        for name, desc in [
            ("Data Structures", "Data Structures with Competitive Coding - Sem 3"),
            ("Databases", "Relational and Distributed Databases - Sem 3"),
            ("Cloud", "Cloud Practitioner AWS notes"),
        ]:
            self.create_workspace(name, desc)

    def counts(self) -> dict:
        return {"workspaces": len(self.workspaces),
                "documents": len(self.documents),
                "chunks": len(self.chunks)}

    # -- workspaces --------------------------------------------------------
    def list_workspaces(self) -> list[dict]:
        return list(self.workspaces.values())

    def create_workspace(self, name: str, description: str) -> dict:
        wid = self._next_id("ws")
        ws = {"id": wid, "name": name, "description": description}
        self.workspaces[wid] = ws
        return ws

    def get_workspace(self, workspace_id: str) -> dict | None:
        return self.workspaces.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> bool:
        if workspace_id not in self.workspaces:
            return False
        del self.workspaces[workspace_id]
        self.chunks = {c: ch for c, ch in self.chunks.items()
                       if ch["workspace_id"] != workspace_id}
        self.documents = {d: doc for d, doc in self.documents.items()
                          if doc["workspace_id"] != workspace_id}
        return True

    # -- documents and chunks ----------------------------------------------
    def create_document(self, workspace_id: str, filename: str, pages: int,
                        chunks: list[dict]) -> dict:
        doc_id = self._next_id("doc")
        for chunk in chunks:
            cid = self._next_id("chunk")
            self.chunks[cid] = {
                "id": cid, "document_id": doc_id, "workspace_id": workspace_id,
                "filename": filename, "page": chunk["page"],
                "index": chunk["index"], "text": chunk["text"],
                "section": chunk.get("section", ""),
                "embedding": chunk.get("embedding"),
            }
        doc = {"id": doc_id, "workspace_id": workspace_id, "filename": filename,
               "pages": pages, "chunk_count": len(chunks)}
        self.documents[doc_id] = doc
        return doc

    def list_documents(self, workspace_id: str | None = None) -> list[dict]:
        docs = list(self.documents.values())
        if workspace_id:
            docs = [d for d in docs if d["workspace_id"] == workspace_id]
        return docs

    def list_chunks(self, workspace_id: str | None = None) -> list[dict]:
        chunks = list(self.chunks.values())
        if workspace_id:
            chunks = [c for c in chunks if c["workspace_id"] == workspace_id]
        return chunks

    # -- vector search -------------------------------------------------------
    def search_by_embedding(self, embedding: list[float],
                            workspace_id: str | None,
                            top_k: int = 4) -> list[dict]:
        candidates = [dict(c) for c in self.list_chunks(workspace_id)
                      if c.get("embedding")]
        for chunk in candidates:
            chunk["similarity"] = _cosine(embedding, chunk["embedding"])
        candidates.sort(key=lambda c: c["similarity"], reverse=True)
        return candidates[:top_k]

    # -- keyword search --------------------------------------------------------
    def search_by_keyword(self, question: str,
                          workspace_id: str | None,
                          top_k: int = 20) -> list[dict]:
        import re

        from ..services.rag import _terms, keyword_score

        terms = _terms(question)
        hits = []
        for chunk in self.list_chunks(workspace_id):
            score = keyword_score(terms, chunk["text"])
            if score > 0:
                chunk = dict(chunk)
                chunk["keyword_score"] = score
                hits.append(chunk)
        hits.sort(key=lambda c: c["keyword_score"], reverse=True)
        return hits[:top_k]
