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
        self.flashcards: dict[str, dict] = {}
        self.generated: dict[tuple, dict] = {}

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
        self.flashcards = {f: card for f, card in self.flashcards.items()
                           if card["workspace_id"] != workspace_id}
        self.generated = {k: v for k, v in self.generated.items()
                          if k[0] != workspace_id}
        return True

    # -- documents and chunks ----------------------------------------------
    def create_document(self, workspace_id: str, filename: str, pages: int,
                        chunks: list[dict], extractor: str = "pypdf",
                        pages_needing_ocr: int = 0) -> dict:
        doc_id = self._next_id("doc")
        # Two passes: parents first so child chunks can be linked to
        # the parent id assigned here (parent-child retrieval).
        ref_to_id: dict[str, str] = {}
        ordered = sorted(chunks, key=lambda c: bool(c.get("parent_ref")))
        for chunk in ordered:
            cid = self._next_id("chunk")
            if chunk.get("ref"):
                ref_to_id[chunk["ref"]] = cid
            parent_id = None
            if chunk.get("parent_ref"):
                parent_id = ref_to_id.get(chunk["parent_ref"])
            self.chunks[cid] = {
                "id": cid, "document_id": doc_id, "workspace_id": workspace_id,
                "filename": filename, "page": chunk["page"],
                "index": chunk["index"], "text": chunk["text"],
                "section": chunk.get("section", ""),
                "embedding": chunk.get("embedding"),
                "parent_id": parent_id,
            }
        doc = {"id": doc_id, "workspace_id": workspace_id, "filename": filename,
               "pages": pages, "chunk_count": len(chunks),
               "extractor": extractor,
               "pages_needing_ocr": pages_needing_ocr}
        self.documents[doc_id] = doc
        return doc

    def get_chunk(self, chunk_id: str) -> dict | None:
        chunk = self.chunks.get(chunk_id)
        return dict(chunk) if chunk else None

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

    # -- flashcards ------------------------------------------------------
    def save_flashcards(self, workspace_id: str,
                        document_id: str | None,
                        cards: list[dict]) -> list[dict]:
        from ..services.scheduling import new_card_state

        saved = []
        for card in cards:
            fid = self._next_id("card")
            record = {"id": fid, "workspace_id": workspace_id,
                      "document_id": document_id,
                      "front": card["front"], "back": card["back"],
                      **new_card_state()}
            self.flashcards[fid] = record
            saved.append(dict(record))
        return saved

    def list_due_flashcards(self, workspace_id: str | None = None,
                            document_id: str | None = None,
                            now: str | None = None) -> list[dict]:
        from ..services.srs import now_utc

        cutoff = now or now_utc().isoformat()
        due = [dict(c) for c in self.flashcards.values()
               if c["due_at"] <= cutoff
               and (workspace_id is None or c["workspace_id"] == workspace_id)
               and (document_id is None or c["document_id"] == document_id)]
        due.sort(key=lambda c: c["due_at"])
        return due

    def get_flashcard(self, flashcard_id: str) -> dict | None:
        card = self.flashcards.get(flashcard_id)
        return dict(card) if card else None

    def update_flashcard_review(self, flashcard_id: str,
                                state: dict) -> dict | None:
        card = self.flashcards.get(flashcard_id)
        if card is None:
            return None
        card.update({k: state[k] for k in
                     ("ease_factor", "interval_days", "repetitions",
                      "due_at", "fsrs_stability", "fsrs_difficulty",
                      "last_review_at") if k in state})
        return dict(card)

    # -- generation cache --------------------------------------------------
    def get_generated(self, workspace_id: str, document_id: str | None,
                      kind: str, count: int) -> dict | None:
        payload = self.generated.get((workspace_id, document_id, kind, count))
        return dict(payload) if payload else None

    def save_generated(self, workspace_id: str, document_id: str | None,
                       kind: str, count: int, payload: dict) -> None:
        self.generated[(workspace_id, document_id, kind, count)] = dict(payload)
