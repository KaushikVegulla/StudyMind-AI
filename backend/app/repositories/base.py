"""The repository interface every backend implements.

Data shapes are plain dicts, matching the API responses:

- workspace: {"id", "name", "description"}
- document:  {"id", "workspace_id", "filename", "pages", "chunk_count"}
- chunk:     {"id", "document_id", "workspace_id", "filename",
              "page", "index", "text", "section",
              "embedding" (list[float] | None)}
- flashcard: {"id", "workspace_id", "document_id", "front", "back",
              "ease_factor", "interval_days", "repetitions", "due_at"}
- generated: cached summary/flashcards/quiz payloads, keyed by
  workspace + document + kind + count (see the generation cache
  methods below)
"""
from __future__ import annotations

from typing import Protocol


class Repository(Protocol):
    name: str

    def seed_if_empty(self) -> None: ...
    def counts(self) -> dict: ...

    # workspaces
    def list_workspaces(self) -> list[dict]: ...
    def create_workspace(self, name: str, description: str) -> dict: ...
    def get_workspace(self, workspace_id: str) -> dict | None: ...
    def delete_workspace(self, workspace_id: str) -> bool: ...

    # documents and chunks
    def create_document(self, workspace_id: str, filename: str, pages: int,
                        chunks: list[dict]) -> dict: ...
    def list_documents(self, workspace_id: str | None = None) -> list[dict]: ...
    def list_chunks(self, workspace_id: str | None = None) -> list[dict]: ...

    # vector search: returns chunk dicts nearest to the query embedding,
    # each with a "similarity" score
    def search_by_embedding(self, embedding: list[float],
                            workspace_id: str | None,
                            top_k: int = 4) -> list[dict]: ...

    # keyword search: returns chunk dicts ranked by keyword relevance,
    # each with a "keyword_score"
    def search_by_keyword(self, question: str,
                          workspace_id: str | None,
                          top_k: int = 20) -> list[dict]: ...

    # flashcards with SM-2 scheduling state (services/srs.py)
    def save_flashcards(self, workspace_id: str,
                        document_id: str | None,
                        cards: list[dict]) -> list[dict]: ...
    def list_due_flashcards(self, workspace_id: str | None = None,
                            document_id: str | None = None,
                            now: str | None = None) -> list[dict]: ...
    def get_flashcard(self, flashcard_id: str) -> dict | None: ...
    def update_flashcard_review(self, flashcard_id: str,
                                state: dict) -> dict | None: ...

    # generation cache: one stored payload per
    # (workspace, document, kind, count); None document = whole workspace
    def get_generated(self, workspace_id: str, document_id: str | None,
                      kind: str, count: int) -> dict | None: ...
    def save_generated(self, workspace_id: str, document_id: str | None,
                       kind: str, count: int, payload: dict) -> None: ...
