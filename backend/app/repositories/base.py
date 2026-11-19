"""The repository interface every backend implements.

Data shapes are plain dicts, matching the API responses:

- workspace: {"id", "name", "description"}
- document:  {"id", "workspace_id", "filename", "pages", "chunk_count",
              "extractor", "pages_needing_ocr"}
- chunk:     {"id", "document_id", "workspace_id", "filename",
              "page", "index", "text", "section",
              "embedding" (list[float] | None),
              "parent_id" (chunk id | None — set on child chunks used
              for parent-child retrieval)}
- flashcard: {"id", "workspace_id", "document_id", "front", "back",
              "ease_factor", "interval_days", "repetitions", "due_at",
              "fsrs_stability", "fsrs_difficulty", "last_review_at"}
              (the scheduling engine — services/scheduling.py — fills
              the fields of the active engine; the rest stay at their
              defaults)
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
                        chunks: list[dict], extractor: str = "pypdf",
                        pages_needing_ocr: int = 0) -> dict: ...
    def list_documents(self, workspace_id: str | None = None) -> list[dict]: ...
    def list_chunks(self, workspace_id: str | None = None) -> list[dict]: ...
    def get_chunk(self, chunk_id: str) -> dict | None: ...

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

    # flashcards with scheduling state (services/scheduling.py picks
    # the engine: FSRS by default, SM-2 when SRS_ENGINE=sm2)
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
