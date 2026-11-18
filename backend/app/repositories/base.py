"""The repository interface every backend implements.

Data shapes are plain dicts, matching the API responses:

- workspace: {"id", "name", "description"}
- document:  {"id", "workspace_id", "filename", "pages", "chunk_count"}
- chunk:     {"id", "document_id", "workspace_id", "filename",
              "page", "index", "text", "embedding" (list[float] | None)}
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

    # vector search: returns chunk dicts nearest to the query embedding
    def search_by_embedding(self, embedding: list[float],
                            workspace_id: str | None,
                            top_k: int = 4) -> list[dict]: ...
