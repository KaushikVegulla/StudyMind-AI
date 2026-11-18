"""Repository selection.

StudyMind stores its data through one small interface with two backends:

- ``PostgresRepository`` — Supabase Postgres + pgvector, used when
  ``DATABASE_URL`` is set. This is the real persistence layer.
- ``MemoryRepository`` — in-memory, used when no database is configured so
  the app and the tests run anywhere with zero setup.

Routers only ever talk to the interface in ``base.py``, so switching from
local development to Supabase is an environment change, not a code change.
"""
from __future__ import annotations

import os

from .base import Repository
from .memory import MemoryRepository

_repo: Repository | None = None


def get_repository() -> Repository:
    global _repo
    if _repo is not None:
        return _repo
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        from .postgres import PostgresRepository

        _repo = PostgresRepository(database_url)
    else:
        _repo = MemoryRepository()
    _repo.seed_if_empty()
    return _repo


def reset_repository() -> None:
    """Test helper: drop the cached repository."""
    global _repo
    _repo = None
