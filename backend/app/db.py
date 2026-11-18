"""Database access lives in ``app.repositories``.

- Set ``DATABASE_URL`` to the Supabase Postgres connection string and the
  Postgres + pgvector repository is used (schema: supabase/schema.sql).
- Leave it unset and the in-memory repository is used for local runs.

This module is kept as a signpost so old imports land somewhere useful.
"""
from __future__ import annotations

from .repositories import get_repository

__all__ = ["get_repository"]
