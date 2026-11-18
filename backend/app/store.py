"""Deprecated scaffold store.

Persistence now lives in ``app.repositories`` (in-memory backend for local
runs, Postgres + pgvector when DATABASE_URL is set). This module is kept
only so older imports fail loudly in one place; do not build on it.
"""
from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "app.store is deprecated - use app.repositories.get_repository()"
    )
