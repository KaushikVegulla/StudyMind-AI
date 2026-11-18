"""In-memory store for the scaffold.

This keeps the scaffold runnable with zero setup (no database, no keys).
It is replaced by Supabase Postgres + pgvector persistence in a later step;
the API shapes in the routers are designed to stay the same when that happens.
"""
from __future__ import annotations

import itertools

_ids = itertools.count(1)


def next_id(prefix: str) -> str:
    return f"{prefix}_{next(_ids)}"


# workspace_id -> {"id", "name", "description"}
workspaces: dict[str, dict] = {}
# document_id -> {"id", "workspace_id", "filename", "pages", "chunk_count"}
documents: dict[str, dict] = {}
# chunk_id -> {"id", "document_id", "workspace_id", "page", "index", "text"}
chunks: dict[str, dict] = {}


def seed() -> None:
    if workspaces:
        return
    for name, desc in [
        ("Data Structures", "Data Structures with Competitive Coding - Sem 3"),
        ("Databases", "Relational and Distributed Databases - Sem 3"),
        ("Cloud", "Cloud Practitioner AWS notes"),
    ]:
        wid = next_id("ws")
        workspaces[wid] = {"id": wid, "name": name, "description": desc}
