"""Postgres + pgvector repository (Supabase).

Activated by setting DATABASE_URL to the Supabase Postgres connection
string. The schema is the one in supabase/schema.sql; ``ensure_schema``
applies it idempotently on first use so a fresh project needs no manual
step beyond creating the project.

Embeddings are stored as pgvector ``vector(384)`` and searched with the
cosine-distance operator ``<=>`` — the same ranking the in-memory backend
approximates, so behaviour matches between local and deployed runs.
"""
from __future__ import annotations

import json
import pathlib

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[3] / "supabase" / "schema.sql"


def _vector_literal(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    return "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"


class PostgresRepository:
    name = "postgres"

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "psycopg is not installed - see backend/requirements.txt"
            ) from exc
        self._conn = psycopg.connect(database_url, row_factory=dict_row,
                                     autocommit=True)
        self.ensure_schema()

    # -- schema --------------------------------------------------------------
    def ensure_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._conn.cursor() as cur:
            cur.execute(sql)

    def seed_if_empty(self) -> None:
        if self.list_workspaces():
            return
        for name, desc in [
            ("Data Structures", "Data Structures with Competitive Coding - Sem 3"),
            ("Databases", "Relational and Distributed Databases - Sem 3"),
            ("Cloud", "Cloud Practitioner AWS notes"),
        ]:
            self.create_workspace(name, desc)

    def counts(self) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                "select (select count(*) from workspaces) as workspaces, "
                "(select count(*) from documents) as documents, "
                "(select count(*) from chunks) as chunks"
            )
            return dict(cur.fetchone())

    # -- workspaces ------------------------------------------------------------
    def list_workspaces(self) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("select id::text, name, description from workspaces "
                        "order by created_at")
            return [dict(r) for r in cur.fetchall()]

    def create_workspace(self, name: str, description: str) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                "insert into workspaces (name, description) values (%s, %s) "
                "returning id::text, name, description",
                (name, description),
            )
            return dict(cur.fetchone())

    def get_workspace(self, workspace_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute("select id::text, name, description from workspaces "
                        "where id = %s", (workspace_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def delete_workspace(self, workspace_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("delete from workspaces where id = %s", (workspace_id,))
            return cur.rowcount > 0

    # -- documents and chunks ----------------------------------------------------
    def create_document(self, workspace_id: str, filename: str, pages: int,
                        chunks: list[dict]) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                "insert into documents (workspace_id, filename, pages) "
                "values (%s, %s, %s) returning id::text",
                (workspace_id, filename, pages),
            )
            doc_id = cur.fetchone()["id"]
            for chunk in chunks:
                cur.execute(
                    "insert into chunks (document_id, workspace_id, page, "
                    "chunk_index, text, embedding) values "
                    "(%s, %s, %s, %s, %s, %s::vector)",
                    (doc_id, workspace_id, chunk["page"], chunk["index"],
                     chunk["text"], _vector_literal(chunk.get("embedding"))),
                )
        return {"id": doc_id, "workspace_id": workspace_id,
                "filename": filename, "pages": pages,
                "chunk_count": len(chunks)}

    def list_documents(self, workspace_id: str | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            if workspace_id:
                cur.execute(
                    "select d.id::text, d.workspace_id::text, d.filename, "
                    "d.pages, (select count(*) from chunks c "
                    "where c.document_id = d.id) as chunk_count "
                    "from documents d where d.workspace_id = %s "
                    "order by d.created_at", (workspace_id,))
            else:
                cur.execute(
                    "select d.id::text, d.workspace_id::text, d.filename, "
                    "d.pages, (select count(*) from chunks c "
                    "where c.document_id = d.id) as chunk_count "
                    "from documents d order by d.created_at")
            return [dict(r) for r in cur.fetchall()]

    def list_chunks(self, workspace_id: str | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            base = ("select c.id::text, c.document_id::text, "
                    "c.workspace_id::text, d.filename, c.page, "
                    "c.chunk_index as index, c.text, "
                    "c.embedding::text as embedding "
                    "from chunks c join documents d on d.id = c.document_id ")
            if workspace_id:
                cur.execute(base + "where c.workspace_id = %s "
                                   "order by c.page, c.chunk_index",
                            (workspace_id,))
            else:
                cur.execute(base + "order by c.page, c.chunk_index")
            rows = []
            for r in cur.fetchall():
                row = dict(r)
                emb = row.get("embedding")
                row["embedding"] = (json.loads(emb) if isinstance(emb, str)
                                    else emb)
                rows.append(row)
            return rows

    # -- vector search -------------------------------------------------------------
    def search_by_embedding(self, embedding: list[float],
                            workspace_id: str | None,
                            top_k: int = 4) -> list[dict]:
        literal = _vector_literal(embedding)
        with self._conn.cursor() as cur:
            if workspace_id:
                cur.execute(
                    "select c.id::text, c.document_id::text, "
                    "c.workspace_id::text, d.filename, c.page, "
                    "c.chunk_index as index, c.text, "
                    "1 - (c.embedding <=> %s::vector) as similarity "
                    "from chunks c join documents d on d.id = c.document_id "
                    "where c.workspace_id = %s and c.embedding is not null "
                    "order by c.embedding <=> %s::vector limit %s",
                    (literal, workspace_id, literal, top_k),
                )
            else:
                cur.execute(
                    "select c.id::text, c.document_id::text, "
                    "c.workspace_id::text, d.filename, c.page, "
                    "c.chunk_index as index, c.text, "
                    "1 - (c.embedding <=> %s::vector) as similarity "
                    "from chunks c join documents d on d.id = c.document_id "
                    "where c.embedding is not null "
                    "order by c.embedding <=> %s::vector limit %s",
                    (literal, literal, top_k),
                )
            return [dict(r) for r in cur.fetchall()]
