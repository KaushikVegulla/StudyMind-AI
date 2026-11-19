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
        # prepare_threshold=None disables psycopg prepared statements.
        # Through Supabase's transaction pooler (port 6543) the server
        # connection behind ours is recycled between transactions, so
        # a prepared statement name can already exist on the "new"
        # connection: the first /health after a deploy failed with
        # psycopg.errors.DuplicatePreparedStatement in counts().
        # Prepared statements buy nothing at this app's query rate.
        self._conn = psycopg.connect(database_url, row_factory=dict_row,
                                     autocommit=True,
                                     prepare_threshold=None)
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
                        chunks: list[dict], extractor: str = "pypdf",
                        pages_needing_ocr: int = 0) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                "insert into documents (workspace_id, filename, pages, "
                "extractor, pages_needing_ocr) "
                "values (%s, %s, %s, %s, %s) returning id::text",
                (workspace_id, filename, pages, extractor,
                 pages_needing_ocr),
            )
            doc_id = cur.fetchone()["id"]
            # Parents first: child chunks link to the parent row id
            # created in this same call (parent-child retrieval).
            ref_to_id: dict[str, str] = {}
            ordered = sorted(chunks,
                               key=lambda c: bool(c.get("parent_ref")))
            for chunk in ordered:
                parent_id = None
                if chunk.get("parent_ref"):
                    parent_id = ref_to_id.get(chunk["parent_ref"])
                cur.execute(
                    "insert into chunks (document_id, workspace_id, page, "
                    "chunk_index, text, section, embedding, parent_id) "
                    "values (%s, %s, %s, %s, %s, %s, %s::vector, %s) "
                    "returning id::text",
                    (doc_id, workspace_id, chunk["page"], chunk["index"],
                     chunk["text"], chunk.get("section", ""),
                     _vector_literal(chunk.get("embedding")), parent_id),
                )
                if chunk.get("ref"):
                    ref_to_id[chunk["ref"]] = cur.fetchone()["id"]
        return {"id": doc_id, "workspace_id": workspace_id,
                "filename": filename, "pages": pages,
                "chunk_count": len(chunks), "extractor": extractor,
                "pages_needing_ocr": pages_needing_ocr}

    def list_documents(self, workspace_id: str | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            if workspace_id:
                cur.execute(
                    "select d.id::text, d.workspace_id::text, d.filename, "
                    "d.pages, coalesce(d.extractor, 'pypdf') as extractor, "
                    "coalesce(d.pages_needing_ocr, 0) as "
                    "pages_needing_ocr, (select count(*) from chunks c "
                    "where c.document_id = d.id) as chunk_count "
                    "from documents d where d.workspace_id = %s "
                    "order by d.created_at", (workspace_id,))
            else:
                cur.execute(
                    "select d.id::text, d.workspace_id::text, d.filename, "
                    "d.pages, coalesce(d.extractor, 'pypdf') as extractor, "
                    "coalesce(d.pages_needing_ocr, 0) as "
                    "pages_needing_ocr, (select count(*) from chunks c "
                    "where c.document_id = d.id) as chunk_count "
                    "from documents d order by d.created_at")
            return [dict(r) for r in cur.fetchall()]

    _CHUNK_COLUMNS = ("c.id::text, c.document_id::text, "
                      "c.workspace_id::text, d.filename, c.page, "
                      "c.chunk_index as index, c.text, "
                      "coalesce(c.section, '') as section, "
                      "c.parent_id::text as parent_id")

    @staticmethod
    def _chunk_row(row: dict) -> dict:
        row = dict(row)
        emb = row.get("embedding")
        if isinstance(emb, str):
            row["embedding"] = json.loads(emb)
        return row

    def list_chunks(self, workspace_id: str | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            base = (f"select {self._CHUNK_COLUMNS}, "
                    "c.embedding::text as embedding "
                    "from chunks c join documents d on d.id = c.document_id ")
            if workspace_id:
                cur.execute(base + "where c.workspace_id = %s "
                                   "order by c.page, c.chunk_index",
                            (workspace_id,))
            else:
                cur.execute(base + "order by c.page, c.chunk_index")
            return [self._chunk_row(r) for r in cur.fetchall()]

    def get_chunk(self, chunk_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"select {self._CHUNK_COLUMNS}, "
                "c.embedding::text as embedding "
                "from chunks c join documents d on d.id = c.document_id "
                "where c.id = %s", (chunk_id,))
            row = cur.fetchone()
            return self._chunk_row(row) if row else None

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
                    "coalesce(c.section, '') as section, "
                    "c.parent_id::text as parent_id, "
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
                    "coalesce(c.section, '') as section, "
                    "c.parent_id::text as parent_id, "
                    "1 - (c.embedding <=> %s::vector) as similarity "
                    "from chunks c join documents d on d.id = c.document_id "
                    "where c.embedding is not null "
                    "order by c.embedding <=> %s::vector limit %s",
                    (literal, literal, top_k),
                )
            return [dict(r) for r in cur.fetchall()]

    # -- keyword search ----------------------------------------------------
    def search_by_keyword(self, question: str,
                          workspace_id: str | None,
                          top_k: int = 20) -> list[dict]:
        """Full-text ranking with tsvector and ts_rank. The GIN index in
        supabase/schema.sql makes this fast. Used as the keyword pool
        for hybrid RRF fusion in services/rag.py."""
        with self._conn.cursor() as cur:
            if workspace_id:
                cur.execute(
                    "select c.id::text, c.document_id::text, "
                    "c.workspace_id::text, d.filename, c.page, "
                    "c.chunk_index as index, c.text, "
                    "coalesce(c.section, '') as section, "
                    "c.parent_id::text as parent_id, "
                    "ts_rank(to_tsvector('english', c.text), "
                    "plainto_tsquery('english', %s)) as keyword_score "
                    "from chunks c join documents d on d.id = c.document_id "
                    "where c.workspace_id = %s and "
                    "to_tsvector('english', c.text) @@ "
                    "plainto_tsquery('english', %s) "
                    "order by keyword_score desc limit %s",
                    (question, workspace_id, question, top_k),
                )
            else:
                cur.execute(
                    "select c.id::text, c.document_id::text, "
                    "c.workspace_id::text, d.filename, c.page, "
                    "c.chunk_index as index, c.text, "
                    "coalesce(c.section, '') as section, "
                    "c.parent_id::text as parent_id, "
                    "ts_rank(to_tsvector('english', c.text), "
                    "plainto_tsquery('english', %s)) as keyword_score "
                    "from chunks c join documents d on d.id = c.document_id "
                    "where to_tsvector('english', c.text) @@ "
                    "plainto_tsquery('english', %s) "
                    "order by keyword_score desc limit %s",
                    (question, question, top_k),
                )
            rows = []
            for r in cur.fetchall():
                row = dict(r)
                row["keyword_score"] = float(row.get("keyword_score") or 0.0)
                rows.append(row)
            return rows

    # -- flashcards ------------------------------------------------------
    _FLASHCARD_COLUMNS = (
        "id::text, workspace_id::text, document_id::text, front, back, "
        "ease_factor, interval_days, repetitions, due_at, "
        "fsrs_stability, fsrs_difficulty, last_review_at")

    @staticmethod
    def _flashcard_row(row: dict) -> dict:
        card = dict(row)
        for field in ("due_at", "last_review_at"):
            value = card.get(field)
            if value is not None and not isinstance(value, str):
                card[field] = value.isoformat()
        card["ease_factor"] = float(card["ease_factor"])
        for field in ("fsrs_stability", "fsrs_difficulty"):
            if card.get(field) is not None:
                card[field] = float(card[field])
        return card

    def save_flashcards(self, workspace_id: str,
                        document_id: str | None,
                        cards: list[dict]) -> list[dict]:
        from ..services.scheduling import new_card_state

        saved = []
        with self._conn.cursor() as cur:
            for card in cards:
                state = new_card_state()
                cur.execute(
                    "insert into flashcards (workspace_id, document_id, "
                    "front, back, ease_factor, interval_days, repetitions, "
                    "due_at) values (%s, %s, %s, %s, %s, %s, %s, %s) "
                    f"returning {self._FLASHCARD_COLUMNS}",
                    (workspace_id, document_id, card["front"], card["back"],
                     state["ease_factor"], state["interval_days"],
                     state["repetitions"], state["due_at"]),
                )
                saved.append(self._flashcard_row(cur.fetchone()))
        return saved

    def list_due_flashcards(self, workspace_id: str | None = None,
                            document_id: str | None = None,
                            now: str | None = None) -> list[dict]:
        from ..services.srs import now_utc

        cutoff = now or now_utc().isoformat()
        sql = (f"select {self._FLASHCARD_COLUMNS} from flashcards "
               "where due_at <= %s")
        params: list = [cutoff]
        if workspace_id:
            sql += " and workspace_id = %s"
            params.append(workspace_id)
        if document_id:
            sql += " and document_id = %s"
            params.append(document_id)
        sql += " order by due_at"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [self._flashcard_row(r) for r in cur.fetchall()]

    def get_flashcard(self, flashcard_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"select {self._FLASHCARD_COLUMNS} from flashcards "
                "where id = %s", (flashcard_id,))
            row = cur.fetchone()
            return self._flashcard_row(row) if row else None

    def update_flashcard_review(self, flashcard_id: str,
                                state: dict) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "update flashcards set ease_factor = %s, "
                "interval_days = %s, repetitions = %s, due_at = %s, "
                "fsrs_stability = %s, fsrs_difficulty = %s, "
                "last_review_at = %s "
                f"where id = %s returning {self._FLASHCARD_COLUMNS}",
                (state["ease_factor"], state["interval_days"],
                 state["repetitions"], state["due_at"],
                 state.get("fsrs_stability"), state.get("fsrs_difficulty"),
                 state.get("last_review_at"), flashcard_id),
            )
            row = cur.fetchone()
            return self._flashcard_row(row) if row else None

    # -- generation cache --------------------------------------------------
    def get_generated(self, workspace_id: str, document_id: str | None,
                      kind: str, count: int) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "select content from generated_items where "
                "workspace_id = %s and document_id is not distinct from %s "
                "and kind = %s and item_count = %s",
                (workspace_id, document_id, kind, count),
            )
            row = cur.fetchone()
            return dict(row["content"]) if row else None

    def save_generated(self, workspace_id: str, document_id: str | None,
                       kind: str, count: int, payload: dict) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "delete from generated_items where workspace_id = %s and "
                "document_id is not distinct from %s and kind = %s and "
                "item_count = %s",
                (workspace_id, document_id, kind, count),
            )
            cur.execute(
                "insert into generated_items (workspace_id, document_id, "
                "kind, item_count, content) values (%s, %s, %s, %s, %s)",
                (workspace_id, document_id, kind, count,
                 json.dumps(payload)),
            )
