import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from app.repositories.memory import MemoryRepository


def make_repo():
    repo = MemoryRepository()
    repo.seed_if_empty()
    return repo


def test_seed_creates_three_subjects():
    repo = make_repo()
    names = {w["name"] for w in repo.list_workspaces()}
    assert {"Data Structures", "Databases", "Cloud"} <= names


def test_document_and_chunks_roundtrip():
    repo = make_repo()
    ws = repo.list_workspaces()[0]
    doc = repo.create_document(ws["id"], "notes.pdf", 2, [
        {"page": 1, "index": 0, "text": "A B-tree keeps data sorted.",
         "embedding": [1.0, 0.0, 0.0]},
        {"page": 2, "index": 0, "text": "Hashing maps keys to buckets.",
         "embedding": [0.0, 1.0, 0.0]},
    ])
    assert doc["chunk_count"] == 2
    chunks = repo.list_chunks(ws["id"])
    assert len(chunks) == 2
    assert chunks[0]["filename"] == "notes.pdf"


def test_vector_search_ranks_nearest_first():
    repo = make_repo()
    ws = repo.list_workspaces()[0]
    repo.create_document(ws["id"], "notes.pdf", 2, [
        {"page": 1, "index": 0, "text": "tree chunk",
         "embedding": [1.0, 0.0, 0.0]},
        {"page": 2, "index": 0, "text": "hash chunk",
         "embedding": [0.0, 1.0, 0.0]},
    ])
    hits = repo.search_by_embedding([0.9, 0.1, 0.0], ws["id"], top_k=1)
    assert hits[0]["text"] == "tree chunk"


def test_delete_workspace_cascades():
    repo = make_repo()
    ws = repo.create_workspace("Temp", "")
    repo.create_document(ws["id"], "t.pdf", 1, [
        {"page": 1, "index": 0, "text": "x", "embedding": None},
    ])
    assert repo.delete_workspace(ws["id"]) is True
    assert repo.list_chunks(ws["id"]) == []
    assert repo.delete_workspace(ws["id"]) is False


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"),
                    reason="TEST_DATABASE_URL not set - Postgres test skipped")
def test_postgres_repository_roundtrip():
    from app.repositories.postgres import PostgresRepository

    repo = PostgresRepository(os.environ["TEST_DATABASE_URL"])
    repo.seed_if_empty()
    ws = repo.create_workspace("PgTest", "roundtrip")
    try:
        doc = repo.create_document(ws["id"], "pg.pdf", 1, [
            {"page": 1, "index": 0, "text": "pgvector chunk",
             "embedding": [0.1] * 384},
        ])
        assert doc["chunk_count"] == 1
        hits = repo.search_by_embedding([0.1] * 384, ws["id"], top_k=1)
        assert hits and hits[0]["text"] == "pgvector chunk"
    finally:
        repo.delete_workspace(ws["id"])
