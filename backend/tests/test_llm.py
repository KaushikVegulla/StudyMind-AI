import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import httpx
import pytest

from app.services import llm


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY",
                "GEMINI_MODEL", "GROQ_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_no_key_means_no_llm(monkeypatch):
    assert llm.configured_provider() is None
    assert llm.generate_text("anything") is None
    assert llm.generate_answer("q?", [{"text": "chunk", "page": 1,
                                      "document_id": "d"}]) is None


def test_gemini_answer_parsing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def fake_post(url, headers=None, json=None, timeout=None):
        assert "generativelanguage.googleapis.com" in url
        assert headers["x-goog-api-key"] == "test-key-not-real"
        return FakeResponse({"candidates": [{"content": {
            "parts": [{"text": "A B-tree keeps data sorted [notes.pdf, p.1]"}]}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    answer = llm.generate_answer("What is a B-tree?",
                                 [{"text": "A B-tree keeps data sorted.",
                                   "page": 1, "document_id": "doc_1",
                                   "filename": "notes.pdf"}])
    assert answer and "B-tree keeps data sorted" in answer


def test_groq_answer_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def fake_post(url, headers=None, json=None, timeout=None):
        assert "api.groq.com" in url
        return FakeResponse({"choices": [{"message": {
            "content": "Groq answer"}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert llm.generate_text("prompt") == "Groq answer"


def test_provider_failure_returns_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(llm.httpx, "post", boom)
    assert llm.generate_text("prompt") is None


def test_generate_endpoints_use_llm_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        llm, "generate_text",
        lambda prompt: '```json\n[{"front": "What is a B-tree?", '
                       '"back": "A self-balancing search tree"}]\n```')

    from fastapi.testclient import TestClient
    from app.main import app
    from app.repositories import get_repository

    client = TestClient(app)
    repo = get_repository()
    ws = repo.list_workspaces()[0]
    repo.create_document(ws["id"], "gen.pdf", 1, [
        {"page": 1, "index": 0,
         "text": "A B-tree is a self-balancing search tree used in databases.",
         "embedding": None},
    ])
    r = client.post("/api/generate/flashcards",
                    json={"workspace_id": ws["id"], "count": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "llm"
    assert body["flashcards"][0]["front"] == "What is a B-tree?"
