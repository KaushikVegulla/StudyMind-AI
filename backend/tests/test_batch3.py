import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import io
import types
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.repositories import get_repository
from app.repositories.memory import MemoryRepository
from app.services import embeddings, fsrs, llm, pdf_parser, rag, scheduling
from app.services.chunking import child_chunks_for, chunk_page_text, with_children
from app.services.rate_limit import RateLimiter

client = TestClient(app)


# -- embeddings runtime chain -------------------------------------------------

class _FakeFastEmbed:
    calls = []

    def __init__(self, model_name=None):
        pass

    def embed(self, texts):
        type(self).calls.append(list(texts))
        return [[0.1] * embeddings.EMBEDDING_DIM for _ in texts]


class _FakeSentenceTransformer:
    calls = []

    def __init__(self, model_name):
        pass

    def encode(self, texts, normalize_embeddings=True):
        type(self).calls.append(list(texts))
        return [[0.2] * embeddings.EMBEDDING_DIM for _ in texts]


@pytest.fixture
def fake_runtimes(monkeypatch):
    """Install fake fastembed / sentence-transformers modules and
    reset the embeddings module's caches. Returns helpers to remove
    one runtime again (sys.modules[name] = None makes the import
    raise ImportError, like a machine without the package)."""
    embeddings._models.clear()
    embeddings._load_failed.clear()
    _FakeFastEmbed.calls = []
    _FakeSentenceTransformer.calls = []
    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = _FakeFastEmbed
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "fastembed", fe)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.delenv("EMBEDDINGS_BACKEND", raising=False)
    yield fe, st
    embeddings._models.clear()
    embeddings._load_failed.clear()


def test_fastembed_is_tried_first(fake_runtimes):
    assert embeddings.active_embedder_name() == "fastembed"
    vectors = embeddings.embed_texts(["hello notes"])
    assert len(vectors[0]) == embeddings.EMBEDDING_DIM == 384
    assert _FakeFastEmbed.calls and not _FakeSentenceTransformer.calls


def test_sentence_transformers_is_the_fallback(fake_runtimes, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert embeddings.active_embedder_name() == "sentence-transformers"
    vectors = embeddings.embed_texts(["hello notes"])
    assert len(vectors[0]) == 384
    assert _FakeSentenceTransformer.calls


def test_no_runtime_means_keyword_mode(fake_runtimes, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert embeddings.active_embedder_name() is None
    assert embeddings.embeddings_available() is False
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["hello"])


def test_failed_fastembed_load_falls_through_to_sentence_transformers(
        fake_runtimes, monkeypatch):
    class Boom:
        def __init__(self, model_name=None):
            raise RuntimeError("model download failed")

    fake_runtimes[0].TextEmbedding = Boom
    vectors = embeddings.embed_texts(["hello"])
    assert vectors[0][0] == 0.2  # served by sentence-transformers
    assert embeddings.active_embedder_name() == "sentence-transformers"


def test_query_prefix_applies_on_both_runtimes(fake_runtimes, monkeypatch):
    embeddings.embed_query("what is a stack?")
    assert _FakeFastEmbed.calls[0][0] == \
        embeddings.QUERY_PREFIX + "what is a stack?"
    embeddings._models.clear()
    embeddings._load_failed.clear()
    monkeypatch.setitem(sys.modules, "fastembed", None)
    embeddings.embed_query("what is a queue?")
    assert _FakeSentenceTransformer.calls[0][0] == \
        embeddings.QUERY_PREFIX + "what is a queue?"


def test_embeddings_backend_env_can_disable_or_force(fake_runtimes,
                                                     monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "none")
    assert embeddings.embeddings_available() is False
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "sentence-transformers")
    assert embeddings.active_embedder_name() == "sentence-transformers"


def test_health_names_the_active_embedder(monkeypatch):
    # conftest sets EMBEDDINGS_BACKEND=none for the suite.
    body = client.get("/health").json()
    assert body["embeddings"] == "not_installed"


# -- parent-child retrieval -----------------------------------------------------

PARENT_TEXT = (
    "Alpha bravo introduces the topic with background material that "
    "sets up everything else on this page of the lecture notes. "
    "Charlie delta describes the distinctive mechanism in the middle "
    "of the passage with its own vocabulary and examples. "
    "Echo foxtrot closes the section with a summary and a warning "
    "about the common examination mistake students make here."
)


def _repo_with_parent_child():
    repo = MemoryRepository()
    ws = repo.create_workspace("PC Test", "isolated")
    parents = chunk_page_text(PARENT_TEXT, page=1, subject="PC Test",
                              document="pc.pdf")
    assert len(parents) == 1
    doc = repo.create_document(ws["id"], "pc.pdf", 1, with_children(parents))
    return repo, ws, doc


def test_children_are_stored_linked_to_parent():
    repo, ws, _doc = _repo_with_parent_child()
    chunks = repo.list_chunks(ws["id"])
    parents = [c for c in chunks if not c.get("parent_id")]
    children = [c for c in chunks if c.get("parent_id")]
    assert len(parents) == 1 and len(children) >= 2
    assert all(c["parent_id"] == parents[0]["id"] for c in children)
    assert all(c["page"] == 1 for c in children)


def test_child_match_returns_parent_context_but_child_citation():
    repo, ws, _doc = _repo_with_parent_child()
    chunks, _mode = rag.retrieve("charlie delta mechanism", ws["id"], repo)
    assert chunks, "expected the child chunk to be retrieved"
    top = chunks[0]
    assert top.get("parent_id"), "top hit should be a child chunk"
    assert "Charlie delta" in top["text"]
    # The prompt receives the parent's full text as context...
    prompt = rag.build_prompt("charlie delta mechanism", chunks)
    assert "Alpha bravo" in prompt and "Echo foxtrot" in prompt
    # ...while the citation still points at the child's own location
    # and quotes the child's text, not the parent's.
    citation = rag.citation_for(rag.assign_source_ids(chunks)[0])
    assert citation["page"] == 1
    assert citation["text"] == top["text"][:300]
    assert "Alpha bravo" not in citation["text"]


def test_parent_child_flag_off_searches_parents(monkeypatch):
    monkeypatch.setenv("PARENT_CHILD_RETRIEVAL", "off")
    repo = MemoryRepository()
    ws = repo.create_workspace("PC Off", "isolated")
    parents = chunk_page_text(PARENT_TEXT, page=1, document="pc.pdf")
    repo.create_document(ws["id"], "pc.pdf", 1, parents)
    chunks, _mode = rag.retrieve("charlie delta mechanism", ws["id"], repo)
    assert chunks and not chunks[0].get("parent_id")
    assert "context_text" not in chunks[0]


def test_searchable_only_drops_parents_with_children():
    chunks = [{"id": "p1", "parent_id": None},
              {"id": "c1", "parent_id": "p1"},
              {"id": "p2", "parent_id": None}]
    kept = rag.searchable_only(chunks)
    assert [c["id"] for c in kept] == ["c1", "p2"]


# -- HyDE -----------------------------------------------------------------------

def _hyde_repo():
    repo = MemoryRepository()
    ws = repo.create_workspace("HyDE Test", "isolated")
    repo.create_document(ws["id"], "devices.pdf", 1, [
        {"page": 1, "index": 0, "text": "Quantum tunnelling diode behaviour",
         "embedding": [1.0] + [0.0] * 383},
        {"page": 1, "index": 1, "text": "Stack LIFO ordering basics",
         "embedding": [0.0, 1.0] + [0.0] * 382},
    ])
    return repo, ws


def _patch_vector_search(monkeypatch, seen):
    monkeypatch.setattr(embeddings, "embeddings_available", lambda: True)

    def fake_embed_query(text):
        seen.append(text)
        if "hypothetical" in text:
            return [1.0] + [0.0] * 383
        return [0.0, 1.0] + [0.0] * 382

    monkeypatch.setattr(embeddings, "embed_query", fake_embed_query)


def test_hyde_embeds_the_hypothetical_answer(monkeypatch):
    repo, ws = _hyde_repo()
    seen = []
    _patch_vector_search(monkeypatch, seen)
    monkeypatch.setattr(llm, "configured_provider", lambda: "gemini")
    monkeypatch.setattr(llm, "generate_text",
                        lambda prompt: "A hypothetical explanation of the device")
    chunks, _mode = rag.retrieve("how does it work",
                                 ws["id"], repo, hyde=True)
    assert seen and "hypothetical" in seen[0]
    assert chunks[0]["text"].startswith("Quantum tunnelling")


def test_hyde_off_embeds_the_raw_question(monkeypatch):
    repo, ws = _hyde_repo()
    seen = []
    _patch_vector_search(monkeypatch, seen)
    monkeypatch.setattr(llm, "configured_provider", lambda: "gemini")
    monkeypatch.setattr(llm, "generate_text",
                        lambda prompt: "A hypothetical explanation")
    chunks, _mode = rag.retrieve("how does it work",
                                 ws["id"], repo, hyde=False)
    assert seen == ["how does it work"]
    assert chunks[0]["text"].startswith("Stack LIFO")


def test_hyde_failure_falls_back_to_normal_retrieval(monkeypatch):
    repo, ws = _hyde_repo()
    seen = []
    _patch_vector_search(monkeypatch, seen)
    monkeypatch.setattr(llm, "configured_provider", lambda: "gemini")

    def boom(prompt):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "generate_text", boom)
    chunks, _mode = rag.retrieve("how does it work",
                                 ws["id"], repo, hyde=True)
    assert seen == ["how does it work"]
    assert chunks and chunks[0]["text"].startswith("Stack LIFO")


def test_chat_accepts_hyde_flag():
    body = client.post("/api/chat/ask",
                       json={"question": "What is a splay tree rotation?",
                             "hyde": True}).json()
    assert body["hyde"] is True
    body = client.post("/api/chat/ask",
                       json={"question": "What is a splay tree rotation?"}
                       ).json()
    assert body["hyde"] is False


# -- FSRS scheduling --------------------------------------------------------------

AT = datetime(2026, 11, 20, tzinfo=timezone.utc)


def _new_fsrs_state():
    return {"fsrs_stability": None, "fsrs_difficulty": None,
            "repetitions": 0, "last_review_at": None}


def test_fsrs_first_review_stability_strictly_ordered():
    stabilities = [fsrs.review(_new_fsrs_state(), rating, at=AT)
                   ["fsrs_stability"] for rating in (1, 2, 3, 4)]
    assert stabilities == sorted(stabilities)
    assert len(set(stabilities)) == 4  # Again < Hard < Good < Easy
    intervals = [fsrs.review(_new_fsrs_state(), rating, at=AT)
                 ["interval_days"] for rating in (1, 2, 3, 4)]
    # Day granularity ties Again/Hard at 1 day; the rest strictly grow.
    assert intervals[0] == intervals[1] == 1
    assert intervals[1] < intervals[2] < intervals[3]


def test_fsrs_stability_grows_with_successful_recalls():
    state = fsrs.review(_new_fsrs_state(), 3, at=AT)  # Good
    first = state["fsrs_stability"]
    state = fsrs.review(state, 3, at=AT + timedelta(days=state["interval_days"]))
    assert state["fsrs_stability"] > first
    second = state["fsrs_stability"]
    state = fsrs.review(state, 4, at=AT + timedelta(days=state["interval_days"]))
    assert state["fsrs_stability"] > second


def test_fsrs_difficulty_adapts_to_grades():
    easy = fsrs.review(_new_fsrs_state(), 4, at=AT)
    again = fsrs.review(_new_fsrs_state(), 1, at=AT)
    assert easy["fsrs_difficulty"] < again["fsrs_difficulty"]
    # A lapse resets repetitions and shrinks stability hard.
    grown = fsrs.review(_new_fsrs_state(), 3, at=AT)
    grown = fsrs.review(grown, 3, at=AT + timedelta(days=3))
    lapsed = fsrs.review(grown, 1, at=AT + timedelta(days=6))
    assert lapsed["repetitions"] == 0
    assert lapsed["fsrs_stability"] < grown["fsrs_stability"]


def test_fsrs_retrievability_curve_anchors():
    assert fsrs.retrievability(0, 5.0) == 1.0
    assert fsrs.retrievability(5.0, 5.0) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        fsrs.review(_new_fsrs_state(), 0, at=AT)
    with pytest.raises(ValueError):
        fsrs.review(_new_fsrs_state(), 5, at=AT)


def test_scheduling_grade_mapping_and_engine_selection(monkeypatch):
    assert [scheduling.grade_to_fsrs_rating(g) for g in range(6)] == \
        [1, 1, 2, 2, 3, 4]
    monkeypatch.delenv("SRS_ENGINE", raising=False)
    assert scheduling.active_engine() == "fsrs"
    monkeypatch.setenv("SRS_ENGINE", "sm2")
    assert scheduling.active_engine() == "sm2"
    state = scheduling.new_card_state(at=AT)
    assert state["fsrs_stability"] is None
    assert state["ease_factor"] == 2.5


def test_fsrs_review_flow_through_api(monkeypatch):
    monkeypatch.delenv("SRS_ENGINE", raising=False)  # fsrs is the default
    repo = get_repository()
    ws = repo.create_workspace("FSRS API", "isolated")
    repo.create_document(ws["id"], "fsrs.pdf", 1, [
        {"page": 1, "index": 0,
         "text": "A stack is a LIFO structure. Push adds an item and "
                 "pop removes the top item.", "embedding": None}])
    body = client.post("/api/generate/flashcards",
                       json={"workspace_id": ws["id"], "count": 1}).json()
    card = body["flashcards"][0]
    assert card["fsrs_stability"] is None

    reviewed = client.post(
        "/api/generate/flashcards/review",
        json={"flashcard_id": card["id"], "grade": 4}).json()
    updated = reviewed["flashcard"]
    assert updated["fsrs_stability"] == pytest.approx(fsrs.WEIGHTS[2])
    assert updated["interval_days"] == 3  # round(S0(Good)) at 0.9 retention
    assert updated["repetitions"] == 1
    due = client.get("/api/generate/flashcards/due",
                     params={"workspace_id": ws["id"]}).json()
    assert all(c["id"] != card["id"] for c in due["flashcards"])


# -- ingestion: extractors and scanned pages --------------------------------------

def _blank_pdf() -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _text_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import (DecodedStreamObject, DictionaryObject,
                               NameObject)
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 18 Tf 40 700 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = stream
    font = DictionaryObject()
    font.update({NameObject("/Type"): NameObject("/Font"),
                 NameObject("/Subtype"): NameObject("/Type1"),
                 NameObject("/BaseFont"): NameObject("/Helvetica")})
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_scanned_page_detection_counts_textless_pages():
    pages = [{"page": 1, "text": ""},
             {"page": 2, "text": "  \n "},
             {"page": 3, "text": "Real lecture text about stacks and queues."}]
    assert pdf_parser.count_pages_needing_ocr(pages) == 2
    extracted = pdf_parser.extract_document(_blank_pdf())
    assert extracted["extractor"] == "pypdf"
    assert pdf_parser.count_pages_needing_ocr(extracted["pages"]) == 1


def test_docling_used_when_importable_and_recorded(monkeypatch):
    monkeypatch.setattr(pdf_parser, "docling_available", lambda: True)
    monkeypatch.setattr(pdf_parser, "_extract_with_docling",
                        lambda data: [{"page": 1, "text": "docling text"}])
    extracted = pdf_parser.extract_document(b"%PDF-fake")
    assert extracted["extractor"] == "docling"
    assert extracted["pages"][0]["text"] == "docling text"


def test_docling_failure_falls_back_to_pypdf(monkeypatch):
    monkeypatch.setattr(pdf_parser, "docling_available", lambda: True)

    def boom(data):
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(pdf_parser, "_extract_with_docling", boom)
    extracted = pdf_parser.extract_document(_text_pdf("Stacks are LIFO."))
    assert extracted["extractor"] == "pypdf"
    assert "Stacks are LIFO" in extracted["pages"][0]["text"]


def test_upload_reports_scanned_pages_and_extractor():
    repo = get_repository()
    ws = repo.create_workspace("Upload OCR", "isolated")
    r = client.post("/api/documents/upload",
                    params={"workspace_id": ws["id"]},
                    files={"file": ("scan.pdf", _blank_pdf(),
                                    "application/pdf")})
    assert r.status_code == 201
    body = r.json()
    assert body["extractor"] == "pypdf"
    assert body["pages_needing_ocr"] == 1
    assert "scanned" in body["warning"]


def test_upload_stores_parent_and_child_chunks():
    repo = get_repository()
    ws = repo.create_workspace("Upload PC", "isolated")
    r = client.post(
        "/api/documents/upload", params={"workspace_id": ws["id"]},
        files={"file": ("notes.pdf",
                        _text_pdf("A stack is a LIFO structure for items."),
                        "application/pdf")})
    assert r.status_code == 201
    assert r.json()["pages_needing_ocr"] == 0
    assert r.json()["chunk_count"] == 2  # one parent + its one child


# -- LLM quota armour -------------------------------------------------------------

def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid")
    return httpx.HTTPStatusError(f"HTTP {status}",
                                 response=httpx.Response(status,
                                                         request=request),
                                 request=request)


@pytest.fixture
def llm_armour(monkeypatch):
    llm._reset_provider_state()
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    monkeypatch.setattr(llm, "_jitter", lambda delay: 0.0)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    yield
    llm._reset_provider_state()


def test_retry_succeeds_after_429(llm_armour, monkeypatch):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _status_error(429)
        return "recovered"

    monkeypatch.setitem(llm._CALLERS, "gemini", flaky)
    assert llm.generate_text("prompt") == "recovered"
    assert calls["n"] == 3


def test_non_retryable_error_is_not_retried(llm_armour, monkeypatch):
    calls = {"n": 0}

    def bad_request(prompt):
        calls["n"] += 1
        raise _status_error(400)

    monkeypatch.setitem(llm._CALLERS, "gemini", bad_request)
    assert llm.generate_text("prompt") is None
    assert calls["n"] == 1


def test_cooldown_skips_provider_and_chain_falls_through(llm_armour,
                                                         monkeypatch):
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_COOLDOWN_THRESHOLD", "2")
    monkeypatch.setenv("LLM_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    gemini_calls = {"n": 0}

    def down(prompt):
        gemini_calls["n"] += 1
        raise httpx.ConnectError("no network")

    monkeypatch.setitem(llm._CALLERS, "gemini", down)
    monkeypatch.setitem(llm._CALLERS, "groq", lambda prompt: "groq answer")

    assert llm.generate_text("p") == "groq answer"   # failure 1
    assert llm.generate_text("p") == "groq answer"   # failure 2 -> cooldown
    assert gemini_calls["n"] == 2
    assert llm.generate_text("p") == "groq answer"   # gemini skipped
    assert gemini_calls["n"] == 2


def test_cooldown_expires(llm_armour, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(llm, "_now", lambda: clock["t"])
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_COOLDOWN_THRESHOLD", "1")
    monkeypatch.setenv("LLM_COOLDOWN_SECONDS", "60")
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("no network")
        return "back"

    monkeypatch.setitem(llm._CALLERS, "gemini", flaky)
    assert llm.generate_text("p") is None       # fails -> cooldown
    assert llm.generate_text("p") is None       # skipped, still cooling
    assert calls["n"] == 1
    clock["t"] += 61
    assert llm.generate_text("p") == "back"     # cooldown over
    assert calls["n"] == 2


def test_provider_chain_order(llm_armour, monkeypatch):
    assert llm.provider_chain() == ["gemini"]
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    assert llm.provider_chain() == ["gemini", "groq"]
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm.provider_chain() == ["groq", "gemini"]


# -- security ---------------------------------------------------------------------

def test_security_headers_on_responses():
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'none'" in r.headers["content-security-policy"]


def test_upload_rejects_non_pdf_content():
    repo = get_repository()
    ws = repo.create_workspace("Upload Validation", "isolated")
    not_pdf = client.post(
        "/api/documents/upload", params={"workspace_id": ws["id"]},
        files={"file": ("fake.pdf", b"this is not a pdf at all",
                        "application/pdf")})
    assert not_pdf.status_code == 400
    wrong_ext = client.post(
        "/api/documents/upload", params={"workspace_id": ws["id"]},
        files={"file": ("notes.txt", _text_pdf("Real PDF, wrong name."),
                        "application/pdf")})
    assert wrong_ext.status_code == 400


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0001")  # about 100 bytes
    repo = get_repository()
    ws = repo.create_workspace("Upload Size", "isolated")
    r = client.post("/api/documents/upload",
                    params={"workspace_id": ws["id"]},
                    files={"file": ("big.pdf", _text_pdf("Some real text."),
                                    "application/pdf")})
    assert r.status_code == 413


def test_rate_limit_trips_with_retry_after():
    from app.main import RateLimitMiddleware

    small = FastAPI()
    small.add_middleware(RateLimitMiddleware, limiter=RateLimiter(3))

    @small.get("/api/chat/ping")
    def ping():
        return {"ok": True}

    limited = TestClient(small)
    for _ in range(3):
        assert limited.get("/api/chat/ping").status_code == 200
    r = limited.get("/api/chat/ping")
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    # Other paths are not limited.
    assert limited.get("/other").status_code == 404


def test_rate_limiter_refills_over_time():
    clock = {"t": 0.0}
    limiter = RateLimiter(2, now=lambda: clock["t"])
    assert limiter.allow("ip") and limiter.allow("ip")
    assert not limiter.allow("ip")
    clock["t"] += 31  # half a minute refills one token at 2/min
    assert limiter.allow("ip")


# -- Postgres repository: pooler-safe connection -----------------------------------

def test_postgres_connect_disables_prepared_statements(monkeypatch):
    """Regression test for the live DuplicatePreparedStatement 500s:
    psycopg prepared statements clash with Supabase's transaction
    pooler, so the repository must connect with
    prepare_threshold=None."""
    import app.repositories.postgres as pg_module

    recorded = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args, **kwargs):
            return None

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")

    def fake_connect(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        return FakeConn()

    fake_psycopg.connect = fake_connect
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    fake_psycopg.rows = fake_rows
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    repo = pg_module.PostgresRepository("postgresql://user:pass@host/db")
    assert recorded["prepare_threshold"] is None
    assert repo.name == "postgres"
