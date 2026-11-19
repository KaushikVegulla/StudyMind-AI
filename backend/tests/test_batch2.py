import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories import get_repository
from app.services import llm, quiz_validation, rag, srs

client = TestClient(app)


def _workspace_with_doc(text="A stack is a LIFO structure. Push adds an "
                              "item and pop removes the top item."):
    repo = get_repository()
    ws = repo.create_workspace("Batch2 Test", "isolated")
    doc = repo.create_document(ws["id"], "batch2.pdf", 1, [
        {"page": 1, "index": 0, "text": text, "embedding": None},
    ])
    return repo, ws, doc


# -- SM-2 scheduling ----------------------------------------------------------

def _state(**over):
    state = {"ease_factor": 2.5, "interval_days": 0, "repetitions": 0,
             "due_at": "2026-11-20T00:00:00+00:00"}
    state.update(over)
    return state


def test_sm2_failed_grade_relearns_tomorrow():
    at = datetime(2026, 11, 20, tzinfo=timezone.utc)
    out = srs.review(_state(interval_days=6, repetitions=2), 2, at=at)
    assert out["repetitions"] == 0
    assert out["interval_days"] == 1
    assert out["due_at"] == (at + timedelta(days=1)).isoformat()


def test_sm2_progression_one_six_then_ease_scaled():
    at = datetime(2026, 11, 20, tzinfo=timezone.utc)
    first = srs.review(_state(), 4, at=at)
    assert (first["repetitions"], first["interval_days"]) == (1, 1)
    second = srs.review(first, 4, at=at)
    assert (second["repetitions"], second["interval_days"]) == (2, 6)
    third = srs.review(second, 4, at=at)
    assert third["repetitions"] == 3
    assert third["interval_days"] == round(6 * third["ease_factor"])


def test_sm2_ease_floor_and_grade_validation():
    at = datetime(2026, 11, 20, tzinfo=timezone.utc)
    out = srs.review(_state(ease_factor=1.35), 0, at=at)
    assert out["ease_factor"] >= 1.3
    with pytest.raises(ValueError):
        srs.review(_state(), 6, at=at)
    with pytest.raises(ValueError):
        srs.review(_state(), -1, at=at)


def test_sm2_new_card_is_due_immediately():
    state = srs.new_card_state()
    assert state["ease_factor"] == 2.5
    assert state["interval_days"] == 0
    assert state["repetitions"] == 0
    due = datetime.fromisoformat(state["due_at"])
    assert due <= datetime.now(timezone.utc)


# -- MCQ validation -------------------------------------------------------------

SOURCE = ["A stack is a last in first out structure. Push adds an item "
          "to the top and pop removes the top item."]

VALID = {"question": "What does push do to a stack?",
         "options": ["Adds an item to the top", "Removes the top item",
                     "Sorts the stack", "Reverses the stack"],
         "answer": "Adds an item to the top"}


def test_mcq_valid_item_normalises():
    valid, rejected = quiz_validation.validate_quiz([VALID], SOURCE)
    assert rejected == []
    assert valid[0]["correct_index"] == 0
    assert valid[0]["answer"] == "Adds an item to the top"


def test_mcq_rejects_wrong_option_count():
    bad = {**VALID, "options": VALID["options"][:3]}
    assert quiz_validation.validate_mcq(bad, SOURCE)


def test_mcq_rejects_duplicate_options():
    bad = {**VALID, "options": ["Adds an item to the top"] * 2
           + VALID["options"][2:]}
    errors = quiz_validation.validate_mcq(bad, SOURCE)
    assert any("distinct" in e for e in errors)


def test_mcq_rejects_empty_question_and_empty_option():
    assert quiz_validation.validate_mcq({**VALID, "question": "  "}, SOURCE)
    bad = {**VALID, "options": ["", *VALID["options"][1:]]}
    assert quiz_validation.validate_mcq(bad, SOURCE)


def test_mcq_rejects_correct_index_out_of_range():
    bad = {"question": VALID["question"], "options": VALID["options"],
           "correct_index": 4}
    errors = quiz_validation.validate_mcq(bad, SOURCE)
    assert any("out of range" in e for e in errors)


def test_mcq_rejects_answer_not_among_options():
    bad = {**VALID, "answer": "Deletes the stack"}
    errors = quiz_validation.validate_mcq(bad, SOURCE)
    assert any("exactly one option" in e for e in errors)


def test_mcq_rejects_ungrounded_item():
    alien = {"question": "Which planet has the largest ring system?",
             "options": ["Saturn", "Mars", "Venus", "Mercury"],
             "answer": "Saturn"}
    errors = quiz_validation.validate_mcq(alien, SOURCE)
    assert any("grounded" in e for e in errors)


def test_mcq_accepts_correct_index_form():
    item = {"question": VALID["question"], "options": VALID["options"],
            "correct_index": 1}
    valid, rejected = quiz_validation.validate_quiz([item], SOURCE)
    assert rejected == []
    assert valid[0]["answer"] == "Removes the top item"


# -- generation cache -----------------------------------------------------------

def test_generation_cache_calls_llm_once(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    calls = {"n": 0}

    def counting_generate(prompt):
        calls["n"] += 1
        return "A cached summary of stacks and LIFO order."

    monkeypatch.setattr(llm, "generate_text", counting_generate)
    _repo, ws, _doc = _workspace_with_doc()

    first = client.post("/api/generate/summary",
                        json={"workspace_id": ws["id"]}).json()
    second = client.post("/api/generate/summary",
                         json={"workspace_id": ws["id"]}).json()
    assert first["mode"] == "llm" and first["cached"] is False
    assert second["cached"] is True
    assert second["summary"] == first["summary"]
    assert calls["n"] == 1

    third = client.post("/api/generate/summary",
                        json={"workspace_id": ws["id"], "force": True}).json()
    assert third["cached"] is False
    assert calls["n"] == 2


# -- validated quiz flow ----------------------------------------------------------

QUIZ_JSON = ('[{"question": "What does push do to a stack?", '
             '"options": ["Adds an item to the top", "Removes the top item", '
             '"Sorts the stack", "Reverses the stack"], '
             '"answer": "Adds an item to the top"}]')
INVALID_JSON = ('[{"question": "Bad item", "options": ["only", "two"], '
                '"answer": "only"}]')


def test_quiz_retries_once_then_uses_valid_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    outputs = iter([INVALID_JSON, QUIZ_JSON])
    monkeypatch.setattr(llm, "generate_text", lambda prompt: next(outputs))
    _repo, ws, _doc = _workspace_with_doc()

    body = client.post("/api/generate/quiz",
                       json={"workspace_id": ws["id"], "count": 1}).json()
    assert body["mode"] == "llm"
    assert body["quiz"][0]["correct_index"] == 0


def test_quiz_falls_back_to_scaffold_when_always_invalid(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "generate_text", lambda prompt: INVALID_JSON)
    _repo, ws, _doc = _workspace_with_doc(
        "A queue is a FIFO structure used for scheduling tasks fairly "
        "across many waiting processes in an operating system.")

    body = client.post("/api/generate/quiz",
                       json={"workspace_id": ws["id"], "count": 1}).json()
    assert body["mode"] == "extractive_scaffold"


# -- flashcard review queue -----------------------------------------------------

def test_flashcard_review_flow(monkeypatch):
    # This flow asserts SM-2 scheduling semantics through the API, so
    # pin the SM-2 engine: the default engine is FSRS (services/
    # scheduling.py), whose first-review intervals differ by design.
    monkeypatch.setenv("SRS_ENGINE", "sm2")
    _repo, ws, _doc = _workspace_with_doc()
    body = client.post("/api/generate/flashcards",
                       json={"workspace_id": ws["id"], "count": 2}).json()
    cards = body["flashcards"]
    assert cards and cards[0]["ease_factor"] == 2.5
    assert cards[0]["repetitions"] == 0

    due = client.get("/api/generate/flashcards/due",
                     params={"workspace_id": ws["id"]}).json()
    assert due["count"] == len(cards)

    reviewed = client.post(
        "/api/generate/flashcards/review",
        json={"flashcard_id": cards[0]["id"], "grade": 4}).json()
    card = reviewed["flashcard"]
    assert card["repetitions"] == 1
    assert card["interval_days"] == 1

    due_after = client.get("/api/generate/flashcards/due",
                           params={"workspace_id": ws["id"]}).json()
    assert all(c["id"] != cards[0]["id"] for c in due_after["flashcards"])

    missing = client.post("/api/generate/flashcards/review",
                          json={"flashcard_id": "card_nope", "grade": 3})
    assert missing.status_code == 404
    bad_grade = client.post("/api/generate/flashcards/review",
                            json={"flashcard_id": cards[0]["id"], "grade": 9})
    assert bad_grade.status_code == 422


# -- answer styles ----------------------------------------------------------------

def test_style_instructions_reach_prompt_builder():
    chunks = [{"filename": "dsa.pdf", "page": 1, "index": 0,
               "text": "A stack is LIFO."}]
    concise = rag.build_prompt("What is a stack?", chunks, style="concise")
    exam = rag.build_prompt("What is a stack?", chunks, style="exam")
    standard = rag.build_prompt("What is a stack?", chunks)
    assert "concise" in concise
    assert "exam" in exam
    assert "concise" not in standard and "Style: exam" not in standard
    # Grounding rules are present in every style.
    for prompt in (concise, exam, standard):
        assert "Never invent" in prompt and "[S1]" in prompt


def test_generate_answer_passes_style_to_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    seen = {}

    def fake_generate(prompt):
        seen["p"] = prompt
        return "ok"

    monkeypatch.setattr(llm, "generate_text", fake_generate)
    chunks = [{"filename": "dsa.pdf", "page": 1, "index": 0,
               "text": "A stack is LIFO."}]
    assert llm.generate_answer("What is a stack?", chunks,
                               style="exam") == "ok"
    assert "Style: exam" in seen["p"]


def test_chat_accepts_style(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "generate_text", lambda prompt: "Styled answer")
    _repo, ws, _doc = _workspace_with_doc()
    body = client.post("/api/chat/ask",
                       json={"question": "What is a stack?",
                             "workspace_id": ws["id"],
                             "style": "concise"}).json()
    assert body["style"] == "concise"
    bad = client.post("/api/chat/ask",
                      json={"question": "What is a stack?",
                            "workspace_id": ws["id"], "style": "poetic"})
    assert bad.status_code == 422
