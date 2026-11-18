"""Quiz / flashcard / summary generation.

Scaffold mode is extractive and labelled as such: summaries quote the
opening chunks, flashcards and quiz questions are built from key sentences.
Fluent LLM generation replaces these builders in the LLM step, with the
same response shapes.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..repositories import get_repository

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateIn(BaseModel):
    workspace_id: str
    count: int = 5


def _workspace_chunks(workspace_id: str) -> list[dict]:
    repo = get_repository()
    if not repo.get_workspace(workspace_id):
        raise HTTPException(404, "Workspace not found")
    chunks = repo.list_chunks(workspace_id)
    if not chunks:
        raise HTTPException(400, "No material uploaded in this workspace yet")
    return sorted(chunks, key=lambda c: (c["page"], c["index"]))


def _sentences(chunks: list[dict]) -> list[str]:
    text = " ".join(c["text"] for c in chunks)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]


@router.post("/summary")
def summary(body: GenerateIn):
    chunks = _workspace_chunks(body.workspace_id)
    text = " ".join(c["text"] for c in chunks[:3])
    return {"summary": text[:1200], "mode": "extractive_scaffold",
            "sources": sorted({c["page"] for c in chunks[:3]})}


@router.post("/flashcards")
def flashcards(body: GenerateIn):
    cards = []
    for s in _sentences(_workspace_chunks(body.workspace_id))[: body.count]:
        cards.append({"front": "Explain: " + s[:80] + "...", "back": s})
    return {"flashcards": cards, "mode": "extractive_scaffold"}


@router.post("/quiz")
def quiz(body: GenerateIn):
    questions = []
    for s in _sentences(_workspace_chunks(body.workspace_id))[: body.count]:
        questions.append({"question": "Which statement from your notes is correct?",
                          "options": [s], "answer": s,
                          "note": "Scaffold quiz - MCQ distractors arrive with LLM generation"})
    return {"quiz": questions, "mode": "extractive_scaffold"}
