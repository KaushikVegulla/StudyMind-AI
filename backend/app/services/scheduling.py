"""Review-engine selection for flashcard scheduling.

Two engines implement the same stored-state contract:

- ``fsrs`` — FSRS-5 (services/fsrs.py). The default.
- ``sm2``  — the classic SM-2 (services/srs.py), kept unchanged.

The engine is chosen by the ``SRS_ENGINE`` environment variable
(``fsrs`` by default, ``sm2`` for SM-2) and read on every call, so a
deployment switches engines with an env change and tests can pin an
engine per test. Cards store the fields of both engines; the active
engine fills its own and leaves the other's untouched.

Grades at the API boundary stay on the SM-2 era 0-5 scale (the
frontend's Again/Hard/Good/Easy buttons send 1/3/4/5). Under FSRS the
grade is mapped onto FSRS ratings: 0-1 Again, 2-3 Hard, 4 Good,
5 Easy.
"""
from __future__ import annotations

import os
from datetime import datetime

from . import fsrs, srs

ENGINES = ("fsrs", "sm2")


def active_engine() -> str:
    value = (os.getenv("SRS_ENGINE") or "fsrs").strip().lower()
    return value if value in ENGINES else "fsrs"


def grade_to_fsrs_rating(grade: int) -> int:
    """Map an API grade (0-5) to an FSRS rating (1-4)."""
    if grade <= 1:
        return fsrs.RATING_AGAIN
    if grade <= 3:
        return fsrs.RATING_HARD
    if grade == 4:
        return fsrs.RATING_GOOD
    return fsrs.RATING_EASY


def new_card_state(at: datetime | None = None) -> dict:
    """Scheduling state for a freshly created flashcard: due now,
    with the fields of both engines at their defaults."""
    state = srs.new_card_state(at)
    state.update({
        "fsrs_stability": None,
        "fsrs_difficulty": None,
        "last_review_at": None,
    })
    return state


def apply_review(card: dict, grade: int, at: datetime | None = None) -> dict:
    """Apply one review under the active engine and return the full
    scheduling state the repositories persist. ``grade`` is the API
    grade (0-5); validation matches the active engine's rules."""
    at = at or srs.now_utc()
    if active_engine() == "sm2":
        result = srs.review(card, grade, at=at)
        return {
            "ease_factor": result["ease_factor"],
            "interval_days": result["interval_days"],
            "repetitions": result["repetitions"],
            "due_at": result["due_at"],
            "fsrs_stability": card.get("fsrs_stability"),
            "fsrs_difficulty": card.get("fsrs_difficulty"),
            "last_review_at": at.isoformat(),
        }
    result = fsrs.review(card, grade_to_fsrs_rating(grade), at=at)
    return {
        "ease_factor": float(card.get("ease_factor", srs.DEFAULT_EASE_FACTOR)),
        "interval_days": result["interval_days"],
        "repetitions": result["repetitions"],
        "due_at": result["due_at"],
        "fsrs_stability": result["fsrs_stability"],
        "fsrs_difficulty": result["fsrs_difficulty"],
        "last_review_at": result["last_review_at"],
    }
