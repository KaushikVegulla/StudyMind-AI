"""Spaced repetition scheduling for flashcards (SM-2).

A flashcard carries four scheduling fields alongside its front/back:

- ``ease_factor``  — starts at 2.5, never drops below 1.3.
- ``interval_days`` — days until the card is due again (0 = new card).
- ``repetitions``  — consecutive successful recalls.
- ``due_at``       — when the card is next due; a new card is due
  immediately, so it enters the review queue straight away.

``review`` is a pure function: given the current scheduling state, a
grade from 0 (total blackout) to 5 (perfect recall) and the current
time, it returns the next state. The classic SM-2 rules:

- grade < 3: the card is relearned — repetitions reset to 0 and it is
  due again tomorrow (interval 1 day).
- grade >= 3: repetitions grow; intervals go 1 day, then 6 days, then
  ``round(interval * ease_factor)``.
- the ease factor is adjusted after every review by
  ``0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)``, floored at 1.3.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

DEFAULT_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_card_state(at: datetime | None = None) -> dict:
    """Scheduling state for a freshly created flashcard: due now."""
    at = at or now_utc()
    return {
        "ease_factor": DEFAULT_EASE_FACTOR,
        "interval_days": 0,
        "repetitions": 0,
        "due_at": at.isoformat(),
    }


def review(state: dict, grade: int, at: datetime | None = None) -> dict:
    """Apply one SM-2 review. Returns a new state dict; ``state`` is
    not mutated. ``grade`` must be an int in [0, 5]."""
    if not isinstance(grade, int) or isinstance(grade, bool) \
            or grade < 0 or grade > 5:
        raise ValueError("grade must be an integer between 0 and 5")
    at = at or now_utc()
    ease = float(state.get("ease_factor", DEFAULT_EASE_FACTOR))
    interval = int(state.get("interval_days", 0))
    repetitions = int(state.get("repetitions", 0))

    ease = max(MIN_EASE_FACTOR,
               ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))

    if grade < 3:
        repetitions = 0
        interval = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = round(interval * ease)

    return {
        "ease_factor": round(ease, 4),
        "interval_days": interval,
        "repetitions": repetitions,
        "due_at": (at + timedelta(days=interval)).isoformat(),
    }
