"""Spaced repetition scheduling for flashcards (FSRS-5).

FSRS is the scheduling algorithm used by modern Anki. Instead of
SM-2's ease factor it tracks two per-card quantities:

- ``fsrs_stability``   — days for the card's retrievability to fall
  from 100% to 90%. Grows with every successful recall.
- ``fsrs_difficulty``  — 1 (easy) to 10 (hard). Adapts toward the
  grades the learner actually gives.

and one derived value, retrievability — the probability of recall
after ``t`` days, on the FSRS power-law forgetting curve
``R = (1 + (19/81) * t / S) ** -0.5`` (so R is exactly 0.9 when
t equals the stability).

The weights below are the published FSRS-5 default parameters
(open-spaced-repetition, as shipped in py-fsrs / Anki 23.10+); they
are not tuned to this app's users — with review logs the weights
could be optimised later, which the stored state fields allow.

Reviews use FSRS ratings: Again=1, Hard=2, Good=3, Easy=4. The API
keeps the 0-5 grade scale of the SM-2 era; services/scheduling.py
maps grades onto ratings, so this module only ever sees 1-4.

``review`` is a pure function over day-granularity state, like the
SM-2 implementation in services/srs.py: given the stored state, a
rating and the current time it returns the next state, including the
shared fields the repositories persist (``interval_days``, ``due_at``,
``repetitions``, ``last_review_at``).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# Published FSRS-5 default weights (open-spaced-repetition).
WEIGHTS = (
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604,
    0.0046, 1.54575, 0.1192, 1.01925, 1.9395, 0.11, 0.29605,
    2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
)
DECAY = -0.5
FACTOR = 19 / 81  # chosen so R(t = stability) = 0.9 exactly
DESIRED_RETENTION = 0.9
MIN_DIFFICULTY, MAX_DIFFICULTY = 1.0, 10.0

RATING_AGAIN, RATING_HARD, RATING_GOOD, RATING_EASY = 1, 2, 3, 4


def _clamp_difficulty(value: float) -> float:
    return min(MAX_DIFFICULTY, max(MIN_DIFFICULTY, value))


def initial_stability(rating: int) -> float:
    """Stability after the first review: S0(G) = w[G-1]."""
    return WEIGHTS[rating - 1]


def initial_difficulty(rating: int) -> float:
    """Difficulty after the first review: D0(G) = w4 - e^(w5*(G-1)) + 1."""
    return _clamp_difficulty(
        WEIGHTS[4] - math.exp(WEIGHTS[5] * (rating - 1)) + 1)


def retrievability(elapsed_days: float, stability: float) -> float:
    """Probability of recall after ``elapsed_days`` at ``stability``."""
    if stability <= 0:
        return 0.0
    return (1 + FACTOR * elapsed_days / stability) ** DECAY


def next_interval(stability: float,
                  desired_retention: float = DESIRED_RETENTION) -> int:
    """Days until retrievability falls to the desired retention.
    At the default 0.9 retention this is round(stability); at least 1."""
    days = (stability / FACTOR) * (desired_retention ** (1 / DECAY) - 1)
    return max(1, round(days))


def _next_difficulty(difficulty: float, rating: int) -> float:
    delta = -WEIGHTS[6] * (rating - 3)
    damped = difficulty + delta * (10 - difficulty) / 9
    # Mean reversion toward the difficulty of an "Easy" first grade.
    reverted = WEIGHTS[7] * initial_difficulty(RATING_EASY) \
        + (1 - WEIGHTS[7]) * damped
    return _clamp_difficulty(reverted)


def _next_recall_stability(difficulty: float, stability: float,
                           recall_probability: float, rating: int) -> float:
    hard_factor = WEIGHTS[15] if rating == RATING_HARD else 1.0
    easy_factor = WEIGHTS[16] if rating == RATING_EASY else 1.0
    gain = (math.exp(WEIGHTS[8])
            * (11 - difficulty)
            * stability ** (-WEIGHTS[9])
            * (math.exp(WEIGHTS[10] * (1 - recall_probability)) - 1)
            * hard_factor * easy_factor)
    return stability * (1 + gain)


def _next_forget_stability(difficulty: float, stability: float,
                           recall_probability: float) -> float:
    return (WEIGHTS[11]
            * difficulty ** (-WEIGHTS[12])
            * ((stability + 1) ** WEIGHTS[13] - 1)
            * math.exp(WEIGHTS[14] * (1 - recall_probability)))


def _as_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def review(state: dict, rating: int, at: datetime | None = None) -> dict:
    """Apply one FSRS review. Returns a new state dict; ``state`` is
    not mutated. ``rating`` must be an int in [1, 4] (Again..Easy)."""
    if not isinstance(rating, int) or isinstance(rating, bool) \
            or rating < 1 or rating > 4:
        raise ValueError("rating must be an integer between 1 and 4")
    at = at or datetime.now(timezone.utc)
    stability = state.get("fsrs_stability")
    difficulty = state.get("fsrs_difficulty")
    repetitions = int(state.get("repetitions", 0))

    if stability is None or difficulty is None:
        # First review of a new card.
        stability = initial_stability(rating)
        difficulty = initial_difficulty(rating)
    else:
        last = _as_datetime(state.get("last_review_at")) or at
        elapsed_days = max(0.0, (at - last).total_seconds() / 86400)
        recall_probability = retrievability(elapsed_days, float(stability))
        difficulty = _next_difficulty(float(difficulty), rating)
        if rating == RATING_AGAIN:
            stability = _next_forget_stability(
                difficulty, float(stability), recall_probability)
        else:
            stability = _next_recall_stability(
                difficulty, float(stability), recall_probability, rating)

    repetitions = 0 if rating == RATING_AGAIN else repetitions + 1
    interval = next_interval(stability)
    return {
        "fsrs_stability": round(float(stability), 6),
        "fsrs_difficulty": round(float(difficulty), 6),
        "interval_days": interval,
        "repetitions": repetitions,
        "due_at": (at + timedelta(days=interval)).isoformat(),
        "last_review_at": at.isoformat(),
    }
