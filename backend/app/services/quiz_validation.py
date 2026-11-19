"""Validation for generated multiple-choice quiz questions.

An LLM quiz item is only shown to the student when it passes every
check in ``validate_mcq``:

- the question text is non-empty;
- there are exactly 4 options, all non-empty and distinct
  (compared case-insensitively, ignoring surrounding whitespace);
- there is exactly one correct answer: either ``correct_index`` (an
  int in range) or ``answer`` (the exact text of one of the options);
- grounding: the question and the correct option together share
  meaningful keywords with the source material the quiz was generated
  from, so an item the model invented from thin air is rejected.

``validate_quiz`` splits a list of raw items into (valid, rejected):
valid items are normalised to always carry both ``answer`` (the
correct option text) and ``correct_index``; rejected items keep their
list of error strings so callers and tests can see why they failed.
"""
from __future__ import annotations

from .rag import _terms

MIN_GROUNDING_OVERLAP = 2


def _grounded(question: str, correct_option: str,
              source_terms: set[str]) -> bool:
    combined = _terms(question) | _terms(correct_option)
    if not combined:
        return False
    shared = combined & source_terms
    # Short items legitimately carry few terms; one shared keyword is
    # enough there. Longer items must show real overlap.
    needed = MIN_GROUNDING_OVERLAP if len(combined) > 2 else 1
    return len(shared) >= needed


def validate_mcq(item, source_texts: list[str]) -> list[str]:
    """Return a list of problems with one raw MCQ item; empty = valid."""
    if not isinstance(item, dict):
        return ["item is not an object"]
    errors: list[str] = []

    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append("question is empty")

    options = item.get("options")
    if not isinstance(options, list) or len(options) != 4:
        errors.append("options must be a list of exactly 4")
        options = []
    else:
        cleaned = [o.strip() if isinstance(o, str) else "" for o in options]
        if any(not o for o in cleaned):
            errors.append("options must all be non-empty strings")
        if len({o.lower() for o in cleaned}) != len(cleaned):
            errors.append("options must be distinct")

    correct_index = None
    if "correct_index" in item and item.get("correct_index") is not None:
        idx = item.get("correct_index")
        if not isinstance(idx, int) or isinstance(idx, bool) \
                or not 0 <= idx < len(options):
            errors.append("correct_index out of range")
        else:
            correct_index = idx
    answer = item.get("answer")
    if correct_index is None:
        if not isinstance(answer, str) or not answer.strip():
            errors.append("no correct answer given")
        elif options:
            matches = [i for i, o in enumerate(options)
                       if isinstance(o, str) and o.strip() == answer.strip()]
            if len(matches) != 1:
                errors.append("answer must match exactly one option")
            else:
                correct_index = matches[0]
    elif isinstance(answer, str) and options and correct_index is not None:
        if answer.strip() != options[correct_index].strip():
            errors.append("answer text disagrees with correct_index")

    if not errors and correct_index is not None:
        source_terms: set[str] = set()
        for text in source_texts:
            source_terms |= _terms(text)
        if not _grounded(str(question), str(options[correct_index]),
                         source_terms):
            errors.append("question is not grounded in the source material")
    return errors


def normalise_mcq(item: dict) -> dict:
    """Return the item with both ``answer`` and ``correct_index`` set.
    Call only on items that passed ``validate_mcq``."""
    options = [o.strip() for o in item["options"]]
    if item.get("correct_index") is not None:
        idx = item["correct_index"]
    else:
        idx = [o.strip() for o in item["options"]].index(item["answer"].strip())
    return {"question": item["question"].strip(), "options": options,
            "answer": options[idx], "correct_index": idx}


def validate_quiz(items, source_texts: list[str]) -> tuple[list[dict], list[dict]]:
    """Split raw items into (valid normalised items, rejected items
    with their error lists)."""
    valid: list[dict] = []
    rejected: list[dict] = []
    if not isinstance(items, list):
        return valid, [{"item": items, "errors": ["quiz output is not a list"]}]
    for item in items:
        errors = validate_mcq(item, source_texts)
        if errors:
            rejected.append({"item": item, "errors": errors})
        else:
            valid.append(normalise_mcq(item))
    return valid, rejected
