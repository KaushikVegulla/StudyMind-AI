"""LLM provider interface (Gemini / Groq free tier).

No provider is configured in the scaffold: when no API key is present the
chat endpoint falls back to extractive answers (it quotes the retrieved
chunks) instead of pretending an LLM answered. Set LLM_PROVIDER plus the
matching API key in .env to enable fluent generation in a later step.
"""
from __future__ import annotations

import os


def configured_provider() -> str | None:
    provider = (os.getenv("LLM_PROVIDER") or "").lower()
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        return "groq"
    return None


def generate_answer(question: str, context_chunks: list[dict]) -> str | None:
    """Return an LLM answer grounded only in context_chunks, or None when no
    LLM is configured. Provider calls are wired up in the LLM step."""
    if configured_provider() is None:
        return None
    return None  # provider integration lands with the LLM step
