"""LLM providers: Gemini and Groq free tiers, called over plain HTTPS.

Design rules:

- No SDKs: both providers are one POST via httpx, so the backend stays
  light and the provider can be swapped with env vars only.
- Keys come from the environment (``GEMINI_API_KEY`` / ``GROQ_API_KEY``)
  and are never logged or returned.
- Every failure returns ``None``. Callers then fall back to the labelled
  extractive mode — the app never invents an "LLM" answer when the LLM
  did not actually answer.
"""
from __future__ import annotations

import os
import sys

import httpx

TIMEOUT_SECONDS = 30.0


def configured_provider() -> str | None:
    provider = (os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        return "groq"
    # If only one key exists, use that provider regardless of the label.
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return None


def _call_gemini(prompt: str) -> str | None:
    model = os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    resp = httpx.post(
        url,
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip() or None


def _call_groq(prompt: str) -> str | None:
    model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip() or None


def generate_text(prompt: str) -> str | None:
    """Run a prompt on the configured provider. None = not configured or
    the provider call failed (details go to stderr, never to the user
    as a fake answer)."""
    provider = configured_provider()
    if provider is None:
        return None
    try:
        return _call_gemini(prompt) if provider == "gemini" else _call_groq(prompt)
    except Exception as exc:  # network, quota, bad key, bad shape
        print(f"[llm] {provider} call failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


def generate_answer(question: str, context_chunks: list[dict]) -> str | None:
    """Answer a question grounded only in the retrieved chunks."""
    from .rag import build_prompt

    if configured_provider() is None:
        return None
    return generate_text(build_prompt(question, context_chunks))
