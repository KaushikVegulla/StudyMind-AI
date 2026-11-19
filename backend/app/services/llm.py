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
import ssl
import sys

import httpx

TIMEOUT_SECONDS = 30.0


def _post(url: str, headers: dict, payload: dict,
          timeout: float = TIMEOUT_SECONDS) -> httpx.Response:
    """POST JSON with deployment-proof networking.

    - ``trust_env=False``: httpx crashes on some platforms' NO_PROXY values
      (bracketed IPv6 entries) — a backend should not depend on them anyway.
    - An explicit HTTPS_PROXY is honoured when the host sets one.
    - TLS verification uses the system CA store, not httpx's bundled one,
      so corporate/egress proxies with their own CA keep working.
    """
    kwargs: dict = {
        "headers": headers,
        "json": payload,
        "timeout": timeout,
        "trust_env": False,
        "verify": ssl.create_default_context(),
    }
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.post(url, **kwargs)


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


def _gemini_models() -> list[str]:
    """Models to try, in order. Free-tier availability shifts: older models
    get retired for new keys (gemini-2.0/2.5-flash returned 404 for a key
    created in Nov 2026) and flash models regularly answer 503 under load,
    so one hard-coded model is brittle. Verified working for new keys on
    18 Nov 2026: gemini-3.5-flash."""
    primary = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
    fallbacks = ["gemini-3.8-flash", "gemini-flash-latest"]
    return [primary] + [m for m in fallbacks if m != primary]


def _call_gemini(prompt: str) -> str | None:
    last_exc: Exception | None = None
    for model in _gemini_models():
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        try:
            resp = _post(
                url,
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                payload={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=30.0,  # per-model cap: worst case ~90s over 3 models
            )
            resp.raise_for_status()
        except Exception as exc:  # 404 retired model / 503 demand / network
            last_exc = exc
            continue
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip() or None
    if last_exc:
        raise last_exc
    return None


def _call_groq(prompt: str) -> str | None:
    model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    resp = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        payload={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
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


def generate_answer(question: str, context_chunks: list[dict],
                    style: str = "standard") -> str | None:
    """Answer a question grounded only in the retrieved chunks.
    ``style`` is one of rag.ANSWER_STYLES and only changes the prompt's
    style instruction; grounding and citations are unchanged."""
    from .rag import build_prompt

    if configured_provider() is None:
        return None
    return generate_text(build_prompt(question, context_chunks, style=style))
