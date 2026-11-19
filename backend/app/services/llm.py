"""LLM providers: Gemini and Groq free tiers, called over plain HTTPS.

Design rules:

- No SDKs: both providers are one POST via httpx, so the backend stays
  light and the provider can be swapped with env vars only.
- Keys come from the environment (``GEMINI_API_KEY`` / ``GROQ_API_KEY``)
  and are never logged or returned.
- Every failure returns ``None``. Callers then fall back to the labelled
  extractive mode — the app never invents an "LLM" answer when the LLM
  did not actually answer.

Quota armour (free tiers are the scarcest resource this app has):

- **Provider chain.** Providers are tried in order — the configured
  ``LLM_PROVIDER`` first, then the other provider when it also has a
  key — before giving up to the extractive fallback. With default
  settings that is Gemini, then Groq, then extractive.
- **Retry with backoff.** A provider call that fails with a retryable
  error (HTTP 429, HTTP 5xx, timeouts, transport errors) is retried
  up to ``LLM_MAX_ATTEMPTS`` times (default 3) with exponential
  backoff plus jitter (base ``LLM_RETRY_BASE_SECONDS``, default 0.5).
- **Cooldown.** After ``LLM_COOLDOWN_THRESHOLD`` consecutive failed
  calls (default 3), a provider is skipped entirely for
  ``LLM_COOLDOWN_SECONDS`` (default 60) so a dead or rate-limited
  provider stops adding latency to every request; the next provider
  in the chain is used instead. A success resets the counter.

Sleeping and the clock are module-level indirections (``_sleep``,
``_now``) so tests run instantly with fakes.
"""
from __future__ import annotations

import os
import random
import ssl
import sys
import time

import httpx

TIMEOUT_SECONDS = 30.0

_sleep = time.sleep
_now = time.monotonic


def _jitter(delay: float) -> float:
    return random.uniform(0, delay * 0.25)


def _max_attempts() -> int:
    try:
        return max(1, int(os.getenv("LLM_MAX_ATTEMPTS") or 3))
    except ValueError:
        return 3


def _retry_base_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("LLM_RETRY_BASE_SECONDS") or 0.5))
    except ValueError:
        return 0.5


def _cooldown_threshold() -> int:
    try:
        return max(1, int(os.getenv("LLM_COOLDOWN_THRESHOLD") or 3))
    except ValueError:
        return 3


def _cooldown_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("LLM_COOLDOWN_SECONDS") or 60))
    except ValueError:
        return 60.0


# Per-provider consecutive-failure counts and cooldown deadlines.
_provider_state: dict[str, dict] = {
    "gemini": {"failures": 0, "cooldown_until": 0.0},
    "groq": {"failures": 0, "cooldown_until": 0.0},
}


def _reset_provider_state() -> None:
    """Test helper: clear failure counts and cooldowns."""
    for state in _provider_state.values():
        state["failures"] = 0
        state["cooldown_until"] = 0.0


def _in_cooldown(provider: str) -> bool:
    return _now() < _provider_state[provider]["cooldown_until"]


def _record_success(provider: str) -> None:
    _provider_state[provider]["failures"] = 0


def _record_failure(provider: str) -> None:
    state = _provider_state[provider]
    state["failures"] += 1
    if state["failures"] >= _cooldown_threshold():
        state["cooldown_until"] = _now() + _cooldown_seconds()
        state["failures"] = 0
        print(f"[llm] {provider} entered cooldown for "
              f"{_cooldown_seconds():.0f}s after repeated failures",
              file=sys.stderr)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _call_with_retry(provider: str, fn) -> str | None:
    """Run one provider call with retry/backoff. Returns the call's
    result; re-raises the last exception when attempts are exhausted
    or the error is not retryable."""
    attempts = _max_attempts()
    for attempt in range(attempts):
        try:
            result = fn()
        except Exception as exc:
            if attempt < attempts - 1 and _is_retryable(exc):
                delay = _retry_base_seconds() * (2 ** attempt)
                _sleep(delay + _jitter(delay))
                continue
            _record_failure(provider)
            raise
        if result is not None:
            _record_success(provider)
        return result
    return None  # pragma: no cover - loop always returns or raises


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


def _provider_has_key(provider: str) -> bool:
    return bool(os.getenv("GEMINI_API_KEY") if provider == "gemini"
                else os.getenv("GROQ_API_KEY"))


def provider_chain() -> list[str]:
    """Providers to try, in order: the configured provider first,
    then the other one when it has a key. Empty when nothing is
    configured."""
    first = configured_provider()
    if first is None:
        return []
    chain = [first]
    for other in ("gemini", "groq"):
        if other != first and _provider_has_key(other):
            chain.append(other)
    return chain


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


_CALLERS = {"gemini": _call_gemini, "groq": _call_groq}


def generate_text(prompt: str) -> str | None:
    """Run a prompt on the provider chain (configured provider, then
    the other, each with retry/backoff and cooldown — see the module
    docstring). None = not configured or every provider failed
    (details go to stderr, never to the user as a fake answer)."""
    for provider in provider_chain():
        if _in_cooldown(provider):
            print(f"[llm] {provider} skipped (cooldown)", file=sys.stderr)
            continue
        try:
            result = _call_with_retry(provider,
                                      lambda: _CALLERS[provider](prompt))
        except Exception as exc:  # network, quota, bad key, bad shape
            print(f"[llm] {provider} call failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if result is not None:
            return result
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
