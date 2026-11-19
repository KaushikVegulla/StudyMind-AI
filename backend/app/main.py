from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .repositories import get_repository
from .routers import chat, documents, generate, workspaces
from .services.rate_limit import RateLimiter

load_dotenv()

app = FastAPI(title="StudyMind AI", version="0.2.0",
              description="RAG-based AI study assistant - answers with page citations")

origins = (os.getenv("CORS_ORIGINS") or "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])

# -- security headers -----------------------------------------------------
# Applied to every response, including errors. The CSP is restrictive
# because this service only ever returns JSON (and API docs): nothing
# it serves should load or be framed by anything.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware: stamp the security headers on the way out."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {k.lower() for k, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name.lower().encode() not in present:
                        headers.append((name.lower().encode(),
                                        value.encode()))
            await send(message)

        await self.app(scope, receive, send_with_headers)


# -- rate limiting ----------------------------------------------------------
# Per-IP token bucket on the endpoints that spend LLM quota or CPU:
# chat and generate. The limit is env-tunable (RATE_LIMIT_PER_MINUTE,
# default 60); 0 disables limiting.
def _rate_limit_per_minute() -> int:
    try:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE") or 60)
    except ValueError:
        return 60


RATE_LIMITED_PREFIXES = ("/api/chat", "/api/generate")
rate_limiter = RateLimiter(_rate_limit_per_minute())


class RateLimitMiddleware:
    """Pure ASGI middleware: 429 with Retry-After when a client's
    bucket for the chat/generate endpoints is empty."""

    def __init__(self, app, limiter: RateLimiter = rate_limiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.limiter.limit_per_minute <= 0:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith(RATE_LIMITED_PREFIXES):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        key = client[0] if client else "unknown"
        if self.limiter.allow(key):
            await self.app(scope, receive, send)
            return
        retry_after = self.limiter.retry_after_seconds(key)
        body = json.dumps(
            {"detail": "Rate limit exceeded. Slow down and retry."}).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(retry_after).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(workspaces.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(generate.router, prefix="/api")


@app.get("/health")
def health():
    from .services.embeddings import active_embedder_name
    from .services.llm import configured_provider

    repo = get_repository()
    embedder = active_embedder_name()
    return {"status": "ok", "service": "studymind-backend",
            "persistence": repo.name,
            "embeddings": embedder if embedder else "not_installed",
            "llm_provider": configured_provider() or "not_configured",
            **repo.counts()}
