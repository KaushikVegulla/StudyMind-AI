from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .repositories import get_repository
from .routers import chat, documents, generate, workspaces

load_dotenv()

app = FastAPI(title="StudyMind AI", version="0.2.0",
              description="RAG-based AI study assistant - answers with page citations")

origins = (os.getenv("CORS_ORIGINS") or "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(workspaces.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(generate.router, prefix="/api")


@app.get("/health")
def health():
    from .services.embeddings import embeddings_available
    from .services.llm import configured_provider

    repo = get_repository()
    return {"status": "ok", "service": "studymind-backend",
            "persistence": repo.name,
            "embeddings": "available" if embeddings_available() else "not_installed",
            "llm_provider": configured_provider() or "not_configured",
            **repo.counts()}
