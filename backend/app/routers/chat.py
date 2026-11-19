from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from fastapi import APIRouter

from ..repositories import get_repository
from ..services import llm, rag

router = APIRouter(prefix="/chat", tags=["chat"])


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    workspace_id: str | None = None
    style: Literal["concise", "standard", "exam"] = "standard"
    # HyDE (hypothetical document embeddings): opt-in, default off.
    # When on, the vector search embeds a short LLM-written
    # hypothetical answer instead of the raw question; any failure
    # falls back to normal retrieval. See services/rag.py.
    hyde: bool = False


@router.post("/ask")
def ask(body: AskIn):
    repo = get_repository()
    chunks, retrieval_mode = rag.retrieve(body.question, body.workspace_id,
                                          repo, hyde=body.hyde)
    if not chunks:
        return {
            "answer": "I could not find this in your uploaded material. "
                      "Upload notes that cover this topic, or rephrase the question.",
            "citations": [], "mode": "no_match",
            "retrieval": retrieval_mode, "style": body.style,
            "hyde": body.hyde,
        }
    # Source labels [S1], [S2] are assigned here by code, in the same
    # order the prompt shows them, so the citations returned to the
    # client always match the labels the model was told to cite.
    chunks = rag.assign_source_ids(chunks)
    citations = [rag.citation_for(c) for c in chunks]
    answer = llm.generate_answer(body.question, chunks, style=body.style)
    if answer:
        return {"answer": answer, "citations": citations, "mode": "llm",
                "retrieval": retrieval_mode, "style": body.style,
                "hyde": body.hyde}
    extractive = "From your notes: " + " ".join(c["text"] for c in chunks[:2])
    return {"answer": extractive, "citations": citations,
            "mode": "extractive_scaffold", "retrieval": retrieval_mode,
            "style": body.style, "hyde": body.hyde}
