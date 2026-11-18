from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter

from ..services import llm, rag

router = APIRouter(prefix="/chat", tags=["chat"])


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    workspace_id: str | None = None


@router.post("/ask")
def ask(body: AskIn):
    chunks = rag.retrieve(body.question, body.workspace_id)
    if not chunks:
        return {
            "answer": "I could not find this in your uploaded material. "
                      "Upload notes that cover this topic, or rephrase the question.",
            "citations": [], "mode": "no_match",
        }
    citations = [rag.citation_for(c) for c in chunks]
    answer = llm.generate_answer(body.question, chunks)
    if answer:
        return {"answer": answer, "citations": citations, "mode": "llm"}
    extractive = "From your notes: " + " ".join(c["text"] for c in chunks[:2])
    return {"answer": extractive, "citations": citations, "mode": "extractive_scaffold"}
