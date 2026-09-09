"""
FastAPI app for the RAG Knowledge Assistant v2: BM25 retrieval over a
small SaaS FAQ knowledge base, with conversational follow-ups and a
low-confidence refusal gate.

Run:
    uvicorn app.main:app --reload

Ask a question:
    curl -X POST http://127.0.0.1:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "How many API calls does the Pro plan include?"}'

Ask a follow-up in the same conversation:
    curl -X POST http://127.0.0.1:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "What about the Enterprise plan?", "session_id": "demo"}'
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import Answer, SessionStore, answer_question, build_index

app = FastAPI(
    title="RAG Knowledge Assistant v2",
    description=(
        "BM25-based retrieval-augmented Q&A agent over a small synthetic "
        "SaaS FAQ knowledge base (billing, API limits, auth, data export). "
        "Supports conversational follow-up questions and refuses to answer "
        "when retrieval confidence is too low. Demo only."
    ),
    version="0.2.0",
)

_index = build_index()
_sessions = SessionStore()


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    session_id: str | None = None


class Citation(BaseModel):
    source: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    question: str
    resolved_query: str
    answer: str
    refused: bool
    citations: list[Citation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents_indexed": len(_index.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    resolved_query = request.question
    if request.session_id:
        session = _sessions.get(request.session_id)
        resolved_query = session.resolve_query(request.question)
        session.record(request.question)

    result: Answer = answer_question(_index, resolved_query, top_k=request.top_k)
    return AskResponse(
        question=request.question,
        resolved_query=resolved_query,
        answer=result.answer,
        refused=result.refused,
        citations=[Citation(**c) for c in result.citations],
    )
