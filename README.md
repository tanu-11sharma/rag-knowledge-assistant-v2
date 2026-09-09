# RAG Knowledge Assistant v2

A retrieval-augmented Q&A agent over a small SaaS product FAQ knowledge base (billing, API rate limits, authentication, data export), with BM25 ranking, conversational follow-up support, and a low-confidence refusal gate.

## What it does

You ask a question about the product ("How many API calls does the Pro plan include?") and the assistant retrieves the most relevant passages using Okapi BM25 (a length-normalized ranking function used in real search and RAG systems), then answers with citations back to the source document. You can also ask a short follow-up in the same conversation ("What about Enterprise?") and the assistant resolves it against your previous question before retrieving. If nothing in the knowledge base is confidently relevant, it says so instead of guessing.

This is a v2 twist on a plain TF-IDF RAG demo: same core RAG shape (chunk -> index -> retrieve -> answer with citations), but with a stronger ranking algorithm, a different data domain, and two production-relevant behaviors -- conversational memory and a confidence gate -- that a plain single-turn RAG demo doesn't need.

## Why this is relevant

RAG remains one of the most common applied-AI patterns in production: ground answers in your own documents and cite sources instead of relying purely on parametric memory. Two things separate a toy RAG demo from something closer to production: (1) a ranking function that accounts for document length and term saturation rather than raw term overlap, and (2) explicit handling of "I don't know" and multi-turn context, since real users ask follow-ups and real systems shouldn't answer confidently when retrieval quality is poor. This project implements both, from the standard library, with zero external API keys.

## Project structure

```
app/
  main.py       FastAPI app exposing POST /ask and GET /health
  rag.py        BM25 index, session/follow-up resolution, confidence-gated answering
  data/         four small sample .txt documents (synthetic SaaS FAQ content)
tests/
  test_rag.py   unit tests for tokenization, BM25 ranking, refusal, and session follow-ups
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest tests/ -v
```

## Example usage

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many API calls does the Pro plan include?"}'
```

Ask a follow-up in the same conversation (pass a `session_id`):

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What about Enterprise?", "session_id": "demo"}'
```

The second call resolves to `"How many API calls does the Pro plan include? What about Enterprise?"` internally before retrieving, so the answer correctly compares against the Enterprise plan.

A question with no relevant match triggers the refusal gate instead of a low-confidence guess:

```json
{
  "refused": true,
  "answer": "I don't have enough relevant information in the knowledge base to answer that confidently. Try rephrasing, or ask about billing, API rate limits, authentication, or data export.",
  "citations": []
}
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

## Notes

- All documents in `app/data/` are synthetic sample FAQ content written for this demo -- they don't describe a real product or company.
- No fabricated metrics, uptime numbers, or usage stats are claimed anywhere in this repo.
- Session state is in-memory only (a plain dict, no persistence or eviction) -- fine for this demo, not for production.
- Follow-up resolution is a simple heuristic (short question + no keyword overlap with the prior turn), not a learned query rewriter -- a natural next step for a real system.
