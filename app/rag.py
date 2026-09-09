"""
RAG Knowledge Assistant v2 -- BM25 retrieval + conversational follow-ups
+ a low-confidence refusal gate.

This is a deliberate twist on the plain TF-IDF/cosine RAG demo (v1):

1. Retrieval algorithm: Okapi BM25 (a stronger, length-normalized ranking
   function widely used in real search/RAG systems) instead of raw
   TF-IDF cosine similarity. Implemented from the Python standard
   library only -- no external search engine or embedding API needed.
2. Conversational memory: a session can ask a follow-up question
   ("what about the enterprise plan?") and the assistant folds the
   previous question into the new query before retrieving, so pronouns
   and implicit references resolve against the prior turn.
3. Confidence gate: if the best BM25 score for a query falls below a
   threshold, the assistant explicitly refuses to answer instead of
   returning a low-relevance guess -- a common production RAG safety
   pattern to avoid confidently wrong answers.

Domain: a small SaaS product's FAQ knowledge base (billing, API rate
limits, authentication, data export) -- synthetic sample data only.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Below this BM25 score, the assistant refuses to answer rather than guess.
LOW_CONFIDENCE_THRESHOLD = 0.5

# BM25 hyperparameters (standard defaults).
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class Chunk:
    doc_id: str
    source: str
    text: str


class Bm25Index:
    """A small, dependency-free Okapi BM25 index over text chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._tokenized: list[list[str]] = [tokenize(c.text) for c in chunks]
        self._doc_len = [len(toks) for toks in self._tokenized]
        self._avg_doc_len = (sum(self._doc_len) / len(self._doc_len)) if chunks else 0.0

        self._doc_freq: Counter[str] = Counter()
        self._term_freq: list[Counter[str]] = []
        for toks in self._tokenized:
            tf = Counter(toks)
            self._term_freq.append(tf)
            for term in tf:
                self._doc_freq[term] += 1

        self._n = len(chunks) or 1

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        # BM25's standard IDF, floored at a small positive value so a term
        # appearing in every chunk still contributes slightly rather than
        # going negative.
        raw = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
        return max(raw, 1e-6)

    def score(self, query_terms: list[str], idx: int) -> float:
        tf = self._term_freq[idx]
        doc_len = self._doc_len[idx] or 1
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
            total += idf * (numerator / denominator)
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        scored = [(chunk, self.score(query_terms, i)) for i, chunk in enumerate(self.chunks)]
        scored = [(c, s) for c, s in scored if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for para in [p.strip() for p in text.split("\n") if p.strip()]:
            chunks.append(Chunk(doc_id=path.stem, source=path.name, text=para))
    return chunks


def build_index(data_dir: Path = DATA_DIR) -> Bm25Index:
    return Bm25Index(load_chunks(data_dir))


@dataclass
class Answer:
    query: str
    answer: str
    citations: list[dict]
    refused: bool = False


def answer_question(index: Bm25Index, query: str, top_k: int = 3) -> Answer:
    results = index.search(query, top_k=top_k)

    if not results or results[0][1] < LOW_CONFIDENCE_THRESHOLD:
        return Answer(
            query=query,
            answer=(
                "I don't have enough relevant information in the knowledge base "
                "to answer that confidently. Try rephrasing, or ask about billing, "
                "API rate limits, authentication, or data export."
            ),
            citations=[],
            refused=True,
        )

    answer_text = " ".join(chunk.text for chunk, _score in results)
    citations = [
        {"source": chunk.source, "excerpt": chunk.text, "score": round(score, 4)}
        for chunk, score in results
    ]
    return Answer(query=query, answer=answer_text, citations=citations, refused=False)


@dataclass
class Session:
    """Tracks the last question asked in a conversation, so a short
    follow-up like "what about the enterprise plan?" can be resolved
    against the prior turn before hitting the retriever."""

    history: list[str] = field(default_factory=list)

    def resolve_query(self, new_question: str) -> str:
        """Fold prior turns into a short follow-up question.

        Heuristic: if the new question is short (<= 6 tokens) and doesn't
        repeat any keyword from the previous question, we assume it's an
        elliptical follow-up and prepend the previous question's tokens
        to give the retriever more context to work with.
        """
        tokens = tokenize(new_question)
        if self.history and len(tokens) <= 6:
            prev_tokens = set(tokenize(self.history[-1]))
            overlap = prev_tokens & set(tokens)
            if not overlap:
                return f"{self.history[-1]} {new_question}"
        return new_question

    def record(self, question: str) -> None:
        self.history.append(question)


class SessionStore:
    """In-memory session store keyed by session id. Demo-only: state is
    lost on process restart, and there is no eviction policy -- fine for
    a small illustrative service, not for production use."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = defaultdict(Session)

    def get(self, session_id: str) -> Session:
        return self._sessions[session_id]
