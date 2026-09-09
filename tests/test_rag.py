from pathlib import Path

from app.rag import (
    Session,
    SessionStore,
    answer_question,
    build_index,
    load_chunks,
    tokenize,
)

DATA_DIR = Path(__file__).parent.parent / "app" / "data"


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Rate Limits, Explained!") == ["rate", "limits", "explained"]


def test_load_chunks_finds_all_sample_documents():
    chunks = load_chunks(DATA_DIR)
    sources = {c.source for c in chunks}
    assert sources == {
        "billing.txt",
        "api_limits.txt",
        "authentication.txt",
        "data_export.txt",
    }
    assert len(chunks) > 0


def test_bm25_search_ranks_relevant_chunk_first_for_billing_question():
    index = build_index(DATA_DIR)
    results = index.search("How much does the Pro plan cost per month?", top_k=3)
    assert results, "expected at least one match"
    top_chunk, top_score = results[0]
    assert top_chunk.source == "billing.txt"
    assert top_score > 0


def test_bm25_search_ranks_relevant_chunk_first_for_rate_limit_question():
    index = build_index(DATA_DIR)
    results = index.search("What happens when I exceed the API rate limit?", top_k=3)
    assert results
    assert results[0][0].source == "api_limits.txt"


def test_answer_question_includes_citations_for_confident_match():
    index = build_index(DATA_DIR)
    result = answer_question(index, "How do I rotate my API key?")
    assert not result.refused
    assert result.citations, "expected at least one citation"
    assert any(c["source"] == "authentication.txt" for c in result.citations)


def test_answer_question_refuses_on_low_confidence():
    index = build_index(DATA_DIR)
    result = answer_question(index, "zzzz qqqq nonexistent gibberish term")
    assert result.refused is True
    assert result.citations == []
    assert "don't have enough" in result.answer.lower()


def test_session_resolves_short_followup_against_prior_question():
    session = Session()
    session.record("How many API calls does the Pro plan include?")
    resolved = session.resolve_query("What about Enterprise?")
    assert "Pro plan" in resolved
    assert "Enterprise" in resolved


def test_session_leaves_self_contained_question_unchanged():
    session = Session()
    session.record("How many API calls does the Pro plan include?")
    long_question = "How do I export all of my account data as CSV?"
    resolved = session.resolve_query(long_question)
    assert resolved == long_question


def test_session_store_isolates_sessions_by_id():
    store = SessionStore()
    a = store.get("a")
    b = store.get("b")
    a.record("question in session a")
    assert b.history == []
    assert store.get("a") is a
