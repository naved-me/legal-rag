import os

import pytest

import chat
from chat import _rank_context, strip_thinking
from langchain_core.documents import Document


def _pinecone_ready() -> bool:
    if not os.getenv("PINECONE_API_KEY"):
        return False
    try:
        chat._pc().Index(chat.PINECONE_INDEX_NAME).describe_index_stats()
        return True
    except Exception:
        return False


def test_strip_thinking_marker():
    assert strip_thinking("thinking...\nresponse\nThe answer.") == "The answer."


def test_strip_thinking_preamble_line_kept():
    assert strip_thinking("thinking about it\nActually the fine is Rs. 1000.") == "Actually the fine is Rs. 1000."


def test_strip_thinking_pure_reasoning():
    assert strip_thinking("thinking") == ""


def test_strip_thinking_plain_answer_untouched():
    assert strip_thinking("The maximum fine is Rs. 1000.") == "The maximum fine is Rs. 1000."


@pytest.mark.skipif(
    not _pinecone_ready(),
    reason="Pinecone key/index not reachable; run ingest.py first",
)
def test_retrieval_returns_documents():
    docs, max_score = _rank_context("What is the maximum fine for driving without a license?")
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(d.metadata.get("page") for d in docs)


def _doc(page):
    return Document(page_content="drive a motor vehicle in contravention of speed limits",
                    metadata={"page": page, "chunk_id": f"chunk-{page}"})


def test_query_rag_refines_on_refusal(monkeypatch):
    calls = []

    def fake_rank(q):
        calls.append(q)
        if q == "whats the ticket price for speeding?":
            return [], -8.0
        return [_doc(88)], 1.7

    monkeypatch.setattr(chat, "_rank_context", fake_rank)
    monkeypatch.setattr(chat, "_refine_query",
                        lambda q: "what is the maximum fine for exceeding the speed limit?")
    monkeypatch.setattr(chat, "_answer", lambda q, docs, history: "answer")

    res = chat.query_rag("whats the ticket price for speeding?", api_mode=True)
    assert calls == ["whats the ticket price for speeding?",
                     "what is the maximum fine for exceeding the speed limit?"]
    assert res["answer"] == "answer"


def test_query_rag_no_refine_when_hit(monkeypatch):
    calls = []

    def fake_rank(q):
        calls.append(q)
        return [_doc(88)], 3.0

    monkeypatch.setattr(chat, "_rank_context", fake_rank)
    monkeypatch.setattr(chat, "_answer", lambda q, docs, history: "answer")

    res = chat.query_rag("what is the maximum fine?", api_mode=True)
    assert calls == ["what is the maximum fine?"]
    assert res["answer"] == "answer"


def test_query_rag_refine_noop_still_refuses(monkeypatch):
    def fake_rank(q):
        return [], -5.0

    monkeypatch.setattr(chat, "_rank_context", fake_rank)
    monkeypatch.setattr(chat, "_refine_query", lambda q: q)  # unchanged -> no retry

    res = chat.query_rag("whats up", api_mode=True)
    assert "not find information" in res["answer"]


def test_invoke_fallback_swaps_model(monkeypatch):
    calls = []

    def fake_invoke(chain, inputs, retries=2, backoff=10.0):
        calls.append(chain)
        if len(calls) == 1:
            raise RuntimeError("primary boom")
        return "ok"

    monkeypatch.setattr(chat, "_invoke", fake_invoke)

    def build(model):
        return f"chain:{model}"

    res = chat._invoke_fallback(build, {})
    assert res == "ok"
    assert calls == [f"chain:{chat.LLM_MODEL}", f"chain:{chat.FALLBACK_MODEL}"]


def test_invoke_fallback_reraises_primary(monkeypatch):
    def fake_invoke(chain, inputs, retries=2, backoff=10.0):
        raise ValueError("primary")

    monkeypatch.setattr(chat, "_invoke", fake_invoke)

    with pytest.raises(ValueError):
        chat._invoke_fallback(lambda m: m, {})


def test_format_context_respects_word_budget(monkeypatch):
    monkeypatch.setattr(chat, "CONTEXT_MAX_WORDS", 10)
    docs = [
        _doc(1),
        Document(page_content="one two three four five six seven eight nine ten eleven",
                 metadata={"page": 2, "chunk_id": "c2"}),
    ]
    out = chat._format_context(docs)
    assert "Page 1" in out
    # second chunk present but trimmed to the remaining 9 words
    assert "eleven" not in out