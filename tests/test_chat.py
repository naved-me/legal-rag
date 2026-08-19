import os

import pytest

import chat
from chat import _rank_context, strip_thinking


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