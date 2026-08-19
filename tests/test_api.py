import os
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage

import api

REAL_KEY = "test-key-1"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    with TestClient(api.app) as c:
        yield c


@pytest.fixture()
def canned_answer(monkeypatch):
    def fake_query_rag(question, history=None, api_mode=False):
        if history is None:
            history = []
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="Mocked answer"))
        return {
            "answer": "Mocked answer with [Page 1].",
            "context": [],
            "scores": {"max_relevance": 1.5, "rerank_min_score": 1.0},
        }

    monkeypatch.setattr(api, "query_rag", fake_query_rag)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"


def test_static_site_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MotorLegal" in resp.text or "chat" in resp.text.lower()


def test_ask_requires_auth(client, canned_answer):
    resp = client.post("/ask", json={"question": "What is a fine?"})
    assert resp.status_code == 401


def test_ask_wrong_key(client, canned_answer):
    resp = client.post(
        "/ask",
        json={"question": "What is a fine?"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


def test_ask_with_key(client, canned_answer):
    resp = client.post(
        "/ask",
        json={"question": "What is a fine?"},
        headers={"Authorization": f"Bearer {REAL_KEY}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked answer with [Page 1]."
    assert body["session_id"]


def test_ask_request_id_echo(client, canned_answer):
    resp = client.post(
        "/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {REAL_KEY}", "X-Request-ID": "test-abc-123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == "test-abc-123"


def test_ask_blank_question(client):
    resp = client.post(
        "/ask",
        json={"question": "   "},
        headers={"Authorization": f"Bearer {REAL_KEY}"},
    )
    assert resp.status_code == 422


def test_ask_missing_question(client):
    resp = client.post(
        "/ask",
        json={},
        headers={"Authorization": f"Bearer {REAL_KEY}"},
    )
    assert resp.status_code == 422


def test_ask_overlong_question(client):
    resp = client.post(
        "/ask",
        json={"question": "x" * 501},
        headers={"Authorization": f"Bearer {REAL_KEY}"},
    )
    assert resp.status_code == 422


def test_session_history_persists(client, monkeypatch):
    seen = []

    def fake_query_rag(question, history=None, api_mode=False):
        if history is None:
            history = []
        seen.append(len(history))
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="ok"))
        return {"answer": "ok", "context": [], "scores": {"max_relevance": 1.0}}

    monkeypatch.setattr(api, "query_rag", fake_query_rag)
    headers = {"Authorization": f"Bearer {REAL_KEY}"}

    r1 = client.post("/ask", json={"question": "first"}, headers=headers).json()
    sid = r1["session_id"]
    r2 = client.post("/ask", json={"question": "second", "session_id": sid}, headers=headers)
    assert r2.status_code == 200
    assert seen == [0, 2]