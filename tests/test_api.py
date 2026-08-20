import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage

import api


@pytest.fixture()
def client():
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


def test_ask_returns_answer(client, canned_answer):
    resp = client.post("/ask", json={"question": "What is a fine?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked answer with [Page 1]."
    assert body["session_id"]


def test_ask_request_id_echo(client, canned_answer):
    resp = client.post(
        "/ask",
        json={"question": "hi"},
        headers={"X-Request-ID": "test-abc-123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == "test-abc-123"


def test_ask_blank_question(client):
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_ask_missing_question(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_overlong_question(client):
    resp = client.post("/ask", json={"question": "x" * 1001})
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

    r1 = client.post("/ask", json={"question": "first"}).json()
    sid = r1["session_id"]
    r2 = client.post("/ask", json={"question": "second", "session_id": sid})
    assert r2.status_code == 200
    assert seen == [0, 2]


def test_sessions_crud(client, canned_answer):
    created = client.post("/sessions").json()
    sid = created["session_id"]
    assert sid

    listed = client.get("/sessions").json()
    assert any(s["session_id"] == sid for s in listed)

    client.post("/ask", json={"question": "first", "session_id": sid})

    detail = client.get(f"/sessions/{sid}")
    assert detail.status_code == 200
    msgs = detail.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "human"
    assert msgs[0]["content"] == "first"

    listed = client.get("/sessions").json()
    entry = next(s for s in listed if s["session_id"] == sid)
    assert entry["title"] == "first"

    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_sessions_delete_missing(client):
    assert client.delete("/sessions/does-not-exist").status_code == 404