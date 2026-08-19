import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

from chat import query_rag
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Comma-separated API keys. Empty = auth disabled (development mode).
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))  # human+ai pairs kept
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
SESSION_DB = os.getenv("SESSION_DB", "sessions.db")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

security = HTTPBearer(auto_error=False)


def _rate_limit_key(request: Request) -> str:
    """Rate limit per API key when auth is enabled; fall back to client IP."""
    if API_KEYS:
        auth = request.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            if token:
                return "key:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

logger = logging.getLogger("legal-rag-api")

app = FastAPI(
    title="Motor Legal Chatbot API",
    description="An API to ask questions about Motor Laws using RAG.",
    version="1.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Reject requests without a valid API key when keys are configured."""
    if not API_KEYS:
        return  # auth disabled
    if credentials is None or credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = Field(None, max_length=64)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceDoc(BaseModel):
    page: int
    content: str


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceDoc]


class SessionSummary(BaseModel):
    session_id: str
    title: str
    updated_at: float


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionDetail(BaseModel):
    session_id: str
    messages: List[SessionMessage]


# ---------------------------------------------------------------------------
# Persistent session store (SQLite). Thread-safe; survives restarts/workers.
# ---------------------------------------------------------------------------

class SessionStore:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            " id TEXT PRIMARY KEY, history TEXT, title TEXT DEFAULT 'New chat',"
            " created_at REAL, updated_at REAL)"
        )
        # Migrate databases created before sessions were listable/deletable.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(sessions)")}
        if "title" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT 'New chat'")
        if "updated_at" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN updated_at REAL")
        self._conn.commit()

    @staticmethod
    def _default_title(history: list) -> str:
        """Sidebar label: the first user question, truncated."""
        for m in history:
            if isinstance(m, HumanMessage):
                return " ".join(m.content.split())[:60] or "New chat"
        return "New chat"

    def get(self, session_id: str) -> Optional[list]:
        with self._lock:
            row = self._conn.execute(
                "SELECT history FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        messages = []
        for item in json.loads(row[0]):
            cls = HumanMessage if item.get("role") == "human" else AIMessage
            messages.append(cls(content=item.get("content", "")))
        return messages

    def create(self, session_id: str) -> None:
        now = time.time()
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if count >= MAX_SESSIONS:
                self._conn.execute(
                    "DELETE FROM sessions WHERE id = ("
                    " SELECT id FROM sessions ORDER BY updated_at ASC, created_at ASC LIMIT 1)"
                )
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions"
                " (id, history, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, json.dumps([]), "New chat", now, now),
            )
            self._conn.commit()

    def save(self, session_id: str, history: list) -> None:
        payload = json.dumps([
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history
        ])
        title = self._current_title(session_id)
        if not title or title == "New chat":
            title = self._default_title(history)
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET history=?, title=?, updated_at=? WHERE id=?",
                (payload, title, time.time(), session_id),
            )
            self._conn.commit()

    def _current_title(self, session_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT title FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return row[0] if row else None

    def list_all(self) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, updated_at FROM sessions"
                " ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (MAX_SESSIONS,),
            ).fetchall()
        return [
            {"session_id": r[0], "title": r[1] or "New chat", "updated_at": r[2] or 0}
            for r in rows
        ]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
        return cur.rowcount > 0


sessions = SessionStore(SESSION_DB)


def _get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if sessions.get(session_id) is None:
        sessions.create(session_id)
    return session_id


def _trim_history(history: list) -> list:
    # Keep the most recent MAX_HISTORY_TURNS exchanges.
    keep = MAX_HISTORY_TURNS * 2  # each turn = 1 human + 1 ai message
    return history[-keep:] if len(history) > keep else history


def _http_error_for(e: Exception) -> HTTPException:
    """Map upstream provider errors to honest HTTP status codes."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is None:
        status = getattr(e, "status_code", None)
    if status == 429:
        return HTTPException(status_code=429, detail="Upstream model rate limit reached; try again later.")
    if status and status >= 500:
        return HTTPException(status_code=503, detail="Upstream model provider unavailable; try again later.")
    return HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Legal RAG API is running"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def ask_question(request: Request, req: AskRequest, response: Response):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = request_id
    start = time.time()

    try:
        session_id = _get_or_create_session(req.session_id)
        history = list(sessions.get(session_id) or [])
        rag_result = query_rag(req.question.strip(), history=history, api_mode=True)
        sessions.save(session_id, _trim_history(history))

        sources = [
            SourceDoc(
                page=doc.metadata.get("page", 0),
                content=doc.page_content[:2000],
            )
            for doc in rag_result.get("context", [])
        ]
        result = AskResponse(
            answer=rag_result["answer"],
            session_id=session_id,
            sources=sources,
        )
        elapsed = time.time() - start
        logger.info(
            "req=%s session=%s latency=%.2fs sources=%d max_relevance=%.2f",
            request_id,
            session_id,
            elapsed,
            len(sources),
            (rag_result.get("scores") or {}).get("max_relevance", 0.0),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("req=%s unhandled error while answering", request_id)
        raise _http_error_for(e)


@app.get("/sessions", response_model=List[SessionSummary], dependencies=[Depends(require_api_key)])
def list_sessions():
    """All sessions, most recently active first (for the sidebar)."""
    return sessions.list_all()


@app.post("/sessions", dependencies=[Depends(require_api_key)])
def create_session():
    """Create a new, empty chat thread (active in parallel with existing ones)."""
    session_id = str(uuid.uuid4())
    sessions.create(session_id)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}", response_model=SessionDetail, dependencies=[Depends(require_api_key)])
def get_session(session_id: str):
    """Full message history for one thread (to re-render the chat on switch)."""
    history = sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history
        ],
    }


@app.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_session(session_id: str):
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


app.mount("/", StaticFiles(directory="static", html=True), name="static")