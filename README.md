# Motor Legal Chatbot — RAG over Legal PDFs

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions about the
Motor Vehicles Act from your PDF, with **grounded, page-cited answers** and a full evaluation harness.

- **Hybrid retrieval**: Pinecone vector search + BM25 keyword search fused with Reciprocal Rank Fusion
- **Cross-encoder re-ranking** for precision, with a confidence gate that refuses out-of-scope questions
- **Fast, deterministic LLM** (Groq Qwen) with inline `[Page N]` citations
- **Production API**: FastAPI, per-IP rate limiting, CORS, SQLite-backed chat sessions with a session sidebar
- **Zero-token evaluation**: hit-rate@k / MRR checks plus an LLM-as-a-Judge accuracy harness

---

## Architecture

```
 ┌──────────────┐   ingest.py    ┌──────────────────────┐
 │ PDF document │ ─────────────► │ Pinecone index (cloud)│   serverless, cosine, 384-d
 └──────────────┘   extract→      └──────────────────────┘
                   clean→chunk→        ▲           ▲
                   embed→upsert         │ vector    │ BM25 (in-memory,
                                       │ search     │ built via enumeration trick)
┌───────────┐   query                  │           │
│  client   │ ───────► chat.py ────────┴───────────┘
│ (browser/ │           hybrid candidates (25) → RRF fusion
│  API)     │              → cross-encoder re-rank → confidence gate
└─────┬─────┘                 → top-6 context + history → Groq LLM
      │                       → answer with [Page N] citations
      └─────────────────────────────▲
              api.py (FastAPI): rate limit, SQLite sessions, CORS, X-Request-ID
```

---

## Quick Start

### 1. Environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
# Required
GROQ_API_KEY=gsk_...
PINECONE_API_KEY=pcsk_...

# Optional
HF_TOKEN=hf_...                 # faster HF model downloads
PINECONE_INDEX_NAME=legal-rag
```

### 3. Ingest the document

```powershell
python ingest.py                # auto-creates the index; idempotent (no duplicates)
python ingest.py --reset        # wipe vectors first (after chunking changes)
python ingest.py --dry-run      # preview extraction/chunk stats, write nothing
```

### 4. Serve

```powershell
venv\Scripts\activate
uvicorn api:app --reload        # http://localhost:8000
```

Terminal-only chat: `python chat.py`

### 5. Test & evaluate

```powershell
python -m pytest -q                              # unit + integration tests
python generate_testset.py --seed 42             # regenerate grounded benchmark (15 Qs)
python retrieval_check.py                        # offline hit-rate@k + MRR (zero tokens)
python evaluate.py                               # LLM-judged answer accuracy
```

---

## API Reference

### `GET /health`
Health probe. Returns `{"status": "online"}`.

### `POST /ask`
Ask a question with optional conversational session.

**Headers**
| Header | Required | Description |
|---|---|---|
| `X-Request-ID` | no | Echoed back for request tracing |

**Body**
```json
{
  "question": "What is the maximum fine for a second conviction?",
  "session_id": "optional-uuid-for-continued-conversation"
}
```

**Response**
```json
{
  "answer": "The fine may extend to five hundred rupees. [Page 88]",
  "session_id": "4f2c...",
  "sources": [ { "page": 88, "content": "drive a motor vehicle ..." } ]
}
```

**Errors**: `422` bad input · `429` upstream rate limit · `503` upstream outage · `500` internal.

---

## Configuration

All optional — sensible defaults shown.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | LLM provider key (required) |
| `PINECONE_API_KEY` | — | Vector DB key (required) |
| `PINECONE_INDEX_NAME` | `legal-rag` | Index (auto-created at ingest) |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless index placement |
| `PINECONE_DIMENSION` | `384` | Must match the embedding model |
| `PINECONE_METRIC` | `cosine` | Index similarity metric |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1100` / `200` | Ingestion chunking |
| `MIN_CHUNK_CHARS` | `60` | Drop shorter chunks at ingest |
| `PDF_PATH` | `motor_laws_sample.pdf` | Source document |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LLM_MODEL` | `qwen/qwen3.6-27b` | Answer model (Groq) |
| `RETRIEVAL_K` | `3` | Chunks passed to the LLM (kept small for cost) |
| `CONTEXT_MAX_WORDS` | `1500` | Hard word budget for the prompt context |
| `FALLBACK_MODEL` | `llama-3.1-8b-instant` | Light model used when the primary LLM fails |
| `RERANK_FETCH_K` | `25` | Candidates before re-ranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranker |
| `RERANK_MIN_SCORE` | `-1.0` | Re-ranker gate for context entry |
| `RERANK_HARD_FLOOR` | `-2.0` | Below this, refuse to answer |
| `RATE_LIMIT` | `20/minute` | Per-IP `/ask` limit |
| `MAX_SESSIONS` | `1000` | Session-store cap (oldest evicted) |
| `MAX_HISTORY_TURNS` | `6` | Chat turns kept per session |
| `SESSION_DB` | `sessions.db` | SQLite session store |
| `ALLOWED_ORIGINS` | `*` | CORS origins (restrict in production) |
| `JUDGE_MODEL` | `=LLM_MODEL` | Eval judge model (set different to avoid self-judging) |
| `JUDGE_VOTES` | `1` | Judge calls per question (majority verdict) |
| `NUM_QUESTIONS` | `15` | Benchmark size |

---

## Deployment (Docker)

Vectors live in the cloud index, so containers are stateless — no index data to persist, just keys.

```powershell
# Build
docker build -t legal-rag .

# Run (mount sessions.db so chat history survives restarts)
docker run -p 8000:8000 `
  -v ${PWD}/sessions.db:/app/sessions.db `
  --env-file .env `
  legal-rag
```

> Populate the index first: `python ingest.py --reset` (locally or from any container with the keys).

---

## Evaluation Results

Run the harness yourself for current numbers; representative output on the included sample:

```
Answer accuracy (LLM-judged):   15/15 (100%)
Retrieval hit-rate@3:           15/15 (100%)
Retrieval MRR@3:                0.822
Refused (out-of-scope):          0/15
```

The offline `retrieval_check.py` needs **zero LLM tokens** — use it as a fast regression check whenever
you change chunking, retrieval, or thresholds.

---

## Project Structure

```
├── api.py               FastAPI server (rate limit, sessions, error mapping)
├── chat.py              RAG pipeline: hybrid retrieval → re-rank → generate
├── ingest.py            PDF → clean → chunk → embed → Pinecone
├── evaluate.py          LLM-as-a-Judge answer accuracy
├── retrieval_check.py   Zero-token hit-rate@k / MRR
├── generate_testset.py  Grounded benchmark generator (seeded, reproducible)
├── benchmark_utils.py   Shared benchmark/gold-hit helpers
├── benchmark.json       Generated test set
├── tests/               Pytest suite (mocks LLM calls; deterministic logic tested hard)
├── static/              Web UI (vanilla JS, XSS-sanitized)
├── Dockerfile           Stateless container
└── requirements.txt     Pinned dependencies
```

---

## Security Notes

- Keys live only in `.env` (gitignored) — never commit them.
- All rendered text is sanitized client-side (XSS protection).
- Set `ALLOWED_ORIGINS` to your frontend domain before exposing publicly.

---

## Production Guardrails & Cost Mitigation

This project is built to run as an **unauthenticated, zero-friction demo** while staying safe against
API abuse and token exhaustion on a shared cloud host. The layout was chosen deliberately:

- **Proxy-Trust Client Tracking (Phase 1):** the server is started with
  `--proxy-headers --forwarded-allow-ips "*"` so it reads `X-Forwarded-For` from the cloud load
  balancer. Without this, every visitor would look like the same IP and rate limiting would be useless.
- **Edge-Guard Rate Limiting (Phase 2):** the `/ask` endpoint is limited to **20 requests/minute per
  real client IP** (SlowAPI). The 21st request in a minute gets a clean `429 Too Many Requests` JSON
  response. This protects the shared Groq quota from any single user or bot.
- **Token-Aware UI Constraints (Phase 3):** the chat input enforces a **1,000-character cap**. If a user
  pastes a huge document, the Send button freezes and a notice explains the demo cap — stopping a single
  click from wiping out the minute's token budget. The API enforces the same limit server-side as a backstop.
- **Defensive Context Ingestion (Phase 4):** retrieval returns at most **3 chunks** (`RETRIEVAL_K=3`) and
  the prompt builder enforces a hard **1,500-word context budget** (`CONTEXT_MAX_WORDS`), trimming excess
  text before the call. This keeps every payload safely under the model's Tokens-Per-Minute limit.
- **High-Availability Failover (Phase 5):** every Groq call is wrapped in `_invoke_fallback()`. If the
  primary model errors (429 quota, 5xx outage), the request is **immediately refired on a lighter 8B
  fallback model** (`FALLBACK_MODEL`, with its own separate quota). Users almost never see an error screen.

These guardrails trade a little context depth for dramatically lower cost and high uptime — the right
balance for a public demo that must never go blank.

---

## License & Disclaimer

Educational/demonstration software. **Not legal advice.** Answers are grounded in the provided document
and may contain errors — verify against the source before relying on them.