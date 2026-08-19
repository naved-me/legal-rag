# Motor Legal Chatbot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Motor Laws. It extracts text from local PDF documents, stores them in a Pinecone vector index, and uses a cloud-based LLM (Groq) to provide factually accurate, hallucination-free legal answers.

## Features
- **Offline Embeddings**: Uses HuggingFace (`all-MiniLM-L6-v2`) to generate document embeddings completely locally.
- **Vector Database**: Uses a Pinecone serverless index (`legal-rag`) for cloud-hosted vector storage and retrieval.
- **Fast Cloud LLM**: Uses Groq (`qwen/qwen3.6-27b`, reasoning disabled) for fast, clean fact-based answer generation.
- **LCEL Architecture**: Built using modern LangChain Expression Language for clean, readable pipelines.
- **Evaluation**: An LLM-as-a-Judge script scores the bot on answer accuracy and reports retrieval metrics (hit-rate@k, MRR) over a regenerable benchmark.
- **Re-ranking**: A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-ranks the top-25 embedding candidates to deliver the final top-6 context, with a confidence gate that refuses low-confidence questions.

## Project Files
- `ingest.py`: Parses the PDFs, splits the text into chunks, creates embeddings, and upserts them to the Pinecone index.
- `chat.py`: The main RAG retrieval chain. Takes user questions, retrieves relevant chunks from Pinecone, and asks the LLM to generate an answer.
- `evaluate.py`: An automated testing script that grades the chatbot's answers to ensure they don't hallucinate.
- `retrieval_check.py`: Offline (zero-token) retrieval evaluation — hit-rate@k and MRR.
- `generate_testset.py`: Regenerates the grounded `benchmark.json` from document chunks.
- `benchmark_utils.py`: Shared benchmark loading + gold-hit helpers used by the eval scripts.
- `api.py`: A FastAPI web server that provides a `/ask` endpoint with conversational memory (SQLite-backed sessions, auth, rate limiting).
- `tests/`: Pytest suite covering the API, ingestion, and retrieval helpers.
- `static/`: Contains the premium Vanilla JS web interface.
- `plan.md`: The roadmap and architecture phases for the project.
- `learn.md`: A learning log of architectural decisions and concepts.

## Setup Instructions

1. **Create and Activate a Virtual Environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   - Create a file named `.env` in the root folder.
   - Add your Groq and Pinecone API keys:
     ```env
     GROQ_API_KEY=gsk_your_api_key_here
     PINECONE_API_KEY=pcsk_your_api_key_here
     ```

4. **Ingest the Data**
   Run the ingestion script. It creates the Pinecone index automatically if it doesn't exist:
   ```powershell
   python ingest.py
   ```
   Re-ingesting upserts by chunk id (idempotent, no duplicates); pass `--reset` to wipe existing vectors first:
   ```powershell
   python ingest.py --reset
   ```
   Preview extraction/chunking without writing with `--dry-run`.

5. **Run the Web Application (Recommended)**
   Launch the FastAPI web server to use the graphical chat interface:
   ```powershell
   uvicorn api:app --reload
   ```
   Then open `http://localhost:8000/` in your web browser.

6. **Chat via Terminal (Alternative)**
   Run the chat script directly to use the terminal interface:
   ```powershell
   python chat.py
   ```

7. **Benchmark & Evaluate**
   Regenerate the test set (grounded in the actual document, 15 questions):
   ```powershell
   python generate_testset.py --seed 42
   ```
   Then run the evaluator:
   ```powershell
   python evaluate.py
   ```
   The report shows answer accuracy (LLM-judged), retrieval hit-rate@6, and MRR.

   Quick offline retrieval check (zero LLM tokens — good for CI):
   ```powershell
   python retrieval_check.py
   ```

8. **Run the tests**
   ```powershell
   python -m pytest -q
   ```

## Configuration (optional `.env` overrides)

| Variable | Default | Purpose |
|---|---|---|
| `API_KEYS` | empty | Comma-separated keys. Empty = auth disabled. Send via `Authorization: Bearer <key>`. |
| `RATE_LIMIT` | `20/minute` | Per-client request limit on `/ask`. |
| `MAX_SESSIONS` | `1000` | Bounds the session store (oldest evicted). |
| `MAX_HISTORY_TURNS` | `6` | Chat turns retained per session. |
| `SESSION_DB` | `sessions.db` | SQLite file backing the session store (survives restarts). |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. |
| `PINECONE_INDEX_NAME` | `legal-rag` | Pinecone index name (auto-created at ingest). |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless index placement. |
| `PINECONE_DIMENSION` | `384` | Vector dimension (must match the embedding model). |
| `PINECONE_METRIC` | `cosine` | Similarity metric for the index. |
| `RETRIEVAL_K` | `6` | Final context chunks passed to the LLM. |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to re-rank candidates. |
| `RERANK_FETCH_K` | `25` | Candidates pulled (vector + BM25, fused by RRF) before re-ranking. |
| `RERANK_MIN_SCORE` | `-1.0` | Re-ranker score for a chunk to enter context. |
| `RERANK_HARD_FLOOR` | `-2.0` | Below this, even the best chunk is refused. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1100` / `200` | Ingestion chunking. |
| `MIN_CHUNK_CHARS` | `60` | Chunks shorter than this are dropped at ingest. |
| `JUDGE_MODEL` | `qwen/qwen3.6-27b` | Judge model used by `evaluate.py` (set different from `LLM_MODEL` to reduce self-judging bias). |
| `JUDGE_VOTES` | `1` | How many judge calls per question (majority verdict). |
| `JUDGE_SLEEP` | `2` | Seconds between benchmark items (avoids rate limits). |
| `NUM_QUESTIONS` | `15` | Questions generated by `generate_testset.py`. |

## Deployment (Docker)

```powershell
docker build --build-arg HF_TOKEN=hf_xxx -t legal-rag .
docker run -p 8000:8000 --env-file .env legal-rag
```

Vectors live in the Pinecone cloud index, so containers don't hold index data — they just need
`PINECONE_API_KEY` (and `GROQ_API_KEY`) at runtime. Run `python ingest.py --reset` once locally
(or from any container) to populate the index before first use.