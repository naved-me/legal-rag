import os
import re
import time
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_FETCH_K = int(os.getenv("RERANK_FETCH_K", "25"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "1.0"))
RERANK_HARD_FLOOR = float(os.getenv("RERANK_HARD_FLOOR", "-2.0"))
# Fallback relevance gate used only if the reranker is unavailable.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.30"))


@lru_cache(maxsize=1)
def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=_embeddings(),
        collection_name=COLLECTION_NAME,
    )


@lru_cache(maxsize=1)
def _reranker():
    try:
        return CrossEncoder(RERANK_MODEL, max_length=512)
    except Exception as e:
        print(f"WARNING: reranker unavailable ({e}); using embedding-only retrieval.")
        return None


def _llm():
    kwargs = {"model": LLM_MODEL, "temperature": 0}
    # qwen is a reasoning model that leaks its chain-of-thought inline.
    # Turn reasoning off at the API level to keep answers clean and fast.
    if "qwen" in LLM_MODEL:
        kwargs["reasoning_effort"] = "none"
    return ChatGroq(**kwargs)


def _invoke(chain, inputs, retries=2, backoff=10.0):
    """Invoke an LCEL chain with retry/backoff for transient provider errors."""
    for attempt in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is None:
                status = getattr(e, "status_code", None)
            msg = str(e)
            daily_quota = status == 429 and ("tokens per day" in msg.lower() or "tpd" in msg.lower())
            retryable = status in (429, 500, 502, 503, 504) and not daily_quota
            if attempt < retries and retryable:
                time.sleep(backoff * (attempt + 1))
                continue
            raise


def strip_thinking(text: str) -> str:
    """Defense-in-depth: remove any chain-of-thought a reasoning model leaks."""
    marker = text.rfind("\nresponse\n")
    if marker != -1:
        return text[marker + len("\nresponse\n"):].strip()
    text = text.strip()
    if re.match(r"^thinking\b", text, flags=re.IGNORECASE):
        # No response marker: drop only the leading "thinking ..." preamble line,
        # keeping anything that looks like an actual answer.
        first_nl = text.find("\n")
        if first_nl == -1:
            return ""
        return text[first_nl + 1:].strip()
    return text


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _bm25_index():
    """In-memory BM25 index over the same chunks as the vector store."""
    data = _store().get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"] or [], data["metadatas"] or [])
    ]
    tokenized = [_tokenize(d.page_content) for d in docs]
    return BM25Okapi(tokenized), docs


def reload_caches():
    """Drop cached singletons (embeddings, store, BM25). Call after re-ingesting."""
    _embeddings.cache_clear()
    _store.cache_clear()
    _bm25_index.cache_clear()
    _reranker.cache_clear()


def _hybrid_candidates(question: str, k: int = RERANK_FETCH_K):
    """Fuse vector + BM25 results via Reciprocal Rank Fusion (then re-ranked upstream)."""
    bm25, all_docs = _bm25_index()
    vec_hits = _store().similarity_search(question, k=k)

    fused = {}
    for rank, doc in enumerate(vec_hits, start=1):
        cid = doc.metadata.get("chunk_id")
        fused[cid] = [1.0 / (60 + rank), doc]

    bm25_scores = bm25.get_scores(_tokenize(question))
    ranked_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm_rank = 0
    for idx in ranked_ids:
        if bm25_scores[idx] <= 0:
            continue
        bm_rank += 1
        doc = all_docs[idx]
        cid = doc.metadata.get("chunk_id")
        entry = fused.get(cid)
        if entry:
            entry[0] += 1.0 / (60 + bm_rank)
        else:
            fused[cid] = [1.0 / (60 + bm_rank), doc]

    ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)
    return [doc for _score, doc in ranked][:k]


def _reranked_with_scores(question: str, candidates):
    """Cross-encode candidates and return (doc, score) pairs."""
    reranker = _reranker()
    if reranker is None:
        return [(doc, 0.0) for doc in candidates]
    try:
        scores = reranker.predict(
            [(question, doc.page_content) for doc in candidates],
            show_progress_bar=False,
        )
    except Exception:
        return [(doc, 0.0) for doc in candidates]
    return list(zip(candidates, scores))


def _rank_context(question: str):
    """Hybrid retrieval -> cross-encoder re-rank -> confidence gate."""
    candidates = _hybrid_candidates(question)
    if not candidates:
        return [], 0.0

    reranker = _reranker()
    if reranker is None:
        # Fallback: gate on vector relevance only.
        hits = _store().similarity_search_with_relevance_scores(question, k=RERANK_FETCH_K)
        kept = [doc for doc, score in hits if (score or 0) >= MIN_RELEVANCE_SCORE]
        return kept[:RETRIEVAL_K], max((s or 0) for _, s in hits)

    scored = _reranked_with_scores(question, candidates)
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    max_score = float(ranked[0][1]) if ranked else 0.0

    # Refuse only when even the best candidate is egregiously irrelevant.
    if max_score < RERANK_HARD_FLOOR:
        return [], max_score

    kept = [doc for doc, score in ranked if float(score) >= RERANK_MIN_SCORE][:RETRIEVAL_K]
    if not kept:
        # Nothing cleared the bar, but the top result is still the best signal
        # the system has; answer from it rather than refusing outright.
        kept = [ranked[0][0]]
    return kept, max_score


def _rewrite_question(question: str, history: list) -> str:
    if not history:
        return question
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )),
        ("system", "Chat history:\n{history_text}"),
        ("human", "{input}"),
    ])
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )
    result = _invoke(prompt | _llm() | StrOutputParser(), {
        "history_text": history_text,
        "input": question,
    }).strip()
    return result if result else question


def _format_context(documents: list) -> str:
    """Prefix each chunk with its source page so the LLM can cite it inline."""
    parts = []
    for doc in documents:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _answer(question: str, documents: list, history: list) -> str:
    system_prompt = (
        "You are a helpful legal assistant specializing in Motor Laws. "
        "Use only the following retrieved context to answer the user's question. "
        "Answer the question directly and precisely; do not add related provisions "
        "that do not directly answer the question. "
        "After each fact or claim you state, add an inline citation like [Page N] "
        "referencing the page of the context it came from. Quote section numbers where possible. "
        "If the answer is not in the context, just say that you don't know."
        "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.replace("{context}", _format_context(documents))),
        ("human", "{input}"),
    ])
    return _invoke(prompt | _llm() | StrOutputParser() | strip_thinking, {"input": question})


REFUSAL = (
    "I could not find information relevant to that question in the "
    "Motor Vehicles Act document I have access to."
)


def query_rag(question: str, history: list = None, api_mode: bool = False):
    if history is None:
        history = []

    if not api_mode:
        print("Loading database (cached across calls)...")

    documents, max_score = _rank_context(question)

    if not documents:
        if not api_mode:
            print(f"\n--- Answer ---\n{REFUSAL}")
            print(f"\n--- Retrieval confidence: {max_score:.3f} (below re-ranking threshold) ---")
        history.extend([HumanMessage(content=question), AIMessage(content=REFUSAL)])
        return {
            "answer": REFUSAL,
            "context": [],
            "scores": {"max_relevance": max_score, "rerank_min_score": RERANK_MIN_SCORE},
        }

    # Multi-turn only: rewrite then re-retrieve (single-turn reuses the first retrieval).
    if history:
        standalone = _rewrite_question(question, history)
        if standalone != question:
            reranked, _ = _rank_context(standalone)
            if reranked:
                documents = reranked
                question = standalone

    if not api_mode:
        print(f"\nThinking about: '{question}'...")
    answer = _answer(question, documents, history)

    if not api_mode:
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources Used ---")
        for doc in documents:
            print(f"- Page {doc.metadata.get('page')}")

    history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])

    return {
        "answer": answer,
        "context": documents,
        "scores": {"max_relevance": max_score, "rerank_min_score": RERANK_MIN_SCORE},
    }


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Welcome to the Motor Legal Chatbot!")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if user_input.strip() == "":
            continue
        query_rag(user_input, chat_history)
        print("\n" + "=" * 50 + "\n")