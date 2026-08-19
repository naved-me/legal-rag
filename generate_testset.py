import os
import json
import math
import random
import argparse
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pydantic import BaseModel

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-rag")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "384"))  # all-MiniLM-L6-v2
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "15"))


class QAPair(BaseModel):
    question: str
    ground_truth: str


def _llm():
    kwargs = {"model": LLM_MODEL, "temperature": 0}
    if "qwen" in LLM_MODEL:
        kwargs["reasoning_effort"] = "none"
    return ChatGroq(**kwargs)


def _enumeration_vector(dim: int):
    """Deterministic unit vector to enumerate the whole corpus from Pinecone."""
    rng = random.Random(42)
    v = [rng.uniform(-1, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def generate_benchmark(seed: int = 42):
    print(f"--- Generating up to {NUM_QUESTIONS} Test Questions (seed={seed}) ---", flush=True)
    random.seed(seed)

    print("Connecting to Pinecone...")
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY is not set. Add it to .env first.")
        return
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    count = int(stats["total_vector_count"])
    result = index.query(
        vector=_enumeration_vector(PINECONE_DIMENSION),
        top_k=count,
        include_metadata=True,
        include_values=False,
    )

    ids = []
    docs = []
    metas = []
    for match in result["matches"]:
        meta = match.get("metadata") or {}
        text = meta.get("text") or ""
        if text:
            ids.append(match.get("id"))
            docs.append(text)
            metas.append(meta)

    if not ids:
        print("Error: no chunks found in the database. Run ingest.py first.")
        return

    # Prefer substantial chunks for meaningful questions.
    candidates = [
        (i, ids[i], docs[i], metas[i])
        for i in range(len(ids))
        if docs[i] and len(docs[i]) >= 150
    ]
    if len(candidates) < NUM_QUESTIONS:
        print(f"Warning: only {len(candidates)} substantial chunks. Adjusting goal.")
        sampled = candidates
    else:
        sampled = random.sample(candidates, NUM_QUESTIONS)

    generator_llm = _llm().with_structured_output(QAPair)
    prompt = ChatPromptTemplate.from_template("""
    You are an expert legal examiner. I will provide you with a paragraph from a legal document.
    Your task is to write ONE specific, highly factual question based on this text, and provide the
    correct answer strictly from the text.

    Rules for the question:
    - Make it UNAMBIGUOUS: reference the exact section/sub-section number (e.g. "Section 183(1)")
      when the text contains one, instead of vague phrases like "the sub-section".
    - The question must be answerable ONLY from the provided text.
    - The answer must be a direct, concise quote or fact from the text.

    TEXT:
    {context}
    """)

    benchmark_dataset = []
    for i, (_, chunk_id, chunk, meta) in enumerate(sampled):
        print(f"Generating question {i+1}/{len(sampled)}...", flush=True)
        try:
            qa = prompt | generator_llm
            result = qa.invoke({"context": chunk})
            benchmark_dataset.append({
                "id": i + 1,
                "question": result.question,
                "ground_truth": result.ground_truth,
                "source_context": chunk,
                "chunk_id": (meta or {}).get("chunk_id") or chunk_id,
                "page": (meta or {}).get("page"),
            })
            time.sleep(1)
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}", flush=True)
            time.sleep(3)

    output_file = "benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully generated {len(benchmark_dataset)} questions -> {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a grounded benchmark from the Pinecone index.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible chunk sampling (default: 42).")
    args = parser.parse_args()
    generate_benchmark(seed=args.seed)