import os
import json
import random
import argparse
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal-docs")
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


def generate_benchmark(seed: int = 42):
    print(f"--- Generating up to {NUM_QUESTIONS} Test Questions (seed={seed}) ---", flush=True)
    random.seed(seed)

    print("Connecting to Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    data = db.get()
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

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
    Your task is to write ONE specific, highly factual question based on this text, and provide the correct answer strictly from the text.

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
    parser = argparse.ArgumentParser(description="Generate a grounded benchmark from the Chroma DB.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible chunk sampling (default: 42).")
    args = parser.parse_args()
    generate_benchmark(seed=args.seed)