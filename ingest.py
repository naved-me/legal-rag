import os
import re
import hashlib
import argparse
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

PDF_PATH = os.getenv("PDF_PATH", "motor_laws_sample.pdf")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "384"))  # all-MiniLM-L6-v2
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_pinecone():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to .env to ingest into Pinecone.")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    """Create the Pinecone index if it doesn't exist yet (serverless)."""
    pc = get_pinecone()
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' "
              f"(dim={PINECONE_DIMENSION}, metric={PINECONE_METRIC})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def clean_text(text: str) -> str:
    # Collapse runs of spaces/tabs but keep paragraph breaks so the
    # structure-aware splitter can still split on "\n\n" and "\n".
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_toc_page(text: str) -> bool:
    upper = text.upper()
    if "ARRANGEMENT OF SECTIONS" in upper:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    numbered = sum(1 for ln in lines if re.match(r"^\d+[A-Z]?\.?\s+\w", ln))
    dense = sum(1 for ln in lines if len(ln) <= 65)
    return numbered / len(lines) > 0.6 and dense / len(lines) > 0.9


def extract_pages() -> list:
    reader = PdfReader(PDF_PATH)
    documents = []
    skipped = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = clean_text(text)
        if len(cleaned) < 10:
            skipped.append((page_num, "near-empty"))
            continue
        if is_toc_page(cleaned):
            skipped.append((page_num, "table-of-contents"))
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={"page": page_num, "source": PDF_PATH},
            )
        )
    return documents, skipped


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    kept = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        chunk.page_content = content
        chunk.metadata["chunk_id"] = hashlib.sha1(
            content.encode("utf-8")
        ).hexdigest()[:12]
        kept.append(chunk)
    return kept


def ingest_data(dry_run: bool = False, reset: bool = False) -> None:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}. Please add it to the folder.")

    print("Extracting text from PDF...")
    documents, skipped = extract_pages()
    print(f"Extracted {len(documents)} usable pages; skipped {len(skipped)} junk pages.")
    for kind, count in _counts(skipped):
        print(f"  - {kind}: {count}")

    print("Chunking text (recursive, structure-aware)...")
    chunks = split_documents(documents)
    lens = [len(c.page_content) for c in chunks]
    print(
        f"Created {len(chunks)} chunks "
        f"(len min/median/max: {min(lens)}/{_median(lens)}/{max(lens)})."
    )

    if dry_run:
        print("DRY RUN: no documents written to the database.")
        return

    index = ensure_index()
    if reset:
        print(f"Deleting all vectors from '{PINECONE_INDEX_NAME}' (--reset)...")
        index.delete(delete_all=True, namespace="")

    print(f"Upserting {len(chunks)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'...")
    vector_store = PineconeVectorStore(index=index, embedding=get_embeddings(), text_key="text")
    # chunk_id is a sha1 of the content, so upserts are idempotent (no duplicates).
    vector_store.add_documents(
        documents=chunks,
        ids=[c.metadata["chunk_id"] for c in chunks],
        namespace="",
    )
    print(f"Successfully saved {len(chunks)} chunks to Pinecone index '{PINECONE_INDEX_NAME}'.")


def _counts(skipped):
    by = {}
    for _, kind in skipped:
        by[kind] = by.get(kind, 0) + 1
    return sorted(by.items())


def _median(nums):
    nums = sorted(nums)
    n = len(nums)
    mid = n // 2
    if n % 2 == 1:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) // 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal PDF into Pinecone.")
    parser.add_argument("--dry-run", action="store_true", help="Extract + chunk but do not write.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all existing vectors first (avoids duplicates).")
    args = parser.parse_args()
    ingest_data(dry_run=args.dry_run, reset=args.reset)