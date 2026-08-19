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
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = os.getenv("PDF_PATH", "motor_laws_sample.pdf")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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

    embeddings = get_embeddings()
    if reset:
        print(f"Deleting existing collection '{COLLECTION_NAME}' (--reset)...")
        Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        ).delete_collection()

    print(f"Writing {len(chunks)} chunks to Chroma collection '{COLLECTION_NAME}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"Successfully saved to {CHROMA_PATH}/{COLLECTION_NAME}.")


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
    parser = argparse.ArgumentParser(description="Ingest legal PDF into Chroma.")
    parser.add_argument("--dry-run", action="store_true", help="Extract + chunk but do not write.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the existing collection first (avoids duplicate chunks).")
    args = parser.parse_args()
    ingest_data(dry_run=args.dry_run, reset=args.reset)