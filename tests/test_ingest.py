import os

import pytest

import ingest


def test_clean_text_keeps_paragraph_breaks():
    raw = "Line one.\n\nLine two.\nLine three."
    cleaned = ingest.clean_text(raw)
    assert "\n\n" in cleaned
    assert "\n" in cleaned


def test_clean_text_collapses_space_runs():
    cleaned = ingest.clean_text("a   b\t\tc")
    assert cleaned == "a b c"


def test_toc_page_detection():
    toc = "ARRANGEMENT OF SECTIONS\n\n1. Short title\n2. Definitions\n3. Extent"
    assert ingest.is_toc_page(toc)
    assert not ingest.is_toc_page("Section 182. Whoever drives a vehicle at a speed exceeding limits.")


def test_split_keeps_min_chunk_size():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="A. " + ("Penalty provisions. " * 60), metadata={"page": 1}),
        Document(page_content="Short.", metadata={"page": 2}),
    ]
    chunks = ingest.split_documents(docs)
    assert all(len(c.page_content) >= ingest.MIN_CHUNK_CHARS for c in chunks)
    assert all(c.metadata.get("chunk_id") for c in chunks)


@pytest.mark.skipif(
    not os.path.exists(ingest.PDF_PATH), reason="sample PDF not present"
)
def test_dry_run_extracts_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(ingest, "COLLECTION_NAME", "test-legal-docs")
    ingest.ingest_data(dry_run=True)
    assert not (tmp_path / "chroma").exists()