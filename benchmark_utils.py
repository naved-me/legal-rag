"""Shared helpers for benchmark loading and gold-hit checking."""

import json


def load_benchmark(path="benchmark.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_hit(item, docs):
    """Check whether the gold chunk appears in the final context, returning (hit, rank)."""
    gold_id = item.get("chunk_id")
    gold_text = (item.get("source_context") or "").strip()
    for rank, doc in enumerate(docs, start=1):
        if gold_id and doc.metadata.get("chunk_id") == gold_id:
            return True, rank
        if gold_text and doc.page_content.strip() == gold_text:
            return True, rank
    return False, None