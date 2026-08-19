"""Offline retrieval check: measures hit-rate@k and MRR on the benchmark
using the exact production retrieval path (hybrid BM25+vector -> re-rank ->
confidence gate). No LLM calls, so it costs zero tokens."""

import os
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from chat import _rank_context, RETRIEVAL_K
from benchmark_utils import gold_hit


def main():
    if not os.path.exists("benchmark.json"):
        print("benchmark.json not found. Run generate_testset.py first.")
        return

    with open("benchmark.json", encoding="utf-8") as f:
        items = json.load(f)

    hits = 0
    mrr_sum = 0.0
    latencies = []
    refused = 0

    print(f"--- Offline retrieval check on {len(items)} questions (k={RETRIEVAL_K}) ---\n", flush=True)
    for item in items:
        q = item["question"]

        t0 = time.time()
        docs, max_score = _rank_context(q)
        latencies.append(time.time() - t0)

        hit, rank = gold_hit(item, docs)
        if hit:
            hits += 1
            mrr_sum += 1.0 / rank
        if not docs:
            refused += 1

        status = f"rank={rank}" if rank else ("REFUSED" if not docs else "miss")
        print(f"  id={item['id']:>2} {status:<8} rerank_top={max_score:+.2f} | {q[:60]}", flush=True)

    total = len(items)
    print("\n" + "=" * 56)
    print("RETRIEVAL CHECK RESULTS")
    print("=" * 56)
    print(f"Hit-rate@{RETRIEVAL_K}: {hits}/{total} ({hits/max(total,1)*100:.1f}%)")
    print(f"MRR@{RETRIEVAL_K}: {mrr_sum/max(total,1):.3f}")
    print(f"Refused (below hard floor): {refused}/{total}")
    print(f"Avg retrieval time: {sum(latencies)/max(len(latencies),1)*1000:.0f} ms")


if __name__ == "__main__":
    main()