import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from chat import query_rag
from benchmark_utils import gold_hit

LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
# Judge can use a different model than the answerer to reduce self-judging bias.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
JUDGE_VOTES = int(os.getenv("JUDGE_VOTES", "1"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
JUDGE_SLEEP = float(os.getenv("JUDGE_SLEEP", "2"))


def _judge_llm():
    kwargs = {"model": JUDGE_MODEL, "temperature": 0}
    if "qwen" in JUDGE_MODEL:
        kwargs["reasoning_effort"] = "none"
    return ChatGroq(**kwargs)


JUDGE_CHAIN = (
    ChatPromptTemplate.from_template("""
You are a strict legal grading assistant. You will be given a QUESTION, a GROUND TRUTH ANSWER,
a source CONTEXT, and a GENERATED ANSWER.

Grade PASS if the GENERATED ANSWER conveys the same facts as the GROUND TRUTH ANSWER and is
supported by the CONTEXT. The generated answer may paraphrase, restructure, or add innocuous
framing (e.g. "According to the text, ...") without penalty, as long as the facts match and
nothing contradicts the ground truth.

Grade FAIL only if the GENERATED ANSWER is NOT supported by the CONTEXT (hallucination) or
CONTRADICTS the GROUND TRUTH ANSWER, or omits the key fact the ground truth states.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
CONTEXT: {context}
GENERATED ANSWER: {answer}

Reply with exactly one word: "PASS" or "FAIL".
""")
    | _judge_llm()
)


def _judge_vote(question: str, ground_truth: str, context: str, answer: str) -> str:
    """Run the judge JUDGE_VOTES times and majority-vote PASS/FAIL."""
    votes = []
    for _ in range(JUDGE_VOTES):
        out = JUDGE_CHAIN.invoke({
            "question": question,
            "ground_truth": ground_truth,
            "context": context,
            "answer": answer,
        }).content.strip().upper()
        votes.append(out)
    passes = sum(1 for v in votes if v == "PASS")
    fails = sum(1 for v in votes if v == "FAIL")
    if passes + fails == 0:
        return "ERROR"
    return "PASS" if passes > fails else "FAIL"


def main():
    benchmark_file = "benchmark.json"
    if not os.path.exists(benchmark_file):
        print(f"Error: {benchmark_file} not found. Run generate_testset.py first.")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(f"--- Evaluating {len(test_questions)} questions ---\n", flush=True)

    passed = 0
    hits = 0
    mrr_sum = 0.0
    latencies = []
    failed_rows = []
    error_rows = []

    for item in test_questions:
        q = item["question"]
        gt = item["ground_truth"]

        print(f"[{item['id']}] {q[:70]}", flush=True)
        t0 = time.time()
        try:
            response = query_rag(q, api_mode=True)
        except Exception as e:
            latencies.append(time.time() - t0)
            print(f"  -> ERROR (query_rag): {str(e)[:120]}", flush=True)
            error_rows.append({"id": item["id"], "question": q, "error": str(e)[:200]})
            time.sleep(JUDGE_SLEEP)
            continue
        latency = time.time() - t0
        latencies.append(latency)

        bot_answer = response["answer"]
        retrieved = response.get("context") or []
        contexts = "\n\n".join(d.page_content for d in retrieved)

        hit, rank = gold_hit(item, retrieved)
        if hit:
            hits += 1
            mrr_sum += 1.0 / rank

        grade = "ERROR"
        if contexts.strip():
            try:
                grade = _judge_vote(q, gt, contexts, bot_answer)
            except Exception as e:
                print(f"  judge error: {str(e)[:120]}", flush=True)
                grade = "ERROR"

        if grade == "PASS":
            passed += 1
        elif grade == "ERROR":
            error_rows.append({"id": item["id"], "question": q, "error": "judge unavailable"})
        else:
            failed_rows.append({
                "id": item["id"],
                "question": q,
                "ground_truth": gt,
                "bot_answer": bot_answer,
            })

        print(f"  -> {grade} | retrieval_hit={hit} rank={rank} | latency={latency:.1f}s", flush=True)
        time.sleep(JUDGE_SLEEP)

    total = len(test_questions)
    evaluated = total - len(error_rows)
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Answer accuracy (LLM-judged): {passed}/{evaluated} ({passed/max(evaluated,1)*100:.1f}%) "
          f"[{len(error_rows)} errored, not counted]")
    print(f"Retrieval hit-rate@{RETRIEVAL_K}: {hits}/{total} ({hits/max(total,1)*100:.1f}%)")
    print(f"Retrieval MRR@{RETRIEVAL_K}: {mrr_sum/max(total,1):.3f}")
    print(f"Avg latency: {sum(latencies)/max(len(latencies),1):.1f}s")

    if failed_rows:
        print("\n--- Failed Questions ---")
        for fq in failed_rows:
            print(f"\nID {fq['id']}: {fq['question']}")
            print(f"  GT: {fq['ground_truth']}")
            print(f"  Bot: {fq['bot_answer'][:300]}")
    if error_rows:
        print(f"\n--- {len(error_rows)} Errored (quota/transient) ---")
        for er in error_rows:
            print(f"  ID {er['id']}: {er['question'][:70]} -> {er['error'][:100]}")


if __name__ == "__main__":
    main()