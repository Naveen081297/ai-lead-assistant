"""Retrieval-only evaluation: embeddings + Chroma, no chat-model calls.

Run:  python eval/retrieval_eval.py [--k 4]

Scores all 25 questions on whether the right document is retrieved, and
prints the top similarity score per question. Comparing top scores of
answerable vs unanswerable questions shows whether a similarity threshold
alone could decide "no relevant document, refuse" (Week 2 guardrail).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lead_assistant.rag import RagPipeline

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    questions = [json.loads(l) for l in (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rag = RagPipeline()
    k = args.k or rag.settings.top_k

    answerable, unanswerable = [], []
    print(f"{'id':4} {'rank':>4} {'top':>6}  question  ->  top doc")
    for q in questions:
        passages = rag.retrieve(q["question"], k=k)
        docs = list(dict.fromkeys(p.source for p in passages))
        expected = set(q["expected_sources"])
        rank = next((i for i, s in enumerate(docs, 1) if s in expected), None) if expected else None
        top = passages[0].score if passages else 0.0
        row = {"id": q["id"], "rank": rank, "top": top}
        (unanswerable if q.get("expect_refusal") else answerable).append(row)
        print(f"{q['id']:4} {str(rank or '-'):>4} {top:6.3f}  {q['question'][:60]}  ->  {docs[0] if docs else '-'}")

    n = len(answerable)
    hit1 = sum(r["rank"] == 1 for r in answerable)
    hitk = sum(r["rank"] is not None for r in answerable)
    mrr = sum(1 / r["rank"] for r in answerable if r["rank"]) / n
    print(f"\nAnswerable questions: {n}")
    print(f"hit@1  {hit1}/{n} ({100 * hit1 / n:.0f}%)")
    print(f"hit@{k}  {hitk}/{n} ({100 * hitk / n:.0f}%)")
    print(f"MRR    {mrr:.3f}")

    if unanswerable:
        lo_ans = min(r["top"] for r in answerable)
        hi_un = max(r["top"] for r in unanswerable)
        print(f"\nTop score, answerable:   min {lo_ans:.3f}  max {max(r['top'] for r in answerable):.3f}")
        print(f"Top score, unanswerable: min {min(r['top'] for r in unanswerable):.3f}  max {hi_un:.3f}")
        if hi_un < lo_ans:
            print(f"Separable: any threshold between {hi_un:.3f} and {lo_ans:.3f} splits them.")
        else:
            print("Not separable by a threshold alone: some unanswerable questions score higher "
                  "than some answerable ones. The LLM's refusal rule is still needed.")


if __name__ == "__main__":
    main()
