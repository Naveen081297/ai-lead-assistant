"""Score the RAG pipeline on eval/questions.jsonl.

Run:  python eval/run_eval.py [--k 4] [--no-judge] [--questions eval/questions_hard.jsonl]

Metrics
- Retrieval (answerable questions): hit@1, hit@k and MRR, at document level.
  "Hit" = one of the expected source files is among the retrieved chunks.
- Citation accuracy: the answer cites at least one expected source.
- All sources retrieved (multi-document questions with `needs_sources`): every
  required document group has at least one file among the retrieved chunks.
- Fact match: every `must_contain` item appears in the answer (case-insensitive).
  An item can be a list of accepted alternatives, e.g. ["1,46,320", "146,320"].
- Correctness (questions with a `reference` answer): the LLM judge compares the
  answer with the reference. Catches wrong answers that still contain the right
  words, which fact match alone can't.
- Groundedness: an LLM judge checks every claim in the answer against the
  retrieved passages. This is the "does it stick to the sources" number.
- Refusals: unanswerable questions should get the no-answer reply;
  answerable ones should not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lead_assistant.rag import NO_ANSWER, RagPipeline, make_structured_llm

EVAL_DIR = Path(__file__).resolve().parent


class Verdict(BaseModel):
    grounded: bool = Field(description="True if every factual claim in the answer is supported by the passages.")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims not supported by the passages.")


class VerdictWithReference(Verdict):
    correct: bool = Field(description="True if the answer agrees with the reference answer on every point the "
                                      "question asks about (same yes/no, numbers, dates and conditions).")
    correctness_note: str = Field(description="One sentence: what matches or differs from the reference.")


JUDGE_PROMPT = """You check whether an answer is fully supported by the given source passages.
A claim is supported only if the passages state it or it follows directly (simple arithmetic is fine).
Saying that something is not mentioned in the passages counts as supported.
Ignore citation markers like [1]. Be strict about numbers, dates and policies.
If a reference answer is given, also judge correctness: the answer must reach the same conclusion
and the same key numbers and dates. Extra correct detail is fine; wording does not matter."""


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return NO_ANSWER.lower().rstrip(".") in a or "don't have that information" in a


def judge(judge_llm, question: str, context: str, answer: str, reference: str | None = None) -> Verdict:
    ref = f"\n\nReference answer:\n{reference}" if reference else ""
    return judge_llm.invoke([
        SystemMessage(JUDGE_PROMPT),
        HumanMessage(f"Passages:\n{context}\n\nQuestion: {question}{ref}\n\nAnswer to check:\n{answer}"),
    ])


def contains_all(answer: str, items: list) -> bool:
    """Each item is a string or a list of accepted alternatives."""
    a = answer.lower()
    return all(any(alt.lower() in a for alt in (it if isinstance(it, list) else [it])) for it in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="chunks to retrieve (default: TOP_K setting)")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl",
                    help="question set (default: eval/questions.jsonl; harder set: eval/questions_hard.jsonl)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM groundedness judge")
    ap.add_argument("--judge-model", default=None,
                    help="model for the groundedness judge (default: CHAT_MODEL). A different model "
                         "has its own free-tier quota and avoids the model grading itself.")
    ap.add_argument("--resume", type=Path, default=None,
                    help="earlier results JSON: keep its successful rows, run only the missing or "
                         "errored questions (the free tier allows 20 requests/day per model)")
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    rag = RagPipeline()
    k = args.k or rag.settings.top_k
    judge_model = args.judge_model or rag.settings.chat_model
    # Claude judge at effort "medium" (answers use the lower CLAUDE_EFFORT): grading
    # numbers and policies strictly is worth a little more reasoning. Ignored for Gemini.
    judge_llm = make_structured_llm(rag.settings, Verdict, judge_model, effort="medium")
    ref_judge_llm = make_structured_llm(rag.settings, VerdictWithReference, judge_model, effort="medium")

    kept: dict[str, dict] = {}
    if args.resume:
        prev = json.loads(args.resume.read_text(encoding="utf-8"))
        ps = prev["summary"]
        mine = {"chat_model": rag.settings.chat_model, "k": k, "questions": args.questions.name,
                "judge_model": None if args.no_judge else judge_model}
        for key, val in mine.items():
            # Runs from before --questions existed always used questions.jsonl.
            if ps.get(key, "questions.jsonl" if key == "questions" else None) != val:  # mixing models would make the combined score meaningless
                raise SystemExit(f"--resume: {key} differs (earlier run {ps.get(key)!r}, now {val!r})")
        kept = {r["id"]: r for r in prev["rows"] if "error" not in r and "judge_error" not in r}
        print(f"Resuming: keeping {len(kept)} rows from {args.resume.name}", file=sys.stderr)

    rows = []
    for q in questions:
        if q["id"] in kept:
            rows.append(kept[q["id"]])
            continue
        t0 = time.perf_counter()
        try:
            result = rag.ask(q["question"], k=k)
        except Exception as e:  # e.g. Gemini 503; record it and keep going
            print(f"{q['id']}  ERROR {type(e).__name__}: {str(e)[:150]}", file=sys.stderr, flush=True)
            rows.append({"id": q["id"], "question": q["question"], "error": f"{type(e).__name__}: {e}"})
            continue
        latency = time.perf_counter() - t0

        expected = set(q["expected_sources"])
        expect_refusal = q.get("expect_refusal", False)
        retrieved_docs = list(dict.fromkeys(p.source for p in result.retrieved))  # dedupe, keep rank order
        rank = next((i for i, s in enumerate(retrieved_docs, 1) if s in expected), None)
        needs = q.get("needs_sources")
        reference = q.get("reference")
        refused = is_refusal(result.answer)
        cited = {p.source for p in result.citations}

        row = {
            "id": q["id"],
            "category": q.get("category", "basic"),
            "question": q["question"],
            "answer": result.answer,
            "expect_refusal": expect_refusal,
            "refused": refused,
            "retrieved_docs": retrieved_docs,
            "cited_docs": sorted(cited),
            "first_hit_rank": rank,
            "citation_ok": bool(cited & expected) if expected else None,
            "all_sources_retrieved": all(set(g) & set(retrieved_docs) for g in needs) if needs else None,
            "facts_ok": contains_all(result.answer, q["must_contain"]) if q["must_contain"] else None,
            "latency_s": round(latency, 2),
        }
        if reference and refused:
            # A refusal is correct exactly when one was expected; no judge call needed.
            row["correct"] = expect_refusal
        if not args.no_judge and not refused:
            try:
                llm = ref_judge_llm if reference else judge_llm
                v = judge(llm, q["question"], rag.build_context(result.retrieved), result.answer, reference)
                row["grounded"] = v.grounded
                row["unsupported_claims"] = v.unsupported_claims
                if reference:
                    row["correct"] = v.correct
                    row["correctness_note"] = v.correctness_note
            except Exception as e:
                row["judge_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

        status = "REFUSED" if refused else "answered"
        print(f"{q['id']}  rank={rank}  {status:8}  cited={row['cited_docs']}  {latency:.1f}s",
              file=sys.stderr, flush=True)

    # ---------- aggregate ----------
    errors = [r for r in rows if "error" in r]
    rows_ok = [r for r in rows if "error" not in r]
    answerable = [r for r in rows_ok if not r["expect_refusal"]]
    unanswerable = [r for r in rows_ok if r["expect_refusal"]]

    def pct(n, d):
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"

    hit1 = sum(1 for r in answerable if r["first_hit_rank"] == 1)
    hitk = sum(1 for r in answerable if r["first_hit_rank"] is not None)
    mrr = sum(1 / r["first_hit_rank"] for r in answerable if r["first_hit_rank"]) / max(len(answerable), 1)
    cit = [r for r in answerable if r["citation_ok"] is not None and not r["refused"]]
    facts = [r for r in answerable if r["facts_ok"] is not None]
    judged = [r for r in rows_ok if "grounded" in r]
    multi = [r for r in rows_ok if r.get("all_sources_retrieved") is not None]
    graded = [r for r in rows_ok if "correct" in r]

    summary = {
        "questions": args.questions.name,
        "chat_model": rag.settings.chat_model,
        "embedding_model": rag.settings.embedding_model,
        "judge_model": None if args.no_judge else judge_model,
        "k": k,
        "n_questions": len(rows),
        "errors": len(errors),
        "retrieval_hit@1": pct(hit1, len(answerable)),
        f"retrieval_hit@{k}": pct(hitk, len(answerable)),
        "retrieval_mrr": round(mrr, 3),
        "all_sources_retrieved": pct(sum(r["all_sources_retrieved"] for r in multi), len(multi)) if multi else "n/a",
        "citation_accuracy": pct(sum(r["citation_ok"] for r in cit), len(cit)),
        "fact_match": pct(sum(r["facts_ok"] for r in facts), len(facts)),
        "correctness": pct(sum(r["correct"] for r in graded), len(graded)) if graded else "n/a",
        "groundedness": pct(sum(r["grounded"] for r in judged), len(judged)) if judged else "skipped",
        "correct_refusals": pct(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "false_refusals": pct(sum(r["refused"] for r in answerable), len(answerable)),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows_ok) / max(len(rows_ok), 1), 2),
    }
    # Correctness per category, so a weak spot (e.g. near-miss traps) isn't averaged away.
    for cat in sorted({r["category"] for r in graded}):
        in_cat = [r for r in graded if r["category"] == cat]
        summary[f"correct[{cat}]"] = pct(sum(r["correct"] for r in in_cat), len(in_cat))

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for key, val in summary.items():
        print(f"{key:24} {val}")
    failures = [r for r in rows_ok if r.get("grounded") is False or r["facts_ok"] is False
                or r["citation_ok"] is False or r["refused"] != r["expect_refusal"]
                or r.get("correct") is False or r.get("all_sources_retrieved") is False]
    if failures:
        print("\n=== Needs a look ===")
        for r in failures:
            print(f"- {r['id']} {r['question']}\n    answer: {r['answer']}")
            if r.get("unsupported_claims"):
                print(f"    unsupported: {r['unsupported_claims']}")
            if r.get("correct") is False:
                print(f"    judge: {r.get('correctness_note', 'refused but an answer was expected')}")
            if r.get("all_sources_retrieved") is False:
                print(f"    retrieved only: {r['retrieved_docs']}")
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
