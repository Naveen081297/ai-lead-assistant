"""Run the harder question set through the Week 2 agent instead of /ask.

Run:  python eval/agent_hard_eval.py [--judge-model claude-sonnet-5-5]

Each question becomes a minimal made-up lead whose message is the question, so
the agent does what it does for a real lead: plans its own searches, scores the
lead and drafts a reply. Compared with run_eval.py on the same file:
- Retrieval is judged over every passage the agent found, across all its searches.
- Every reply is judged, including "a counsellor will follow up" replies (run_eval
  skips refusals, which let a wrong claim inside a refusal go unchecked).

Note: the agent's draft validator only accepts figures written in a passage or
returned by the calculator tools (calculate, shift_date), so computed answers
(h03-h05) depend on the agent using those tools rather than its own arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from run_eval import EVAL_DIR, VerdictWithReference, contains_all, judge  # same folder

from lead_assistant.agent import LeadAgent
from lead_assistant.leads import LeadNotFound
from lead_assistant.rag import make_structured_llm
from lead_assistant.schemas import Lead


class QuestionLeads:
    """Lead source that turns each eval question into a lead with no other details."""

    def __init__(self, questions: list[dict]):
        self._by_id = {f"Q-{q['id']}": q for q in questions}

    def get(self, lead_id: str) -> Lead:
        q = self._by_id.get(lead_id)
        if q is None:
            raise LeadNotFound(lead_id)
        return Lead(id=lead_id, First_Name="Alex", Last_Name="Eval", Lead_Source="Website",
                    Description=q["question"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions_hard.jsonl")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    agent = LeadAgent(leads=QuestionLeads(questions))
    judge_llm = make_structured_llm(agent.settings, VerdictWithReference, args.judge_model, effort="medium")

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            result = agent.process(f"Q-{q['id']}")
        except Exception as e:
            print(f"{q['id']}  ERROR {type(e).__name__}: {str(e)[:150]}", file=sys.stderr, flush=True)
            rows.append({"id": q["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        latency = time.perf_counter() - t0

        # The passages the agent found are in the audit log; read this run's record.
        record = next(json.loads(l) for l in reversed(
            agent.settings.audit_log_path.read_text(encoding="utf-8").splitlines())
            if json.loads(l)["run_id"] == result.run_id)
        searches = [c["args"].get("query") for e in record["events"] if e["kind"] == "agent"
                    for c in e["tool_calls"] if c["name"] == "search_docs"]
        context, found_docs = _passages_for(record, agent)
        needs = q.get("needs_sources") or ([q["expected_sources"]] if q["expected_sources"] else None)
        reply = result.reply.body

        row = {
            "id": q["id"], "category": q["category"], "question": q["question"], "reply": reply,
            "searches": searches, "found_docs": found_docs,
            "all_sources_found": all(set(g) & set(found_docs) for g in needs) if needs else None,
            "facts_ok": contains_all(reply, q["must_contain"]) if q["must_contain"] else None,
            "flags": result.flags, "unanswered": result.unanswered_questions,
            "latency_s": round(latency, 1), "run_id": result.run_id,
        }
        try:
            v = judge(judge_llm, q["question"], context, reply, q["reference"])
            row.update(correct=v.correct, grounded=v.grounded, note=v.correctness_note,
                       unsupported_claims=v.unsupported_claims)
        except Exception as e:
            row["judge_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"{q['id']}  correct={row.get('correct')}  sources={row['all_sources_found']}  "
              f"searches={len(searches)}  {latency:.0f}s", file=sys.stderr, flush=True)

    ok = [r for r in rows if "error" not in r and "judge_error" not in r]

    def pct(n, d):
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"

    with_needs = [r for r in ok if r["all_sources_found"] is not None]
    summary = {
        "questions": args.questions.name, "mode": "agent", "chat_model": agent.settings.chat_model,
        "judge_model": args.judge_model, "n_questions": len(rows), "errors": len(rows) - len(ok),
        "correctness": pct(sum(r["correct"] for r in ok), len(ok)),
        "groundedness": pct(sum(r["grounded"] for r in ok), len(ok)),
        "all_sources_found": pct(sum(r["all_sources_found"] for r in with_needs), len(with_needs)),
        "avg_searches": round(sum(len(r["searches"]) for r in ok) / max(len(ok), 1), 1),
        "avg_latency_s": round(sum(r["latency_s"] for r in ok) / max(len(ok), 1), 1),
    }
    for cat in sorted({r["category"] for r in ok}):
        in_cat = [r for r in ok if r["category"] == cat]
        summary[f"correct[{cat}]"] = pct(sum(r["correct"] for r in in_cat), len(in_cat))

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"agent-hard-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:22} {v}")
    print("\n=== Per question ===")
    for r in rows:
        if "error" in r:
            print(f"- {r['id']} ERROR {r['error']}")
            continue
        print(f"- {r['id']} [{r['category']}] correct={r.get('correct')} grounded={r.get('grounded')} "
              f"sources={r['all_sources_found']} flags={r['flags']}\n    searches: {r['searches']}\n"
              f"    judge: {r.get('note') or r.get('judge_error')}")
        if r.get("unsupported_claims"):
            print(f"    unsupported: {r['unsupported_claims']}")
    print(f"\nFull results: {out}")


def _passages_for(record: dict, agent: LeadAgent) -> tuple[str, list[str]]:
    """Re-run the agent's searches, in order, to rebuild the numbered passages its
    draft was written from, plus their source files. Retrieval is deterministic
    (local embeddings, same index), so this matches the run. The audit log alone
    isn't enough: it truncates tool results."""
    seen, blocks, sources = set(), [], set()
    for e in record["events"]:
        if e["kind"] != "agent":
            continue
        for c in e["tool_calls"]:
            if c["name"] != "search_docs":
                continue
            for p in agent.rag.retrieve(c["args"]["query"]):
                sources.add(p.source)
                if p.chunk_id not in seen:
                    seen.add(p.chunk_id)
                    blocks.append(f"[{len(blocks) + 1}] (source: {p.source})\n{p.text}")
    return "\n\n".join(blocks) or "(no passages were found)", sorted(sources)


if __name__ == "__main__":
    main()
