# AI Lead Assistant for Zoho CRM

A new lead arrives in Zoho CRM. n8n sends it to a Python AI agent, which answers the lead's question from course documents, scores the lead and drafts a reply. A person approves the draft, then n8n writes everything back to Zoho.

Roadmap: Week 1, RAG core (this repo today). Week 2, a LangGraph agent with lead scoring and prompt-injection guardrails. Week 3, n8n orchestration with a human approval step and Zoho CRM write-back.

**Status:** Week 1, the RAG core, is done. `/ask` answers with citations, and the 25-question eval scores 25/25. Answers come from Claude Haiku 5.5; embeddings run locally (free) with `BAAI/bge-small-en-v1.5`.

All course documents in `data/docs/` describe a made-up academy ("Brightwater Digital Academy"). No real student or academy data is used.

## How the RAG core works

```
data/docs/*.md|.txt|.pdf
   │  load_documents()      one Document per file (per page for PDFs), with source + title
   ▼
   │  split_documents()     RecursiveCharacterTextSplitter, 1000 chars, 150 overlap,
   │                        splits on headings → paragraphs → lines; title prefixed to each chunk
   ▼
   │  local embeddings      BAAI/bge-small-en-v1.5 (sentence-transformers, CPU, 384-dim, normalised)
   ▼
ChromaDB (chroma_db/, cosine distance, records which embedding model built it)

POST /ask {question}
   │  embed question locally (bge query prefix) → top-k nearest chunks
   │  numbered passages [1]..[k] + strict prompt → chat model (Claude Haiku 5.5 by default)
   ▼
{answer with [n] citations, sources: cited passages, retrieved: all passages}
```

## Models and keys

| Role | Model | Key (environment variable) |
|---|---|---|
| Embeddings (ingest + every question) | `BAAI/bge-small-en-v1.5`, local | none |
| Answers | `claude-haiku-5-5` (default) | `LEAD_ASSISTANT_CLAUDE_KEY` |
| Eval judge | same as answers unless `--judge-model` is given | same |

Set `CHAT_MODEL` to switch the answer model. Names starting with `claude` use Anthropic, anything else uses Gemini (e.g. `CHAT_MODEL=gemini-3.5-flash`, which needs `GEMINI_API_KEY`). Keys are read as secrets and never logged.

Embeddings, from `config.py`:
- **Download:** the model (~130 MB) downloads once into the Hugging Face cache on first use, then runs offline on CPU. Re-indexing the 28 chunks takes about 35s.
- **Query prefix:** bge expects questions, not documents, to start with an instruction ("Represent this sentence for searching relevant passages: "). `store.py` adds it to questions only.
- **Switching models:** set `EMBEDDING_MODEL`. Any sentence-transformers name works (e.g. `sentence-transformers/all-MiniLM-L6-v2`), and `models/gemini-embedding-2` uses the Gemini API. The index records the model that built it, and querying with a different one fails with `IndexMismatchError` until you re-run `python -m lead_assistant.ingest`. Vectors from different models aren't comparable.

Claude settings, from `config.py`:
- **Effort:** `CLAUDE_EFFORT=low` for answers. Haiku 5.5 always thinks, and effort sets how much. The eval judge uses `medium` so it grades numbers and policies more strictly.
- **No `temperature`:** Haiku 5.5 rejects non-default sampling values with a 400.
- **Max tokens:** `CLAUDE_MAX_TOKENS=16000`. Thinking tokens count against it.
- **Judge output:** native JSON-schema structured output (`output_config.format`), not a forced tool call.

Prompt rules, from `rag.py`:
- **Sources only:** answer only from the numbered passages and cite them. Never guess fees, dates or policies; reply "I don't have that information in the course documents." when the passages don't answer the question.
- **Complete-list exception:** if a passage gives the full set of options (payment methods, programs, campuses) and the item asked about isn't in it, answer "not offered" or "not accepted", name the options that are, and cite the list. Lists given only as examples ("such as") don't count.

| File | What it does |
|---|---|
| `src/lead_assistant/config.py` | Settings from env / `.env` |
| `src/lead_assistant/store.py` | Chroma + embeddings setup (local or Gemini), index/model mismatch guard |
| `src/lead_assistant/ingest.py` | Load → chunk → embed → store |
| `src/lead_assistant/rag.py` | Retrieve, prompt, parse citations; picks Claude or Gemini by model name |
| `src/lead_assistant/api.py` | FastAPI `/ask` and `/health` |
| `eval/questions.jsonl` | 25-question test set (21 answerable, 3 unanswerable, 1 edge case) |
| `eval/run_eval.py` | Retrieval, citation, fact-match, groundedness and refusal scores |
| `eval/retrieval_eval.py` | Retrieval-only scores (embeddings only, no chat model) |
| `tests/` | Offline unit tests: loading, chunking, model selection, Claude request settings, index guard |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # or set LEAD_ASSISTANT_CLAUDE_KEY as an environment variable
python -m lead_assistant.ingest   # first run downloads the embedding model (~130 MB)
```

## Run

```powershell
python -m lead_assistant.ingest            # build the index (rebuilds from scratch each time)
uvicorn lead_assistant.api:app --reload    # http://127.0.0.1:8000/docs
```

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/ask -ContentType application/json `
  -Body '{"question": "Can I pay the Data Analytics fee in instalments?"}'
```

## Test

```powershell
pytest                          # offline unit tests, no API calls
python eval/retrieval_eval.py   # retrieval only: local embeddings, free
python eval/run_eval.py         # full eval: local retrieval + Claude answers and judge (~47 Claude calls)
```

## Results

Full per-question output is in `eval/results/` (git-ignored).

### Full eval — 2026-11-06: 25/25

Answers `claude-haiku-5-5` (effort low), judge `claude-haiku-5-5` (effort medium), embeddings `BAAI/bge-small-en-v1.5`, k=4. One run of all 25 questions, no errors (`eval-20261106-181520.json`).

| Metric | Score |
|---|---|
| Retrieval hit@1 | 22/22 (100%) |
| Retrieval hit@4 | 22/22 (100%) |
| Retrieval MRR | 1.000 |
| Citation accuracy | 22/22 (100%) |
| Fact match | 21/21 (100%) |
| Groundedness (LLM judge) | 22/22 (100%) |
| Correct refusals (q22–q24) | 3/3 (100%) |
| False refusals | 0/22 |
| Average latency per question | 1.45 s |

**q25, the complete-list fix.** Asked "Do you accept payment in cryptocurrency like Bitcoin?", the answer is now: *"The payment methods listed are UPI, debit or credit card, net banking and NEFT/RTGS bank transfer. We do not accept cash or cheques [1]. Cryptocurrency such as Bitcoin is not among the accepted methods, so it is not offered [1]."* Before the prompt change, Haiku refused this question in a smoke test, which would have counted as a false refusal.

**Refusals still hold.** The cybersecurity course (q22), 2028 fees (q23) and CEO (q24) questions all got the no-answer reply. q22 is a case to watch: the About page does list all five programs, so if retrieval ever returns that chunk, the complete-list rule would correctly answer "not offered", and q22 would need relabelling as answerable. With bge it isn't in the top 4.

**How far to trust 25/25:**
- **Saturated.** The test set is small and was written alongside the documents, so a perfect score shows the pipeline works but can't tell close variants apart. Harder questions are needed (paraphrases, multi-document answers, near-miss traps) before comparing models or prompts.
- **Self-graded.** Haiku judges Haiku. Pass `--judge-model` with a different model for an independent groundedness check.

### Retrieval: local bge vs Gemini (`eval/retrieval_eval.py`, k=4)

| Metric | bge-small-en-v1.5 (local) | gemini-embedding-2 (API) |
|---|---|---|
| hit@1 | **22/22 (100%)** | 21/22 (95%) |
| hit@4 | 22/22 (100%) | 22/22 (100%) |
| MRR | **1.000** | 0.977 |
| Top similarity, answerable | 0.605 – 0.810 | 0.652 – 0.838 |
| Top similarity, unanswerable | 0.548 – 0.716 | 0.635 – 0.744 |

The free local model is at least as good as the API model on this set. It fixed Gemini's only miss: q12 ("What salary did data analytics graduates get?") now ranks the placement page first instead of the course page.

The similarity ranges still overlap for both models, so a threshold can't decide when to refuse. With bge, "What are the fees for batches starting in 2028?" scores 0.716 against the 2027 fee sheet, higher than 10 of the 22 answerable questions. Only the prompt's refusal rule stops a made-up 2028 fee. This becomes a Week 2 guardrail test.

### Earlier runs (Gemini answers and embeddings)

Before the switch, `gemini-3.5-flash` answers with `gemini-3.1-flash-lite` judging and `gemini-embedding-2` retrieval passed all 24 questions it could score: q25 never ran because of the Gemini free-tier daily limit. The Gemini path still works if you set `CHAT_MODEL` / `EMBEDDING_MODEL` and a `GEMINI_API_KEY` whose project has the free tier or billing. On the free tier, set `LLM_REQUESTS_PER_MINUTE=4` (limit: 5 requests/min and 20/day per model) and continue across days with `run_eval.py --resume <results.json>`, which refuses to mix runs with different models or k.
