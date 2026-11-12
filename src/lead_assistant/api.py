"""FastAPI app exposing the RAG pipeline.

Run:  uvicorn lead_assistant.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager, contextmanager

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from langchain_google_genai.chat_models import GoogleGenerativeAIError, GoogleRateLimitError
from pydantic import BaseModel, Field

from lead_assistant.agent import LeadAgent
from lead_assistant.audit import find_run, record_decision
from lead_assistant.config import get_settings
from lead_assistant.leads import LeadNotFound
from lead_assistant.rag import Passage, RagPipeline
from lead_assistant.schemas import AgentResult, HumanDecision


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    n: int
    source: str
    title: str
    page: int | None
    chunk_id: str
    score: float
    snippet: str

    @classmethod
    def from_passage(cls, p: Passage, snippet_len: int = 300) -> "Source":
        snippet = p.text if len(p.text) <= snippet_len else p.text[:snippet_len].rstrip() + "..."
        return cls(n=p.n, source=p.source, title=p.title, page=p.page,
                   chunk_id=p.chunk_id, score=p.score, snippet=snippet)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(description="Passages the answer cites.")
    retrieved: list[Source] = Field(description="All passages given to the model.")


class ProcessLeadRequest(BaseModel):
    lead_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the pipeline and agent once at startup, not per request.
    app.state.rag = RagPipeline()
    app.state.agent = LeadAgent(rag=app.state.rag)
    yield


app = FastAPI(title="AI Lead Assistant", version="0.3.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Every endpoint except /health needs the shared secret once LEAD_ASSISTANT_API_KEY is set.
    compare_digest takes the same time whether the first or last character is wrong,
    so the key can't be guessed one character at a time from response timings."""
    expected = get_settings().api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        raise HTTPException(401, "Missing or wrong X-API-Key header.")


protected = [Depends(require_api_key)]


@contextmanager
def llm_errors_as_http():
    """Let callers (n8n in Week 3) see a retryable status, not a bare 500.
    Order matters: the rate-limit classes subclass the general ones below them."""
    try:
        yield
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "15")
        raise HTTPException(429, "LLM rate limit hit, retry shortly.", headers={"Retry-After": retry_after})
    except GoogleRateLimitError:
        raise HTTPException(429, "LLM quota exceeded, retry shortly.", headers={"Retry-After": "15"})
    except (anthropic.APIStatusError, anthropic.APIConnectionError, GoogleGenerativeAIError) as e:
        # Includes Gemini embedding errors (e.g. 403 when the project has no billing).
        raise HTTPException(503, f"LLM unavailable: {type(e).__name__}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/process-lead", response_model=AgentResult, dependencies=protected)
def process_lead(body: ProcessLeadRequest, request: Request) -> AgentResult:
    """Score a lead and draft a reply. Reads the CRM but writes nothing: in Week 3,
    n8n shows this result to a person, and only their approval updates Zoho."""
    try:
        with llm_errors_as_http():
            return request.app.state.agent.process(body.lead_id)
    except LeadNotFound:
        raise HTTPException(404, f"Lead {body.lead_id} not found.")


@app.post("/agent/decision", dependencies=protected)
def agent_decision(body: HumanDecision) -> dict:
    """n8n calls this after the approver clicks, so the audit log holds both what the
    agent proposed and what a person decided. It writes nothing to Zoho: n8n does that."""
    path = get_settings().audit_log_path
    run = find_run(path, body.run_id)
    if run is None or run.get("lead_id") != body.lead_id:
        # Catches a mis-wired workflow recording a decision against the wrong run.
        raise HTTPException(404, f"No agent run {body.run_id} for lead {body.lead_id}.")
    if body.written_to_crm and body.decision != "approved":
        raise HTTPException(422, "Only an approved run can be written to the CRM.")
    record_decision(path, body.model_dump())
    return {"recorded": True, "run_id": body.run_id}


@app.post("/ask", response_model=AskResponse, dependencies=protected)
def ask(body: AskRequest, request: Request) -> AskResponse:
    # Sync def: FastAPI runs it in a thread pool, so the blocking LLM call
    # doesn't stall other requests.
    with llm_errors_as_http():
        result = request.app.state.rag.ask(body.question, k=body.top_k)
    return AskResponse(
        answer=result.answer,
        sources=[Source.from_passage(p) for p in result.citations],
        retrieved=[Source.from_passage(p) for p in result.retrieved],
    )
