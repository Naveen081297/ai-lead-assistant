"""FastAPI app exposing the RAG pipeline.

Run:  uvicorn lead_assistant.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request
from langchain_google_genai.chat_models import GoogleGenerativeAIError, GoogleRateLimitError
from pydantic import BaseModel, Field

from lead_assistant.agent import LeadAgent
from lead_assistant.leads import LeadNotFound
from lead_assistant.rag import Passage, RagPipeline
from lead_assistant.schemas import AgentResult


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


app = FastAPI(title="AI Lead Assistant", version="0.2.0", lifespan=lifespan)


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


@app.post("/agent/process-lead", response_model=AgentResult)
def process_lead(body: ProcessLeadRequest, request: Request) -> AgentResult:
    """Score a lead and draft a reply. Reads the CRM but writes nothing: in Week 3,
    n8n shows this result to a person, and only their approval updates Zoho."""
    try:
        with llm_errors_as_http():
            return request.app.state.agent.process(body.lead_id)
    except LeadNotFound:
        raise HTTPException(404, f"Lead {body.lead_id} not found.")


@app.post("/ask", response_model=AskResponse)
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
