"""FastAPI app exposing the RAG pipeline.

Run:  uvicorn lead_assistant.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request
from langchain_google_genai.chat_models import GoogleGenerativeAIError, GoogleRateLimitError
from pydantic import BaseModel, Field

from lead_assistant.rag import Passage, RagPipeline


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the pipeline once at startup, not per request.
    app.state.rag = RagPipeline()
    yield


app = FastAPI(title="AI Lead Assistant", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    # Sync def: FastAPI runs it in a thread pool, so the blocking Gemini call
    # doesn't stall other requests.
    try:
        result = request.app.state.rag.ask(body.question, k=body.top_k)
    # Let callers (n8n in Week 3) see a retryable status, not a bare 500.
    # Order matters: the rate-limit classes subclass the general ones below them.
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "15")
        raise HTTPException(429, "LLM rate limit hit, retry shortly.", headers={"Retry-After": retry_after})
    except GoogleRateLimitError:
        raise HTTPException(429, "LLM quota exceeded, retry shortly.", headers={"Retry-After": "15"})
    except (anthropic.APIStatusError, anthropic.APIConnectionError, GoogleGenerativeAIError) as e:
        # Includes Gemini embedding errors (e.g. 403 when the project has no billing).
        raise HTTPException(503, f"LLM unavailable: {type(e).__name__}")
    return AskResponse(
        answer=result.answer,
        sources=[Source.from_passage(p) for p in result.citations],
        retrieved=[Source.from_passage(p) for p in result.retrieved],
    )
