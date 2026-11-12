"""Settings, read from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Embeddings run locally with sentence-transformers: free, no API key, no rate
    # limits. bge-small-en-v1.5 is a small (384-dim) retrieval model that runs on CPU.
    # A name starting with "models/gemini" uses the Gemini API instead.
    # Changing this requires re-running ingest (the index records which model built it).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models retrieve better when short queries (not documents) get this prefix.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Gemini: only needed for a Gemini embedding or chat model.
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    # Chat model for answers (and the eval judge unless --judge-model is given).
    # A name starting with "claude" uses Anthropic; anything else uses Gemini.
    chat_model: str = "claude-haiku-5-5"
    # Claude: needed only when a Claude model is used.
    claude_api_key: SecretStr | None = Field(default=None, validation_alias="LEAD_ASSISTANT_CLAUDE_KEY")
    # Haiku 5.5 always runs adaptive thinking; effort sets how much (default "medium").
    # Short, source-bound answers don't need deep reasoning, so "low" here.
    claude_effort: str = "low"
    # Thinking tokens count against max_tokens, so leave headroom beyond the answer.
    claude_max_tokens: int = 16000

    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "course_docs"

    # Chunking: ~1000 characters is a few paragraphs, small enough to be specific,
    # large enough to keep a fee table or a policy clause together. The overlap
    # stops a sentence that straddles a boundary from being lost to both chunks.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # How many chunks to retrieve and hand to the LLM.
    top_k: int = 4

    # Fail fast when the provider is overloaded instead of retrying for minutes.
    llm_timeout_s: float = 45
    llm_max_retries: int = 2
    # Client-side throttle for the chat model. Needed on the Gemini free tier
    # (5 requests/min per model, so set 4). 0 = no throttle.
    llm_requests_per_minute: float = 0

    # ---------- Week 2: agent ----------
    # Hard cap on agent -> tools round trips, so a confused or hijacked model
    # can't loop forever (each round is a paid LLM call).
    agent_max_steps: int = 8
    # One JSON line per agent run: inputs, every tool call, validation results, output.
    audit_log_path: Path = PROJECT_ROOT / "logs" / "audit.jsonl"

    # Where get_lead reads leads: "sample" (data/sample_leads.json, offline) or
    # "zoho" (Zoho CRM REST API; needs the three ZOHO_* secrets below).
    lead_source: str = "sample"
    sample_leads_path: Path = PROJECT_ROOT / "data" / "sample_leads.json"
    # Zoho data centre: com (US), in (India), eu, com.au, jp, ca ...
    zoho_dc: str = "com"
    zoho_api_version: str = "v8"
    zoho_client_id: str | None = None
    zoho_client_secret: SecretStr | None = None
    zoho_refresh_token: SecretStr | None = None

    # ---------- Week 3: n8n ----------
    # Shared secret n8n sends in the X-API-Key header. Once the API is on a public
    # server, anyone who finds the URL could otherwise spend your Claude credits.
    # Unset = no check (local development only; the Docker deploy requires it).
    api_key: SecretStr | None = Field(default=None, validation_alias="LEAD_ASSISTANT_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
