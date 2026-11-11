"""Offline tests for chat-model selection (no API calls)."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from lead_assistant.config import Settings
from lead_assistant.rag import make_llm, make_structured_llm


@pytest.fixture(autouse=True)
def _ignore_local_overrides(monkeypatch):
    # These tests check the defaults in config.py, so a developer's own
    # CLAUDE_EFFORT or CHAT_MODEL environment variable must not leak in.
    for name in ("CLAUDE_EFFORT", "CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def settings(**overrides) -> Settings:
    # Pass values by env-var name; init values take priority over the real environment.
    values = {"GEMINI_API_KEY": "test-gemini", "LEAD_ASSISTANT_CLAUDE_KEY": "test-claude", **overrides}
    return Settings(**values)


def test_claude_model_uses_anthropic_without_temperature():
    llm = make_llm(settings(), "claude-haiku-5-5")
    assert isinstance(llm, ChatAnthropic)
    assert llm.temperature is None  # Haiku 5.5 rejects non-default temperature
    assert llm.output_config == {"effort": "low"}
    assert llm.anthropic_api_key.get_secret_value() == "test-claude"


def test_effort_override():
    assert make_llm(settings(), "claude-haiku-5-5", effort="medium").output_config == {"effort": "medium"}


def test_gemini_model_uses_google():
    assert isinstance(make_llm(settings(), "gemini-3.5-flash"), ChatGoogleGenerativeAI)


def test_default_chat_model_is_claude_haiku():
    assert isinstance(make_llm(settings()), ChatAnthropic)


def test_missing_claude_key_fails_clearly():
    with pytest.raises(RuntimeError, match="LEAD_ASSISTANT_CLAUDE_KEY"):
        make_llm(settings(LEAD_ASSISTANT_CLAUDE_KEY=None), "claude-haiku-5-5")


def test_secrets_not_in_repr():
    s = settings()
    assert "test-claude" not in repr(s) and "test-gemini" not in repr(s)


class Verdict(BaseModel):
    grounded: bool


def test_claude_judge_request_keeps_effort_and_uses_json_schema():
    # Inspect the request LangChain would send (private helper, but it is the
    # only way to check the merged payload without calling the API).
    chain = make_structured_llm(settings(), Verdict, "claude-haiku-5-5", effort="medium")
    binding = chain.first
    payload = binding.bound._get_request_payload([HumanMessage("hi")], **binding.kwargs)
    assert payload["output_config"]["effort"] == "medium"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert not payload.get("tools") and "tool_choice" not in payload  # no forced tool call
    assert "temperature" not in payload
