"""Data contracts for the Week 2 agent.

Every structure that crosses a boundary is a Pydantic model, so malformed data
is rejected where it enters instead of causing a strange failure later:
- Lead: what get_lead returns (from Zoho CRM or the sample file).
- tool argument models: what the agent LLM may pass to each tool.
- IntentAssessment / DraftOutput: what the scoring and drafting LLM calls must return.
- AgentResult: what the API returns to n8n in Week 3.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROGRAMS = (
    "Data Analytics Pro",
    "Full Stack Web Development",
    "Digital Marketing Accelerator",
    "UI/UX Design Professional",
    "Python Foundations",
)

Qualification = Literal["Class 10", "Class 12", "Undergraduate (final year)", "Undergraduate",
                        "Graduate", "Postgraduate"]


class Lead(BaseModel):
    """A lead as stored in Zoho CRM. Aliases are the Zoho API field names.

    Standard Zoho fields: id, First_Name, Last_Name, Email, Phone, Lead_Source,
    City, Description. Program_Interest, Qualification and Planned_Start are
    custom fields you create in Zoho CRM in Week 3, with exactly these API names.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    first_name: str | None = Field(default=None, alias="First_Name")
    last_name: str = Field(alias="Last_Name")  # mandatory in Zoho
    email: str | None = Field(default=None, alias="Email")
    phone: str | None = Field(default=None, alias="Phone")
    lead_source: str | None = Field(default=None, alias="Lead_Source")
    city: str | None = Field(default=None, alias="City")
    program_interest: str | None = Field(default=None, alias="Program_Interest")
    qualification: Qualification | None = Field(default=None, alias="Qualification")
    planned_start: date | None = Field(default=None, alias="Planned_Start")
    # The lead's own words. Untrusted: anyone on the internet can type anything here.
    message: str = Field(default="", alias="Description", max_length=4000)


# ---------- tool arguments (validated before any tool runs) ----------

class GetLeadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(min_length=1, max_length=40)


class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=300)


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200,
                            description="Arithmetic on figures from the passages, e.g. '1,24,000 * 15%'.")
    purpose: str = Field(max_length=120, description="What the result means, e.g. 'merit scholarship saving'.")


class ShiftDateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date = Field(description="A date written in the passages, as YYYY-MM-DD.")
    days: int = Field(ge=-366, le=366, description="Days to add; negative for 'N days before'.")
    purpose: str = Field(max_length=120, description="What the result means, e.g. 'application deadline'.")


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- LLM outputs ----------
# These schemas are sent to Claude as JSON schemas (structured outputs), which don't
# support length constraints (maxLength, maxItems). Limits are enforced in code instead
# (guardrails.validate_draft, scoring.score_lead).

class IntentAssessment(BaseModel):
    """The LLM half of lead scoring: how ready to enrol the message sounds."""
    intent: Literal["high", "medium", "low"] = Field(
        description="high: concrete plans (fees, start dates, how to enrol); medium: exploring options; "
                    "low: vague, off-topic, spam, or no real question.")
    signals: list[str] = Field(description="Up to 3 short phrases from the message that justify the rating.")


class DraftOutput(BaseModel):
    """What the drafting LLM must return. Checked again in code after parsing."""
    subject: str = Field(description="Email subject line, under 100 characters.")
    body: str = Field(description="Plain-text email body with [n] citations after factual sentences.")
    answered_questions: list[str] = Field(description="The lead's questions this reply answers from the passages.")
    unanswered_questions: list[str] = Field(
        description="The lead's questions the passages don't answer; a counsellor will follow up on these.")


# ---------- final result (the API contract) ----------

class ScoreItem(BaseModel):
    rule: str
    points: int


class LeadScore(BaseModel):
    score: int = Field(ge=0, le=100)
    band: Literal["hot", "warm", "cold"]
    breakdown: list[ScoreItem]
    intent: Literal["high", "medium", "low"]
    intent_signals: list[str]


class Citation(BaseModel):
    n: int
    source: str
    title: str


class Reply(BaseModel):
    subject: str
    body: str
    citations: list[Citation]


class Calculation(BaseModel):
    """A figure the reply may state because code computed it from documented figures."""
    tool: Literal["calculate", "shift_date"]
    input: str
    purpose: str
    result: str


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    lead_id: str
    score: LeadScore
    reply: Reply
    unanswered_questions: list[str]
    # How each computed figure in the reply was worked out, for the reviewer to check.
    calculations: list[Calculation] = []
    # Anything a human reviewer should know: possible_prompt_injection, fallback_reply,
    # draft_retried, not_eligible, forced_step:<tool>, ...
    flags: list[str]
    tools_called: list[str]
    # Week 3: nothing is written to Zoho without a human yes, whatever this says.
    # It marks drafts that need a closer look than usual.
    needs_attention: bool
