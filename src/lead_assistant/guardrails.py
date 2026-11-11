"""Code-level guardrails: checks that don't trust the model.

Prompts tell the model to treat the lead's message as data and to use only the
documents. These functions check that it actually did. A failed check means the
draft is retried once with the errors, then replaced by a safe fallback.
"""

from __future__ import annotations

import re

from lead_assistant.schemas import DraftOutput

MAX_SUBJECT, MAX_BODY = 120, 3000
INTERNAL_NAMES = ("get_lead", "search_docs", "score_lead", "draft_reply", "system prompt")

# Phrases typical of prompt injection. Matching one does NOT block the lead (real
# people write "ignore my last email"); it adds a flag so the human reviewer looks
# closer. The real defences are the checks below and the narrow tools.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the |your )?(previous|prior|above) (instructions|rules)",
    r"ignore your (rules|instructions)",
    r"\b(system|admin|developer) (mode|override|prompt|note)\b",
    r"\byou are now\b",
    r"(print|reveal|show|repeat) (me )?(your|the) (full )?(system )?(prompt|instructions)",
    r"\b(set|change) the (lead )?(score|band)\b",
    r"\buse (get_lead|search_docs|score_lead|draft_reply)\b",
]

_CITATION = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def detect_injection(text: str) -> list[str]:
    """Return the injection patterns found in `text` (empty list if none)."""
    t = text.lower()
    return [p for p in _INJECTION_PATTERNS if re.search(p, t)]


def numbers_in(text: str) -> set[str]:
    """Numbers in `text`, commas removed, so 1,46,320 and 146,320 compare equal.
    Single digits are skipped: they're counts ("3 months") that docs often spell out."""
    text = _CITATION.sub(" ", text)
    out = set()
    for m in _NUMBER.findall(text):
        n = m.replace(",", "").rstrip(".")
        if len(n.replace(".", "")) >= 2:
            out.add(n)
    return out


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip(".,;:!?)")


def validate_draft(draft: DraftOutput, passages: list[dict], computed: list[str] = ()) -> list[str]:
    """Check a drafted reply against the passages it may use. `computed` holds results
    of the calculator tools, which only accept documented operands. Returns a list of
    problems; an empty list means the draft passed."""
    errors: list[str] = []
    numbers = {p["n"] for p in passages}
    source_text = "\n".join(p["text"] for p in passages)

    if len(draft.subject) > MAX_SUBJECT:
        errors.append(f"subject is longer than {MAX_SUBJECT} characters")
    if len(draft.body) > MAX_BODY:
        errors.append(f"body is longer than {MAX_BODY} characters")

    # 1. Citations must point at passages that exist.
    cited = {int(n) for n in _CITATION.findall(draft.body)}
    if bad := sorted(cited - numbers):
        errors.append(f"cites passages that don't exist: {bad}")

    # 2. Every figure (fee, percentage, date, year, phone number) must appear in a
    #    passage or be the result of a verified calculation. This is what stops a
    #    made-up or injected fee. Arithmetic the model does in its head (rather than
    #    through the calculator tools) fails here.
    allowed = numbers_in(source_text) | numbers_in(" ".join(computed))
    if unsupported := sorted(numbers_in(draft.body) - allowed):
        errors.append(f"figures not found in the passages or verified calculations: {unsupported}")
    if numbers_in(draft.body) and not cited:
        errors.append("states figures but cites no passage")

    # 3. Links and email addresses only if a passage contains them (the academy's own).
    for url in _URL.findall(draft.body):
        if _strip_trailing_punct(url) not in source_text:
            errors.append(f"link not from the documents: {_strip_trailing_punct(url)}")
    for email in _EMAIL.findall(draft.body):
        if email not in source_text:
            errors.append(f"email address not from the documents: {email}")

    # 4. Don't reveal internals.
    body_lower = (draft.subject + " " + draft.body).lower()
    for name in INTERNAL_NAMES:
        if name in body_lower:
            errors.append(f"mentions internal detail: {name!r}")
    return errors
