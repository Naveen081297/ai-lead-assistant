"""Offline tests for the Week 3 API surface n8n talks to: auth and human decisions.

TestClient is used without a `with` block, so the lifespan (which loads the
embedding model and builds the agent) never runs. No LLM or Chroma is touched.
"""

import json

import pytest
from fastapi.testclient import TestClient

from lead_assistant import api
from lead_assistant.audit import AuditRun, find_run
from lead_assistant.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = Settings(_env_file=None, LEAD_ASSISTANT_API_KEY="test-key", audit_log_path=tmp_path / "audit.jsonl")
    monkeypatch.setattr(api, "get_settings", lambda: s)
    return s


@pytest.fixture
def client(settings):
    return TestClient(api.app)


def past_run(settings, lead_id="L-1001") -> str:
    run = AuditRun(settings.audit_log_path, lead_id, "test-model")
    run.finish(output={"ok": True})
    return run.run_id


def decision(run_id, **over):
    return {"run_id": run_id, "lead_id": "L-1001", "decision": "approved", "reviewer": "admissions", **over}


# ---------- auth ----------

def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_endpoints_reject_missing_or_wrong_key(client, headers):
    for path, body in [("/ask", {"question": "What is the fee?"}),
                       ("/agent/process-lead", {"lead_id": "L-1001"}),
                       ("/agent/decision", decision("abc"))]:
        assert client.post(path, json=body, headers=headers).status_code == 401, path


def test_no_key_configured_means_no_check(tmp_path, monkeypatch):
    s = Settings(_env_file=None, audit_log_path=tmp_path / "audit.jsonl")
    monkeypatch.setattr(api, "get_settings", lambda: s)
    # Reaches validation (422 for the bad body) instead of stopping at auth.
    assert TestClient(api.app).post("/agent/decision", json={}).status_code == 422


# ---------- human decisions ----------

def test_decision_is_appended_after_the_run(client, settings):
    run_id = past_run(settings)
    r = client.post("/agent/decision", json=decision(run_id, reply_edited=True, written_to_crm=True),
                    headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    lines = [json.loads(l) for l in settings.audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[1]["kind"] == "human_decision"
    assert lines[1]["run_id"] == run_id and lines[1]["reply_edited"] is True
    # The decision line isn't mistaken for the run itself.
    assert find_run(settings.audit_log_path, run_id)["events"] == []


def test_decision_for_unknown_run_or_other_lead_is_refused(client, settings):
    run_id = past_run(settings, lead_id="L-1002")
    h = {"X-API-Key": "test-key"}
    assert client.post("/agent/decision", json=decision("nosuchrun"), headers=h).status_code == 404
    assert client.post("/agent/decision", json=decision(run_id), headers=h).status_code == 404


def test_rejected_run_cannot_be_marked_written(client, settings):
    run_id = past_run(settings)
    r = client.post("/agent/decision", json=decision(run_id, decision="rejected", written_to_crm=True),
                    headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_decision_rejects_unknown_values(client, settings):
    run_id = past_run(settings)
    r = client.post("/agent/decision", json=decision(run_id, decision="maybe"), headers={"X-API-Key": "test-key"})
    assert r.status_code == 422
