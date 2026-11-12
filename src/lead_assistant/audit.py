"""Audit trail: one JSON line per agent run.

Events are collected in memory while the run happens and written once at the
end, so a run's record is never split or interleaved with another run's.
API keys never reach this module. Lead fields do (they're the inputs being
audited); in production, restrict access to the log file like any CRM export.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditRun:
    def __init__(self, path: Path, lead_id: str, model: str):
        self.path = path
        self.run_id = uuid.uuid4().hex[:12]
        self._t0 = time.perf_counter()
        self.record: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lead_id": lead_id,
            "model": model,
            "events": [],
        }

    def event(self, kind: str, **data: Any) -> None:
        self.record["events"].append({"t": round(time.perf_counter() - self._t0, 2), "kind": kind, **data})

    def finish(self, output: dict | None = None, error: str | None = None) -> None:
        self.record["duration_s"] = round(time.perf_counter() - self._t0, 2)
        self.record["output"] = output
        if error:
            self.record["error"] = error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False, default=str) + "\n")


# ---------- Week 3: human decisions ----------

def find_run(path: Path, run_id: str) -> dict | None:
    """The audit record for run_id, or None. A linear scan is fine at this volume;
    a database would replace the JSONL file long before it got slow."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if f'"run_id": "{run_id}"' not in line:  # cheap pre-filter before parsing
                continue
            record = json.loads(line)
            if record.get("run_id") == run_id and "events" in record:
                return record
    return None


def record_decision(path: Path, decision: dict[str, Any]) -> dict[str, Any]:
    """Append a human decision as its own line. Agent runs are never rewritten,
    so the log stays append-only: what the AI proposed, then what a person did."""
    record = {"kind": "human_decision",
              "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **decision}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
