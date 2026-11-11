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
