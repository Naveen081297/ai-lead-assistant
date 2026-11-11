"""Where get_lead reads leads from: a local sample file or the Zoho CRM API.

Both return a validated `Lead`, so the agent never sees raw CRM JSON.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

import httpx

from lead_assistant.config import Settings
from lead_assistant.schemas import Lead


class LeadNotFound(LookupError):
    pass


class LeadSource(Protocol):
    def get(self, lead_id: str) -> Lead: ...


class SampleLeadSource:
    """Made-up leads from data/sample_leads.json, stored with Zoho field names."""

    def __init__(self, settings: Settings):
        rows = json.loads(settings.sample_leads_path.read_text(encoding="utf-8"))
        self._leads = {r["id"]: r for r in rows}

    def get(self, lead_id: str) -> Lead:
        if lead_id not in self._leads:
            raise LeadNotFound(lead_id)
        return Lead.model_validate(self._leads[lead_id])


class ZohoLeadSource:
    """Zoho CRM REST API (Week 3; not yet run against a real account).

    Auth is Zoho's OAuth refresh-token flow: a long-lived refresh token (from a
    Self Client in the Zoho API console, scope ZohoCRM.modules.leads.READ) is
    exchanged for an access token that lasts about an hour.
    """

    def __init__(self, settings: Settings):
        missing = [n for n in ("zoho_client_id", "zoho_client_secret", "zoho_refresh_token")
                   if getattr(settings, n) is None]
        if missing:
            raise RuntimeError(f"LEAD_SOURCE=zoho needs {', '.join(m.upper() for m in missing)}")
        self.s = settings
        self._token: str | None = None
        self._token_expires = 0.0
        self._http = httpx.Client(timeout=15)

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires - 60:
            return self._token
        r = self._http.post(f"https://accounts.zoho.{self.s.zoho_dc}/oauth/v2/token", params={
            "grant_type": "refresh_token",
            "client_id": self.s.zoho_client_id,
            "client_secret": self.s.zoho_client_secret.get_secret_value(),
            "refresh_token": self.s.zoho_refresh_token.get_secret_value(),
        })
        r.raise_for_status()
        body = r.json()
        if "access_token" not in body:  # Zoho reports bad credentials as 200 + {"error": ...}
            raise RuntimeError(f"Zoho token refresh failed: {body.get('error', 'unknown error')}")
        self._token = body["access_token"]
        self._token_expires = time.monotonic() + int(body.get("expires_in", 3600))
        return self._token

    def get(self, lead_id: str) -> Lead:
        r = self._http.get(
            f"https://www.zohoapis.{self.s.zoho_dc}/crm/{self.s.zoho_api_version}/Leads/{lead_id}",
            headers={"Authorization": f"Zoho-oauthtoken {self._access_token()}"},
        )
        if r.status_code in (204, 404):
            raise LeadNotFound(lead_id)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise LeadNotFound(lead_id)
        return Lead.model_validate(data[0])


def make_lead_source(settings: Settings) -> LeadSource:
    if settings.lead_source == "zoho":
        return ZohoLeadSource(settings)
    if settings.lead_source == "sample":
        return SampleLeadSource(settings)
    raise ValueError(f"LEAD_SOURCE must be 'sample' or 'zoho', got {settings.lead_source!r}")
