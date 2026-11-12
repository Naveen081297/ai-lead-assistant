"""Check the Zoho CRM setup before wiring n8n to it. Read-only: it changes nothing in Zoho.

    python scripts/zoho_check.py              # token + field check
    python scripts/zoho_check.py <lead_id>    # also fetch and parse one real lead

Needs ZOHO_DC, ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET and ZOHO_REFRESH_TOKEN in .env.
The refresh token's scope must be: ZohoCRM.modules.leads.READ,ZohoCRM.settings.fields.READ
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from lead_assistant.config import Settings
from lead_assistant.leads import LeadNotFound, ZohoLeadSource
from lead_assistant.schemas import Qualification

# API name -> (Zoho data types that work, what it's for). Zoho builds the API name
# from the label you type ("AI Score" -> AI_Score), so labels must match exactly.
EXPECTED_FIELDS = {
    "Program_Interest": ({"picklist", "text"}, "read by the agent"),
    "Qualification": ({"picklist"}, "read by the agent"),
    "Planned_Start": ({"date"}, "read by the agent"),
    "AI_Score": ({"integer"}, "written by n8n"),
    "AI_Band": ({"picklist", "text"}, "written by n8n"),
    "AI_Reply_Subject": ({"text"}, "written by n8n"),
    "AI_Approved_Reply": ({"textarea"}, "written by n8n"),
    "AI_Notes": ({"textarea"}, "written by n8n"),
    "AI_Run_ID": ({"text"}, "written by n8n"),
}


def main() -> int:
    s = Settings()
    if s.lead_source != "zoho":
        print("Note: LEAD_SOURCE isn't 'zoho', so the API still reads the sample file. Checking Zoho anyway.\n")
    zoho = ZohoLeadSource(s)

    print(f"1. Token refresh at accounts.zoho.{s.zoho_dc} ...", end=" ")
    zoho._access_token()
    print("ok")

    print("2. Lead fields ...")
    r = zoho._http.get(f"https://www.zohoapis.{s.zoho_dc}/crm/{s.zoho_api_version}/settings/fields",
                       params={"module": "Leads"},
                       headers={"Authorization": f"Zoho-oauthtoken {zoho._access_token()}"})
    if r.status_code == 401:
        print("   401: the refresh token lacks ZohoCRM.settings.fields.READ. Regenerate it with both scopes.")
        return 1
    r.raise_for_status()
    fields = {f["api_name"]: f for f in r.json().get("fields", [])}
    problems = 0
    for name, (types, purpose) in EXPECTED_FIELDS.items():
        f = fields.get(name)
        if f is None:
            print(f"   MISSING  {name:<18} ({purpose})")
            problems += 1
        elif f["data_type"] not in types:
            print(f"   TYPE     {name:<18} is {f['data_type']}, expected {' or '.join(sorted(types))}")
            problems += 1
        else:
            print(f"   ok       {name:<18} {f['data_type']}")
    # The agent validates Qualification against a fixed list; a different picklist
    # value would make every lead with it fail to load.
    q = fields.get("Qualification")
    if q and q["data_type"] == "picklist":
        actual = {v["actual_value"] for v in q.get("pick_list_values", [])} - {"-None-"}
        extra = actual - set(Qualification.__args__)
        if extra:
            print(f"   VALUES   Qualification has values the agent doesn't accept: {sorted(extra)}")
            problems += 1

    if len(sys.argv) > 1:
        lead_id = sys.argv[1]
        print(f"3. Lead {lead_id} ...", end=" ")
        try:
            lead = zoho.get(lead_id)
        except LeadNotFound:
            print("not found (check the ID and the data centre)")
            return 1
        except ValidationError as e:
            print(f"fetched but doesn't fit the Lead model:\n{e}")
            return 1
        print(f"ok: {lead.first_name} {lead.last_name}, {lead.program_interest}, message {len(lead.message)} chars")

    print("\nAll good." if not problems else f"\n{problems} field problem(s): fix them in Zoho (Setup > Modules > Leads).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
