# Week 3 setup: Zoho CRM and n8n

This connects a real Zoho CRM account to the n8n workflow in `n8n/`. Do it in order: each step is checked before the next one depends on it.

Use a Zoho CRM **trial** (Professional or Enterprise, 15 days) rather than the Free edition: the flow needs custom fields, workflow rules and webhooks. Use only made-up leads.

Throughout, `{dc}` is your Zoho data centre: `in` if your account URL is `crm.zoho.in`, `com` for `crm.zoho.com`, `eu`, `com.au`, and so on. It must be the same everywhere.

## 1. Fields on the Leads module

Setup → Customization → Modules and Fields → Leads → Standard layout. Zoho makes the API name from the label (`AI Score` → `AI_Score`), so type the labels exactly.

| Label | Type | Notes | Used by |
|---|---|---|---|
| Program Interest | Pick list | Data Analytics Pro, Full Stack Web Development, Digital Marketing Accelerator, UI/UX Design Professional, Python Foundations | agent (read) |
| Qualification | Pick list | Class 10, Class 12, Undergraduate (final year), Undergraduate, Graduate, Postgraduate. **Exactly these values**: the agent rejects others. | agent (read) |
| Planned Start | Date | | agent (read) |
| AI Score | Number | | n8n (write) |
| AI Band | Pick list | Hot, Warm, Cold | n8n (write) |
| AI Reply Subject | Single Line | | n8n (write) |
| AI Approved Reply | Multi Line (Large) | Small holds only 2,000 characters | n8n (write) |
| AI Notes | Multi Line (Large) | | n8n (write) |
| AI Run ID | Single Line | Links the CRM record to `logs/audit.jsonl` | n8n (write) |

Also add these values to the standard **Lead Source** pick list, since the scoring rules give them points: Website, Referral, Walk-in, Demo Class, Google Ads, Facebook Ads.

## 2. Two refresh tokens, two scopes

The agent API only ever reads the CRM. Only n8n can write, and only after a person approves. Each gets its own token with only the scope it needs, so a bug or a prompt injection in the AI service can't change CRM data.

1. Open `https://api-console.zoho.{dc}` → **Self Client** → Create. Note the client ID and secret.
2. **Generate Code** tab, scope `ZohoCRM.modules.leads.READ,ZohoCRM.settings.fields.READ`, duration 10 minutes. Copy the code, then exchange it within the 10 minutes:
   ```powershell
   Invoke-RestMethod -Method Post "https://accounts.zoho.{dc}/oauth/v2/token?grant_type=authorization_code&client_id=CLIENT_ID&client_secret=CLIENT_SECRET&code=CODE"
   ```
   The `refresh_token` in the response is the **read token**. It doesn't expire unless revoked. The code works only once.
3. Repeat step 2 with scope `ZohoCRM.modules.leads.UPDATE`. That `refresh_token` is the **write token**, for n8n.

Put the read token in `.env`:
```
LEAD_SOURCE=zoho
ZOHO_DC=in
ZOHO_CLIENT_ID=...
ZOHO_CLIENT_SECRET=...
ZOHO_REFRESH_TOKEN=<read token>
```

## 3. Check the Zoho side

Create a test lead in Zoho by hand (made-up name, a Program Interest, a question in Description). Copy its ID from the URL (`.../tab/Leads/5725767000000412345`), then run:

```powershell
python scripts/zoho_check.py 5725767000000412345
```

It refreshes the token, checks all nine fields exist with the right types and pick-list values, and fetches and parses that lead. Then run the agent on it with `LEAD_SOURCE=zoho`:

```powershell
python -m lead_assistant.agent 5725767000000412345
```

## 4. n8n

Import both files (Workflows → Import from File): `n8n/error-alert.json` first, then `n8n/lead-approval.json`. Then create these credentials and select them on the nodes that show a warning:

| Credential (type) | Fields | Nodes |
|---|---|---|
| Zoho webhook token (Header Auth) | Name `X-Webhook-Token`, Value: a long random string | Zoho new lead |
| Lead Assistant API (Header Auth) | Name `X-API-Key`, Value: same as `LEAD_ASSISTANT_API_KEY` | Run agent, Record approval, Record rejection |
| Zoho token request (Custom Auth) | `{"qs": {"grant_type": "refresh_token", "client_id": "...", "client_secret": "...", "refresh_token": "<write token>"}}` | Get Zoho token |
| Approver login (Basic Auth) | A user name and password for whoever approves | Wait for approval |
| SMTP | e.g. Gmail: `smtp.gmail.com`, port 465, SSL on, an app password | Email approver, Alert, Alert email |

Make a random value with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

Then:
- Open the **Config** node and set `api_base_url`, the two Zoho URLs for your `{dc}` (`https://accounts.zoho.in`, `https://www.zohoapis.in/crm/v8`) and the three email addresses. Do the same for the email addresses in the error workflow.
- Workflow settings → Error workflow → "Lead assistant: error alert".
- **Publish** the approval workflow. Its production URL is `https://<your n8n>/webhook/zoho-new-lead`.

### Why Zoho is called with HTTP Request nodes, not the Zoho node

Zoho expects `Authorization: Zoho-oauthtoken <token>`, not the standard `Bearer <token>`, and n8n's built-in Zoho credential asks for broad scopes. The workflow instead exchanges the write token for an access token ("Get Zoho token") and sends the header itself. This is the same flow `leads.py` uses, so both sides behave the same way.

## 5. The Zoho webhook

Zoho runs in the cloud, so it can only call an n8n it can reach: the DigitalOcean deploy, or a tunnel such as `cloudflared tunnel --url http://localhost:5678` while testing. Set n8n's `WEBHOOK_URL` to that public address, or the approval links in emails will point at `localhost`.

Setup → Automation → Workflow Rules → Create rule, module Leads:
- **When:** on a record action → **Create** only. Don't choose "Create or Edit": n8n's update would then trigger the rule again. The workflow also sends `"trigger": []` with its update for this reason.
- **Condition:** all leads (or e.g. Lead Source is not empty).
- **Instant action → Webhook:** method POST, URL = the production webhook URL above.
  - Add a header (custom parameter) `X-Webhook-Token` = the value from the n8n credential.
  - Send the lead ID as `lead_id`: either a raw JSON body `{"lead_id": "${Leads.Lead Id}"}` or a module parameter named `lead_id` mapped to Lead Id. The workflow accepts the ID from a JSON body, a form body or the query string.

## 6. End-to-end test

Create a new lead in Zoho. You should get the approval email within about 20 seconds. Approve it, then reload the lead in Zoho: the six AI fields should be filled in, and `logs/audit.jsonl` on the API server should have a `human_decision` line with the same run ID as the AI Run ID field.

## What happens when things fail

| Failure | What the workflow does |
|---|---|
| Wrong or missing webhook token | n8n answers 403; no execution starts |
| Agent API down, slow or erroring | 3 attempts, 5 s apart, then an alert email. No approval email is sent, and Zoho is not touched. |
| Approver doesn't answer in 3 days | The workflow ends at "No decision in 3 days"; nothing is written |
| Approver rejects | Nothing is written to Zoho; the rejection and comment go to the audit log |
| Zoho token refresh or update fails, including HTTP 200 with `"status": "error"` (e.g. a wrong field name) | 3 attempts, then an alert. The approval isn't recorded as written to the CRM. |
| Anything else (a code bug, the SMTP server down) | The error workflow emails the failed node and execution ID. If SMTP itself is down, the failure shows only in n8n's execution list. |
