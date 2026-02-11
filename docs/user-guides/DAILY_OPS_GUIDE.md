# BrainOps AI OS — Daily Operations Guide

## 1. System Health Check

### Quick status (CLI)
```bash
source ~/dev/_secure/BrainOps.env
curl -s "https://brainops-ai-agents.onrender.com/health" \
  -H "X-API-Key: $BRAINOPS_API_KEY" | jq '{status, version, system_count, database, circuit_breakers}'
```

### Expected output
- `status`: `healthy`
- `database`: `connected`
- `system_count`: 19
- `circuit_breakers.overall_health`: `healthy`

---

## 2. Invariant Monitor

The invariant monitor runs every 5 minutes and checks 9 invariants:

| # | Check | Severity | What it detects |
|---|-------|----------|-----------------|
| 1 | `runtime_identity` | critical | Agents running as wrong DB user |
| 2 | `orphaned_invoices` | high | Invoices with NULL tenant_id |
| 3 | `cross_tenant_mismatch` | critical | Jobs linked to wrong tenant's customers |
| 4 | `stuck_webhooks` | warning | Webhooks stuck in processing >1hr |
| 5 | `rls_no_policies` | high | Tables with RLS enabled but zero policies |
| 6 | `invoice_line_item_mismatch` | high | Invoice total != sum of line items |
| 7 | `negative_invoices` | high | Invoices with negative amounts |
| 8 | `revenue_demo_flag_missing` | warning | Revenue rows missing is_demo flag |
| 9 | `synthetic_canary` | info | Pipeline liveness proof |

### Query open violations
```sql
SELECT check_name, severity, message, created_at
FROM invariant_violations
WHERE resolved = false
ORDER BY created_at DESC
LIMIT 20;
```

### Query violation history (last 24h)
```sql
SELECT check_name, severity, count(*) as occurrences,
       min(created_at) as first_seen, max(created_at) as last_seen
FROM invariant_violations
WHERE created_at > NOW() - INTERVAL '24 hours'
  AND check_name != 'synthetic_canary'
GROUP BY check_name, severity
ORDER BY occurrences DESC;
```

### Verify canary drill is running
```sql
SELECT count(*) as canary_count,
       max(created_at) as last_canary
FROM invariant_violations
WHERE check_name = 'synthetic_canary'
  AND created_at > NOW() - INTERVAL '1 hour';
```
Expected: `canary_count` >= 12 (every 5 min for 1 hour).

---

## 3. Slack Alerts

Invariant violations with severity `critical` or `high` are sent to Slack.

### Verify Slack is configured
```bash
curl -s "https://brainops-ai-agents.onrender.com/health" \
  -H "X-API-Key: $BRAINOPS_API_KEY" | jq '.capabilities'
```

The `SLACK_WEBHOOK_URL` env var must be set on Render for alerts to fire.

---

## 4. Webhook Deduplication

All 3 active webhook handlers now have event deduplication:

| Handler | System | Dedup Method |
|---------|--------|-------------|
| ERP (`/api/stripe/webhook`) | Weathercraft ERP | Atomic upsert + processing lock + stale takeover |
| CC (`/api/income/stripe`) | Command Center | Atomic upsert + status check |
| MRG Backend (`/api/v1/stripe/webhook`) | MRG Backend | `ON CONFLICT DO NOTHING` + rowcount check |
| Agents (`/stripe/webhook`) | BrainOps Agents | Top-level `event.id` dedup + status tracking |

### Verify dedup is working
```sql
-- Recent webhook events (should show unique event_ids)
SELECT event_id, event_type, status, created_at
FROM stripe_webhook_events
ORDER BY created_at DESC
LIMIT 10;

-- Check for any stuck-processing events
SELECT count(*) as stuck
FROM stripe_webhook_events
WHERE status = 'processing'
  AND created_at < NOW() - INTERVAL '1 hour';
```

---

## 5. DB Blast-Radius Cap

Agents run as `agent_worker` (not `postgres`). Verify:

```sql
-- Check who is connected
SELECT usename, count(*) as sessions
FROM pg_stat_activity
WHERE datname = 'postgres'
GROUP BY usename ORDER BY sessions DESC;
```

Expected: `agent_worker` has sessions, `postgres` sessions are from Backend/MCP Bridge only.

```sql
-- Verify DELETE/TRUNCATE are blocked
SELECT privilege_type, count(*)
FROM information_schema.role_table_grants
WHERE grantee = 'app_agent_role'
  AND privilege_type IN ('DELETE', 'TRUNCATE')
GROUP BY privilege_type;
```
Expected: 0 rows (no DELETE/TRUNCATE grants).

---

## 6. Daily Checklist

- [ ] Run health check (section 1)
- [ ] Check for open violations: `SELECT count(*) FROM invariant_violations WHERE resolved = false AND check_name != 'synthetic_canary'`
- [ ] Verify canary is running (section 2)
- [ ] Check Slack channel for alerts
- [ ] Verify agent_worker sessions are active (section 5)
- [ ] Review recent deploys: Render dashboard or `curl Render API`
