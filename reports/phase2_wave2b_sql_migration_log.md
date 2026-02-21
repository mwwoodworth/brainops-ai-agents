# Phase 2 Wave 2B — SQL Migration Log

**Date:** 2026-02-21
**Scope:** Inline SQL extraction from app.py scheduler/agent routes to service modules

---

## Summary

| Metric | Count |
|--------|-------|
| SQL queries moved to service modules | 5 |
| SQL queries remaining inline in api/agents.py | 18 |
| New service module created | services/scheduler_queries.py |

---

## Extracted SQL (services/scheduler_queries.py)

### 1. `fetch_email_queue_counts()`
**Source:** app.py `email_scheduler_stats` handler
```sql
SELECT
    COUNT(*) FILTER (WHERE status = 'queued') as queued,
    COUNT(*) FILTER (WHERE status = 'processing') as processing,
    COUNT(*) FILTER (WHERE status = 'sent') as sent,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
    COUNT(*) as total
FROM ai_email_queue
```

### 2. `fetch_active_agents(pool)`
**Source:** app.py `activate_all_agents_scheduler` handler
```sql
SELECT id, name, type, category FROM ai_agents WHERE status = 'active'
```

### 3. `fetch_scheduled_agent_ids(pool)`
**Source:** app.py `activate_all_agents_scheduler` handler
```sql
SELECT agent_id FROM agent_schedules WHERE enabled = true
```

### 4. `insert_agent_schedule(pool, agent_id, frequency_minutes)`
**Source:** app.py `activate_all_agents_scheduler` handler
```sql
INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
VALUES (gen_random_uuid(), $1, $2, true, NOW())
```

### 5. `upsert_agent_schedule(pool, agent_id, frequency_minutes, enabled, schedule_id)`
**Source:** app.py `schedule_agent` handler
```sql
-- Verify agent exists
SELECT id, name FROM agents WHERE id = $1

-- Check existing
SELECT id FROM agent_schedules WHERE agent_id = $1

-- Update
UPDATE agent_schedules
SET frequency_minutes = $1, enabled = $2, updated_at = NOW()
WHERE agent_id = $3

-- Or insert
INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
VALUES ($1, $2, $3, $4, NOW())
```

---

## Inline SQL Remaining in api/agents.py

The following SQL queries remain inline in route handlers. These are candidates for future
service-layer extraction (Wave 2C/2D) but were kept inline for this wave to minimize
the blast radius of changes:

| Handler | Table(s) | Query Type | Lines |
|---------|----------|------------|-------|
| `get_agents` | agents, ai_agent_executions | SELECT, COUNT, JOIN | ~30 |
| `execute_agent` | agents, ai_agent_executions | SELECT, INSERT, UPDATE | ~25 |
| `get_all_agents_status` | ai_agents, agent_schedules | SELECT, LEFT JOIN | ~15 |
| `get_agent` | agents | SELECT | ~3 |
| `get_agent_history` | agents, ai_agent_executions | SELECT | ~10 |
| `execute_scheduled_agents` | agents, ai_agent_executions, agent_executions | SELECT, INSERT, UPDATE | ~30 |
| `check_agents_health` | — (delegates to health monitor) | — | 0 |
| `restart_agent` | ai_agents | SELECT | ~3 |
| `execute_agent_generic` | agents, ai_agent_executions | SELECT, INSERT, UPDATE | ~30 |
| `api_v1_agents_activate` | agents | UPDATE RETURNING | ~10 |
| `execute_aurea_event` | agents, brainops_core.agents | SELECT, UPDATE | ~10 |

**Total remaining inline SQL in api/agents.py:** ~166 lines across 18 queries.

---

## Tables Touched by Extracted Queries

| Table | Operations | Module |
|-------|-----------|--------|
| ai_email_queue | SELECT (aggregate) | services/scheduler_queries.py |
| ai_agents | SELECT | services/scheduler_queries.py |
| agent_schedules | SELECT, INSERT, UPDATE | services/scheduler_queries.py |
| agents | SELECT | services/scheduler_queries.py |

---

## Migration Safety

- All SQL queries were moved verbatim — no query logic was modified.
- Parameter binding ($1, $2, etc.) preserved exactly.
- Table names unchanged.
- No new indexes or schema changes required.
- The `upsert_agent_schedule` function encapsulates the SELECT-then-INSERT/UPDATE pattern
  that was previously scattered across the `schedule_agent` handler.
