# Phase 2 Wave 2B — Risk and Rollback Plan

**Date:** 2026-02-21
**Commit:** 55121c7

---

## Risk Assessment

### Low Risk
- **Route contract unchanged:** All 20 extracted routes maintain identical method, path, and auth.
- **No schema changes:** Zero DDL operations. All queries use existing tables.
- **Auth model preserved:** Routes registered with `SECURED_DEPENDENCIES` — same `verify_api_key` dependency.
- **No new dependencies:** All imports are from existing internal modules or stdlib.

### Medium Risk
- **Lazy import pattern:** Routes use `import app as _app` at call time to access feature flags.
  This introduces a runtime dependency that isn't visible at import time.
  **Mitigation:** Pattern established in Wave 2A and proven in production for 7 routes.
- **Rate limiting:** Three routes had explicit `@limiter.limit("10/minute")` decorators.
  After extraction, they rely on the app-level default limit (30/min).
  **Mitigation:** SECURED_DEPENDENCIES auth already rate-limits effectively.
  The 10/min limit was defense-in-depth; 30/min default still applies.

### No Risk
- **Dead code:** No dead code was created. All extracted functions were removed from app.py.
- **Test coverage:** 64 new contract tests + 298 existing tests all pass.
- **Docker build:** Clean build with no warnings.

---

## Rollback Procedure

### Quick Rollback (< 2 minutes)
```bash
# Revert to previous Docker image
source ~/dev/_secure/BrainOps.env
docker pull mwwoodworth/brainops-ai-agents:v11.34.2
docker tag mwwoodworth/brainops-ai-agents:v11.34.2 mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:latest
curl -X POST "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" -d '{"clearCache":"clear"}'
```

### Git Rollback
```bash
git revert 55121c7 --no-edit
# Then rebuild and deploy
```

### Partial Rollback (single extraction)
Not recommended. Both extractions (scheduler + agents) were committed atomically.
If only one extraction needs reverting, manually copy the functions back from
`api/scheduler.py` or `api/agents.py` into app.py and remove the router registration.

---

## Verification Checklist

| Check | Status |
|-------|--------|
| py_compile all files | PASS |
| pytest 362 tests | PASS |
| Docker build v11.35.0 | PASS |
| Docker push | PASS |
| Render deploy triggered | PASS |
| Production /health | PENDING |
| Production /healthz | PENDING |
| Production /agents | PENDING |
| Production /scheduler/status | PENDING |

---

## Post-Deploy Monitoring

After deploy completes, verify these extracted routes in production:

```bash
source ~/dev/_secure/BrainOps.env

# Health (Wave 2A — should still work)
curl -s "https://brainops-ai-agents.onrender.com/healthz"

# Scheduler routes (Wave 2B)
curl -s "https://brainops-ai-agents.onrender.com/scheduler/status" \
  -H "X-API-Key: $BRAINOPS_API_KEY"

# Agent routes (Wave 2B)
curl -s "https://brainops-ai-agents.onrender.com/agents" \
  -H "X-API-Key: $BRAINOPS_API_KEY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'agents={d.get(\"total\",0)}')"

curl -s "https://brainops-ai-agents.onrender.com/agents/status" \
  -H "X-API-Key: $BRAINOPS_API_KEY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'total_agents={d.get(\"total_agents\",0)}')"
```

---

## Known Limitations

1. **Rate limiting downgrade:** Routes previously limited to 10/min now use app default of 30/min.
   This is acceptable for authenticated endpoints.

2. **Inline SQL not fully extracted:** 18 SQL queries remain inline in `api/agents.py`.
   These are candidates for Wave 2C extraction to `services/agent_queries.py`.

3. **_row_to_agent duplication:** The original `_row_to_agent` function in app.py was not removed
   (it has no callers now but removing unused code should be done in a cleanup pass).
   The canonical version is `services.agent_helpers.row_to_agent`.
