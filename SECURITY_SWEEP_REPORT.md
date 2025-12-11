# BrainOps AI Agents – Security Sweep (2024-05-15)

- Reviewed `app.py`, `agent_executor.py`, `config.py`, all `api/` modules, and supporting middleware (none present).
- Executed required scans:\
  `grep -r 'password.*=' --include='*.py' .`\
  `grep -r 'subprocess' --include='*.py' .`\
  `grep -r 'eval(' --include='*.py' .`\
  `grep -r 'exec(' --include='*.py' .`
- High-risk issues identified: hardcoded secrets, unauthenticated high-value endpoints, command execution with untrusted input, insecure eval/exec paths, and permissive CORS.

## Findings

1) Hardcoded secrets (HIGH)
- `api/gumroad_webhook.py`: Default values include real-looking keys (`CONVERTKIT_API_KEY`, `CONVERTKIT_API_SECRET`, `CONVERTKIT_FORM_ID`) and placeholders for Stripe/SendGrid. These load even in production if env vars are absent.
- `self_healing_recovery.py`: Database defaults embed Supabase host/user and password (`Brain0ps2O2S`).
- Test artifacts (`test_everything.py`, `remove_secrets.py`) still contain the same database password, increasing leakage risk.
Recommendation: Remove defaults; require env vars; rotate exposed credentials immediately across ConvertKit/DB/related systems; scrub test artifacts.

2) Authentication bypass of critical APIs (HIGH)
- Routers under `api/` (`api/memory.py`, `api/brain.py`, `api/memory_coordination.py`, `api/codebase_graph.py`, `api/state_sync.py`, `api/gumroad_webhook.py`, `api/customer_intelligence.py`) are included in `app.py` without `Depends(verify_api_key)`. With `AUTH_REQUIRED` enabled, callers can still hit these endpoints unauthenticated, exposing memory stores, brain context, schema intel, change logs, git status, and webhook processing.
Recommendation: Apply `Depends(verify_api_key)` (or stronger auth) at router or app include level; segregate public vs private routes explicitly.

3) Command injection via shell execution (HIGH)
- `agent_executor.py` `DeploymentAgent.deploy_backend/build_docker`: Builds and pushes Docker images using `subprocess.run(..., shell=True)` with the user-supplied `version` and `service` fields interpolated into shell commands. A crafted version string can inject arbitrary commands on the host.
Recommendation: Avoid `shell=True`; pass args list; strictly validate/whitelist version/service tokens; run builds in isolated worker.

4) Insecure eval/exec execution (HIGH)
- `self_healing_recovery.py`: Healing rules use `eval(rule['condition'])` and `exec(rule['action'])` on database-provided strings. Compromise of DB or rule authoring yields arbitrary code execution.
Recommendation: Remove eval/exec; replace with safe rule engine or vetted allow-list of actions; validate rule content at write time.

5) CORS misconfiguration (MEDIUM)
- `app.py`: CORS defaults to `allow_origins=["*"]` with `allow_credentials=True` when `ALLOWED_ORIGINS` is unset (production default). This enables credentialed cross-site requests from any origin.
Recommendation: Set explicit origin allowlist per environment; disable credentials unless required.

6) Sensitive data exposure (MEDIUM)
- `api/state_sync.py`: Unauthenticated access to `/api/state-sync/context/raw`, `/api/state-sync/changes*`, and `/api/state-sync/git-status` leaks local files (e.g., `AI_SYSTEM_STATE.json`), change history, and git status.
- `api/codebase_graph.py`: Exposes database schema metadata and ERD without auth.
- `api/gumroad_webhook.py`: Logs purchaser emails/names; successful requests processed without auth if webhook secret unset.
Recommendation: Protect endpoints with auth; restrict filesystem access; remove PII from logs or redact.

7) SQL safety review (LOW)
- Most queries are parameterized. Minor string interpolation of `LIMIT` in `api/memory.py` uses FastAPI-constrained ints, reducing injection risk but still better parameterized.
Recommendation: Use query params for limits/filters consistently.

8) Debug/info leakage (LOW)
- `config.SecurityConfig.dev_mode` controls detailed error responses; if `DEV_MODE=true` in production, stack traces/messages are returned to clients and logged.
Recommendation: Ensure `DEV_MODE` is false in production; add runtime guardrails.

## Additional notes
- Logging: Global exception handler uses `dev_mode` to decide whether to return error messages; ensure sensitive payloads are not logged at info level. Gumroad webhook logs include purchaser email/sale_id.
- No explicit path traversal found, but file-serving endpoints are unauthenticated (see Finding 6).

## Recommended Remediations (prioritized)
1. Rotate exposed keys/passwords; remove all hardcoded defaults; scrub tests and history if secrets were committed.
2. Enforce authentication on all non-public routers; add integration tests to assert 401/403 when missing API key.
3. Replace shell-based deploy/build commands with parameterized subprocess calls or orchestration service; validate inputs.
4. Remove eval/exec in `self_healing_recovery.py`; implement safe, declarative healing actions.
5. Lock down CORS to known origins; disable `allow_credentials` unless explicitly required.
6. Restrict or remove unauthenticated endpoints exposing internal state/schema; gate with auth and least privilege.
7. Parameterize remaining SQL fragments and add linting/checks for unsafe string formatting.

