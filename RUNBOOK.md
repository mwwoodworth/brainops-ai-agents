# BrainOps AI Agents – Operations Runbook

Production: Render background worker (start command `python main.py`).

## Environment (minimum)
- Database: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` (Supabase/pg).
- AI keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (as features require).
- Routing/config: `AGENT_CONFIG` or per-agent configs in env/JSON as defined in `ai_config.py`.
- Optional: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` if writing to Supabase tables.

## Install & Run (local)
```bash
cd brainops-ai-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py    # or python agent_scheduler.py for scheduled loops
```

## Deploy (Render)
- Type: Background Worker.
- Build: `pip install -r requirements.txt`.
- Start: `python main.py`.
- Health: add a cron or lightweight `/health` task that exercises at least one model call and database write; fail deployment if it cannot reach keys/DB.

## Checks
- Verify keys present before start.
- Verify DB connectivity (Supabase/pg) on boot.
- Run a “can route task” smoke: create a dummy task via the task queue interface (if exposed) and ensure it completes.

## Logs
- Render dashboard → service logs.
- Local: stdout; rotate via `logging` config if running long-lived.

## Testing
- Add unit tests for agent routing and tool selection.
- Add an integration smoke that hits the public entrypoint (if exposed) and asserts a task completes.
