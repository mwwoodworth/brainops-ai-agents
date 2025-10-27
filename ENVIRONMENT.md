# BrainOps AI Agents – Deployment Environment

Set the following variables in Render (or your secret manager) before deploying. Do **not** commit actual values to source control.

| Key | Purpose |
|-----|---------|
| `DB_HOST` | Postgres host for Supabase |
| `DB_NAME` | Database name |
| `DB_USER` | Database role with required privileges |
| `DB_PASSWORD` | Database password |
| `DB_PORT` | Database port (e.g. `5432`) |
| `SYSTEM_USER_ID` | Default system user UUID used by agents |
| `TENANT_ID` | Default tenant identifier |
| `OPENAI_API_KEY` | OpenAI access key (if used) |
| `ANTHROPIC_API_KEY` | Anthropic access key (if used) |
| `GOOGLE_AI_API_KEY` | Google Generative AI key (if used) |
| `REDIS_URL` | Redis connection string (optional) |

Keep service-specific config (e.g. `PORT`, `WEB_CONCURRENCY`, `LOG_LEVEL`, `ENVIRONMENT`) synchronized with Render but these may remain in version control because they are non-sensitive.
