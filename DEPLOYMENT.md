# BrainOps AI Agents - Deployment SOP

## CURRENT SETUP: Docker Hub Image

**Render is configured to pull from Docker Hub, NOT build from Dockerfile.**

- **Image URL:** `docker.io/mwwoodworth/brainops-ai-agents:latest`
- **Docker Hub:** https://hub.docker.com/r/mwwoodworth/brainops-ai-agents

> **NOTE:** The render.yaml in this repo says `runtime: docker` but Render dashboard
> settings override it. The actual config is `Image` mode pointing to Docker Hub.

---

## Standard Deployment Workflow

### Every deploy requires these steps:

```bash
# 1. Make your code changes

# 2. Update version in app.py
VERSION = "X.Y.Z"  # Increment appropriately

# 3. Commit changes
git add -A && git commit -m "feat: Description of change"

# 4. Build Docker image with BOTH tags
docker build -t mwwoodworth/brainops-ai-agents:latest -t mwwoodworth/brainops-ai-agents:vX.Y.Z .

# 5. Push BOTH tags to Docker Hub
docker push mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:vX.Y.Z

# 6. Push to git (for version control)
git push

# 7. Trigger Render deploy (one of these methods):
#    a) Render Dashboard → Manual Deploy → Deploy latest commit
#    b) Or use deploy hook (see below)

# 8. Verify deployment
curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'
```

### Quick Deploy Script
Save this as `deploy.sh` for convenience:
```bash
#!/bin/bash
set -e
VERSION=$(grep "^VERSION" app.py | cut -d'"' -f2)
echo "Deploying version $VERSION..."
docker build -t mwwoodworth/brainops-ai-agents:latest -t mwwoodworth/brainops-ai-agents:v$VERSION .
docker push mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:v$VERSION
git push
echo "✅ Pushed to Docker Hub and Git. Now trigger deploy in Render dashboard."
```

---

## Render Deploy Hook

The deploy hook URL is in Render dashboard under Settings → Deploy Hook.
Keep it secret. Use it to trigger deploys programmatically:
```bash
curl -X POST "YOUR_DEPLOY_HOOK_URL"
```

---

## Why Docker Hub Instead of Dockerfile?

| Aspect | Docker Hub | Dockerfile Build |
|--------|------------|------------------|
| Deploy time | ~30 seconds | ~8 minutes |
| Control | Exact image you tested | Built on Render |
| Complexity | More steps | Just git push |

We use Docker Hub because:
1. Faster deploys (critical for hotfixes)
2. Same image tested locally = deployed to prod
3. Tagged versions for rollback

---

## Rollback Procedure

If a deploy breaks production:

```bash
# 1. Identify last working version
# Check Docker Hub tags: https://hub.docker.com/r/mwwoodworth/brainops-ai-agents/tags

# 2. Retag the old version as :latest
docker pull mwwoodworth/brainops-ai-agents:v9.66.0  # example
docker tag mwwoodworth/brainops-ai-agents:v9.66.0 mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:latest

# 3. Trigger Render deploy
# Use dashboard or deploy hook
```

---

## Monitoring & Verification

```bash
# Check deployed version
curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'

# Check full health
curl -s https://brainops-ai-agents.onrender.com/health | jq '.'

# Test brain endpoint
curl -s https://brainops-ai-agents.onrender.com/brain/critical \
  -H "X-API-Key: brainops_prod_key_2025" | jq 'length'
```

---

## Troubleshooting

### "MaxClientsInSessionMode" errors
Database connection pool exhausted. The service has too many concurrent connections.
- Check Supabase dashboard for connection count
- Reduce `min_size`/`max_size` in async_connection.py
- Or restart the service to clear stale connections

### OpenAI 429 errors
API quota exceeded. Check billing at https://platform.openai.com/account/billing

### Deploy not updating
1. Verify `:latest` tag was pushed: `docker pull mwwoodworth/brainops-ai-agents:latest`
2. Check Render logs for pull errors
3. Manually trigger deploy from dashboard

---

## Service URLs

| Service | URL |
|---------|-----|
| Production | https://brainops-ai-agents.onrender.com |
| Render Dashboard | https://dashboard.render.com |
| Docker Hub | https://hub.docker.com/r/mwwoodworth/brainops-ai-agents |
| GitHub Repo | https://github.com/mwwoodworth/brainops-ai-agents |

---

## Version History (Recent)

| Version | Date | Changes |
|---------|------|---------|
| 9.67.0 | 2025-12-31 | Disable InMemory fallback in production |
| 9.66.0 | 2025-12-31 | Security hardening, SQL injection fixes |
| 9.65.0 | 2025-12-31 | Async/await fixes for brain endpoints |

---

**IMPORTANT:** Always push both `:latest` AND `:vX.Y.Z` tags. The version tag is for rollback capability.
