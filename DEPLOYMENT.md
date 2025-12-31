# BrainOps AI Agents - Deployment SOP

## Current Setup: Dockerfile Auto-Deploy (Recommended)

**How it works:**
1. Push to `main` branch
2. Render detects push via GitHub webhook
3. Render builds from `Dockerfile`
4. New version deployed (~5-8 min)

**Config in render.yaml:**
```yaml
runtime: docker
dockerfilePath: ./Dockerfile
autoDeploy: true
```

### Standard Deployment Workflow
```bash
# 1. Make changes
# 2. Update version in app.py
VERSION = "X.Y.Z"

# 3. Commit and push
git add -A && git commit -m "feat: Description of change" && git push

# 4. Monitor (wait 5-8 min, then check)
curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'
```

### If Auto-Deploy Isn't Working
Check Render dashboard: https://dashboard.render.com/
1. Go to brainops-ai-agents service
2. Settings → Build & Deploy
3. Verify "Auto-Deploy" is ON
4. Verify GitHub repo is connected
5. Verify branch is `main`

**Manual deploy trigger:**
- Go to Render dashboard → brainops-ai-agents → Manual Deploy → Deploy latest commit

---

## Alternative: Docker Hub (Faster deploys, more manual steps)

**When to use:** If Dockerfile builds are too slow or failing on Render.

### Setup (one-time)
```bash
# Login to Docker Hub
docker login -u mwwoodworth

# Update render.yaml to use image
# Change: runtime: docker → runtime: image
# Add: image.url: docker.io/mwwoodworth/brainops-ai-agents:vX.Y.Z
```

### Docker Hub Deployment Workflow
```bash
# 1. Update version in app.py
VERSION = "X.Y.Z"

# 2. Build image locally
docker build -t mwwoodworth/brainops-ai-agents:vX.Y.Z .

# 3. Push to Docker Hub
docker push mwwoodworth/brainops-ai-agents:vX.Y.Z

# 4. Update render.yaml with new tag
sed -i "s|brainops-ai-agents:v[0-9.]*|brainops-ai-agents:vX.Y.Z|" render.yaml

# 5. Commit and push (triggers Render to pull new image)
git add -A && git commit -m "deploy: vX.Y.Z" && git push

# 6. Verify (~30 sec)
curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'
```

**Pros:** Fast deploys (~30 sec vs 5-8 min)
**Cons:** Extra manual steps, easy to forget to update tag

---

## Troubleshooting

### Render shows old version
1. Check Render dashboard for build status
2. Look for build errors in logs
3. Try manual deploy from dashboard
4. If all else fails, use Docker Hub approach

### Build failing on Render
- Check Dockerfile syntax
- Verify requirements.txt has all deps
- Check Render build logs for specific error

### API token for Render (if needed)
Generate new token at: https://dashboard.render.com/u/settings#api-keys
Save to: `~/.render_token` or env var `RENDER_API_KEY`

---

## Quick Reference

| Action | Command |
|--------|---------|
| Check deployed version | `curl -s https://brainops-ai-agents.onrender.com/health \| jq '.version'` |
| Check health | `curl -s https://brainops-ai-agents.onrender.com/health` |
| Test brain endpoint | `curl -s https://brainops-ai-agents.onrender.com/brain/critical -H "X-API-Key: brainops_prod_key_2025"` |

## Service URLs
- **Production:** https://brainops-ai-agents.onrender.com
- **Render Dashboard:** https://dashboard.render.com
- **GitHub Repo:** https://github.com/mwwoodworth/brainops-ai-agents
- **Docker Hub:** https://hub.docker.com/r/mwwoodworth/brainops-ai-agents
