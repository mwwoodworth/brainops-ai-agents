#!/bin/bash
# BrainOps AI Agents - Full Automated Deployment
# Builds, pushes to Docker Hub, triggers Render deploy via API
set -euo pipefail

# Get version from centralized config (never parse arbitrary source lines into a shell var)
VERSION="$(
  env -u VERSION python3 -c "from config import config; print(getattr(config, 'version', 'unknown'))" 2>/dev/null \
    | tr -d '[:space:]'
)"
if [ -z "$VERSION" ] || [ "$VERSION" = "unknown" ]; then
    echo "ERROR: Unable to determine VERSION from config.py"
    exit 1
fi
# Normalize VERSION for Docker tags (strip leading v/V)
VERSION="${VERSION#v}"
VERSION="${VERSION#V}"
SERVICE_ID="srv-d413iu75r7bs738btc10"

echo "=========================================="
echo "Deploying BrainOps AI Agents v$VERSION"
echo "=========================================="

# Capture currently deployed build/version for a robust post-deploy verification.
PRE_HEALTH="$(curl -s "https://brainops-ai-agents.onrender.com/health?force_refresh=true" 2>/dev/null || true)"
PRE_DEPLOYED_BUILD="$(echo "$PRE_HEALTH" | jq -r '.build // empty' 2>/dev/null | tr -d '[:space:]' || true)"
PRE_DEPLOYED_VERSION_RAW="$(echo "$PRE_HEALTH" | jq -r '.version // empty' 2>/dev/null | tr -d '[:space:]' || true)"
PRE_DEPLOYED_VERSION="${PRE_DEPLOYED_VERSION_RAW#v}"
PRE_DEPLOYED_VERSION="${PRE_DEPLOYED_VERSION#V}"

# Check for RENDER_API_KEY
if [ -z "${RENDER_API_KEY:-}" ]; then
    echo "ERROR: RENDER_API_KEY not set"
    echo "Get it from: brainops_credentials table or ~/.bashrc"
    exit 1
fi

# Check required CLIs
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required (used to parse Render + health responses)"
  exit 1
fi

# Step 1: Build Docker image with both tags
echo ""
echo "Step 1: Building Docker image..."
docker build -t mwwoodworth/brainops-ai-agents:latest -t mwwoodworth/brainops-ai-agents:v$VERSION .

# Step 2: Push to Docker Hub
echo ""
echo "Step 2: Pushing to Docker Hub..."
docker push mwwoodworth/brainops-ai-agents:latest
docker push mwwoodworth/brainops-ai-agents:v$VERSION

# Step 3: Push to Git
echo ""
echo "Step 3: Pushing to Git..."
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  AHEAD_COUNT="$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")"
  if [ "${AHEAD_COUNT:-0}" -gt 0 ]; then
    git push || echo "Git push failed"
  else
    echo "No new commits to push"
  fi
else
  echo "No upstream configured; skipping git push"
fi

# Step 4: Trigger Render deploy via API
echo ""
echo "Step 4: Triggering Render deploy..."
DEPLOY_RESPONSE=$(curl -s -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" \
    -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json")

DEPLOY_ID=$(echo "$DEPLOY_RESPONSE" | jq -r '.id // empty')
DEPLOY_STATUS=$(echo "$DEPLOY_RESPONSE" | jq -r '.status // empty')
if [ -z "${DEPLOY_ID:-}" ] || [ -z "${DEPLOY_STATUS:-}" ]; then
  echo "ERROR: Render deploy trigger returned unexpected response:"
  echo "$DEPLOY_RESPONSE" | jq . || echo "$DEPLOY_RESPONSE"
  exit 1
fi

echo "Deploy ID: $DEPLOY_ID"
echo "Status: $DEPLOY_STATUS"

# Step 5: Wait and verify
echo ""
echo "Step 5: Waiting for deploy..."
for i in {1..30}; do
  sleep 10
  STATUS_RESPONSE=$(curl -s "https://api.render.com/v1/services/$SERVICE_ID/deploys/$DEPLOY_ID" \
    -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json" || true)

  LIVE_STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
  if [ "$LIVE_STATUS" = "live" ]; then
    break
  fi
  if [ "$LIVE_STATUS" = "failed" ] || [ "$LIVE_STATUS" = "canceled" ]; then
    if [ "$LIVE_STATUS" = "canceled" ]; then
      # Render sometimes cancels an API-triggered deploy when a newer deploy is queued
      # (e.g., auto-deploy from image updates). Follow the newest deploy instead of failing.
      echo "WARN: Render deploy $DEPLOY_ID was canceled; checking for a superseding deploy..."
      NEWEST_RESPONSE=$(curl -s "https://api.render.com/v1/services/$SERVICE_ID/deploys?limit=2" \
        -H "Authorization: Bearer $RENDER_API_KEY" \
        -H "Content-Type: application/json" || true)
      NEWEST_ID=$(echo "$NEWEST_RESPONSE" | jq -r '.[0].deploy.id // empty' 2>/dev/null || true)
      NEWEST_STATUS=$(echo "$NEWEST_RESPONSE" | jq -r '.[0].deploy.status // empty' 2>/dev/null || true)
      if [ -n "${NEWEST_ID:-}" ] && [ "$NEWEST_ID" != "$DEPLOY_ID" ] && [ "$NEWEST_STATUS" != "canceled" ] && [ "$NEWEST_STATUS" != "failed" ]; then
        echo "  Following newer deploy: $NEWEST_ID ($NEWEST_STATUS)"
        DEPLOY_ID="$NEWEST_ID"
        continue
      fi
    fi

    echo "ERROR: Render deploy status: $LIVE_STATUS"
    echo "$STATUS_RESPONSE" | jq . || echo "$STATUS_RESPONSE"
    exit 1
  fi
  echo "  Render status: $LIVE_STATUS"
done

HEALTH=""
# Render may report `live` before the app instance has fully recycled; wait briefly for /health to reflect the new build.
for j in {1..12}; do
  HEALTH="$(curl -s "https://brainops-ai-agents.onrender.com/health?force_refresh=true" 2>/dev/null || true)"
  DEPLOYED_BUILD_CANDIDATE="$(echo "$HEALTH" | jq -r '.build // empty' 2>/dev/null | tr -d '[:space:]' || true)"
  if [ -n "${DEPLOYED_BUILD_CANDIDATE:-}" ] && [ "${PRE_DEPLOYED_BUILD:-}" != "${DEPLOYED_BUILD_CANDIDATE:-}" ]; then
    break
  fi
  sleep 5
done
DEPLOYED_VERSION=$(echo "$HEALTH" | jq -r '.version // ""' | tr -d '[:space:]')
# Normalize deployed version to compare apples-to-apples
DEPLOYED_VERSION="${DEPLOYED_VERSION#v}"
DEPLOYED_VERSION="${DEPLOYED_VERSION#V}"
DEPLOYED_BUILD="$(echo "$HEALTH" | jq -r '.build // ""' | tr -d '[:space:]')"
DB_STATUS=$(echo "$HEALTH" | jq -r '.database')
HEALTH_STATUS=$(echo "$HEALTH" | jq -r '.status // "unknown"')

echo ""
echo "=========================================="
BUILD_CHANGED="false"
if [ -n "${PRE_DEPLOYED_BUILD:-}" ] && [ -n "${DEPLOYED_BUILD:-}" ] && [ "$DEPLOYED_BUILD" != "$PRE_DEPLOYED_BUILD" ]; then
  BUILD_CHANGED="true"
fi

if [ "$BUILD_CHANGED" = "true" ]; then
  echo "✅ SUCCESS: Deploy is live (build changed)"
  echo "   Build:     $DEPLOYED_BUILD"
  echo "   Health:    $HEALTH_STATUS"
  echo "   Database:  $DB_STATUS"
  if [ "$DEPLOYED_VERSION" = "$VERSION" ]; then
    echo "   Version:   v$DEPLOYED_VERSION"
  else
    echo "   Version:   v$DEPLOYED_VERSION (local tag: v$VERSION)"
  fi
else
  if [ "$DEPLOYED_VERSION" = "$VERSION" ]; then
      echo "✅ SUCCESS: v$VERSION deployed"
      echo "   Health:    $HEALTH_STATUS"
      echo "   Database:  $DB_STATUS"
  else
      echo "⚠️  Could not confirm deploy via version/build"
      echo "   Pre-build: ${PRE_DEPLOYED_BUILD:-unknown}"
      echo "   Post-build:${DEPLOYED_BUILD:-unknown}"
      echo "   Expected:  v$VERSION"
      echo "   Current:   v$DEPLOYED_VERSION"
      echo ""
      echo "Check status: curl -s \"https://brainops-ai-agents.onrender.com/health?force_refresh=true\" | jq '{version,build,status,database}'"
  fi
fi
echo "=========================================="
