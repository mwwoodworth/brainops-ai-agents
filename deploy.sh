#!/bin/bash
# BrainOps AI Agents - Full Automated Deployment
# Builds, pushes to Docker Hub, triggers Render deploy via API
set -e

# Get version from app.py
VERSION=$(grep "^VERSION" app.py | cut -d'"' -f2)
SERVICE_ID="srv-d413iu75r7bs738btc10"

echo "=========================================="
echo "Deploying BrainOps AI Agents v$VERSION"
echo "=========================================="

# Check for RENDER_API_KEY
if [ -z "$RENDER_API_KEY" ]; then
    echo "ERROR: RENDER_API_KEY not set"
    echo "Get it from: brainops_credentials table or ~/.bashrc"
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
git push || echo "Git push failed or nothing to push"

# Step 4: Trigger Render deploy via API
echo ""
echo "Step 4: Triggering Render deploy..."
DEPLOY_RESPONSE=$(curl -s -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" \
    -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json")

DEPLOY_ID=$(echo "$DEPLOY_RESPONSE" | jq -r '.deploy.id // "unknown"')
DEPLOY_STATUS=$(echo "$DEPLOY_RESPONSE" | jq -r '.deploy.status // "unknown"')

echo "Deploy ID: $DEPLOY_ID"
echo "Status: $DEPLOY_STATUS"

# Step 5: Wait and verify
echo ""
echo "Step 5: Waiting for deploy..."
sleep 30

HEALTH=$(curl -s https://brainops-ai-agents.onrender.com/health)
DEPLOYED_VERSION=$(echo "$HEALTH" | jq -r '.version')
DB_STATUS=$(echo "$HEALTH" | jq -r '.database')

echo ""
echo "=========================================="
if [ "$DEPLOYED_VERSION" = "$VERSION" ]; then
    echo "✅ SUCCESS: v$VERSION deployed"
    echo "   Database: $DB_STATUS"
else
    echo "⏳ Deploy in progress..."
    echo "   Expected: $VERSION"
    echo "   Current:  $DEPLOYED_VERSION"
    echo ""
    echo "Check status: curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'"
fi
echo "=========================================="
