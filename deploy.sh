#!/bin/bash
# BrainOps AI Agents - Deployment Script
# This script builds, pushes to Docker Hub, and prepares for Render deploy
set -e

# Get version from app.py
VERSION=$(grep "^VERSION" app.py | cut -d'"' -f2)

echo "=========================================="
echo "Deploying BrainOps AI Agents v$VERSION"
echo "=========================================="

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

echo ""
echo "=========================================="
echo "✅ v$VERSION pushed to Docker Hub"
echo "=========================================="
echo ""
echo "NEXT STEP: Trigger deploy in Render dashboard"
echo "  → https://dashboard.render.com"
echo "  → brainops-ai-agents → Manual Deploy"
echo ""
echo "Or use deploy hook:"
echo "  curl -X POST \"\$RENDER_DEPLOY_HOOK\""
echo ""
echo "Verify after deploy:"
echo "  curl -s https://brainops-ai-agents.onrender.com/health | jq '.version'"
