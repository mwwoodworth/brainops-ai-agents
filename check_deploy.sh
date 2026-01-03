#!/bin/bash
source /home/matt-woodworth/dev/_secure/BrainOps.env
curl -s "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys?limit=3" \
  -H "Authorization: Bearer $RENDER_API_KEY" | jq '.[].deploy | {id, status, createdAt}'

echo ""
echo "=== Production Health ==="
curl -s "https://brainops-ai-agents.onrender.com/health" \
  -H "X-API-Key: $BRAINOPS_API_KEY" | jq '{version, status, database}'
