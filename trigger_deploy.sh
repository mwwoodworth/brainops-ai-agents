#!/bin/bash
source /home/matt-woodworth/dev/_secure/BrainOps.env

echo "Triggering new deploy..."
RESULT=$(curl -s -X POST "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clearCache":"clear"}')

echo "$RESULT" | jq '.'
