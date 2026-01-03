#!/bin/bash
source /home/matt-woodworth/dev/_secure/BrainOps.env

echo "=== Render Service Details ==="
curl -s "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10" \
  -H "Authorization: Bearer $RENDER_API_KEY" | jq '.type, .serviceDetails'

echo ""
echo "=== Latest Deploy Status ==="
curl -s "https://api.render.com/v1/services/srv-d413iu75r7bs738btc10/deploys?limit=1" \
  -H "Authorization: Bearer $RENDER_API_KEY" | jq '.[0]'
