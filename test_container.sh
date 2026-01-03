#!/bin/bash
source /home/matt-woodworth/dev/_secure/BrainOps.env

# Stop and remove any existing test container
docker rm -f test-health 2>/dev/null

# Run the container
echo "Starting container..."
docker run -d --name test-health -p 10000:10000 \
  -e "DATABASE_URL=$DATABASE_URL" \
  -e "PORT=10000" \
  -e "OPENAI_API_KEY=$OPENAI_API_KEY" \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  -e "BRAINOPS_API_KEY=brainops_prod_key_2025" \
  -e "ENVIRONMENT=test" \
  mwwoodworth/brainops-ai-agents:v9.99.7

echo "Waiting for startup..."
sleep 25

echo "=== Testing health endpoint ==="
curl -s http://localhost:10000/health | jq '.'

echo ""
echo "=== Container logs ==="
docker logs test-health 2>&1 | tail -50

echo ""
echo "=== Cleanup ==="
docker rm -f test-health
