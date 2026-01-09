#!/bin/bash
# Autonomous QA Agent Runner
# Usage: qa-agent [url]

URL=${1:-"http://localhost:3000"}

echo "🤖 BrainOps QA Agent: Starting Autonomous Audit on $URL..."

# Run Playwright in headless mode
cd /home/matt-woodworth/dev/brainops-command-center
npx playwright test tests/e2e/qa-agent/audit.spec.ts --reporter=line

if [ $? -eq 0 ]; then
    echo "✅ QA Audit PASSED. System is operational."
    # Store success in Brain
    python3 /home/matt-woodworth/dev/scripts/ai/brain-bridge.py store "QA Audit Passed: All systems nominal."
else
    echo "❌ QA Audit FAILED. Analyzing logs..."
    # Store failure in Brain
    python3 /home/matt-woodworth/dev/scripts/ai/brain-bridge.py store "QA Audit FAILED: Immediate attention required."
    exit 1
fi
