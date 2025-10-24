#!/bin/bash

# Fix Render Start Command Script
# This fixes the issue where Render is running app.py instead of main_integration.py

echo "========================================="
echo "Fixing Render Start Command"
echo "========================================="
echo ""

# Option 1: Direct Python execution (RECOMMENDED)
echo "Option 1: Direct Python Command"
echo "Change start command in Render dashboard to:"
echo ""
echo "  python main_integration.py"
echo ""
echo "This is the simplest and most reliable option."
echo ""

# Option 2: Uvicorn with correct module
echo "Option 2: Uvicorn Command (if Option 1 doesn't work)"
echo "Change start command to:"
echo ""
echo "  uvicorn main_integration:app --host 0.0.0.0 --port 10000"
echo ""
echo "This explicitly tells uvicorn to use main_integration.py"
echo ""

# Option 3: Update startup.sh (if you can't change Render dashboard)
echo "Option 3: Update startup.sh"
echo "If you can't access Render dashboard, we'll fix startup.sh:"
echo ""

# Check current startup.sh
if [ -f startup.sh ]; then
    echo "Current startup.sh runs:"
    grep "python" startup.sh | head -2
    echo ""

    # Create fixed version
    cp startup.sh startup.sh.backup
    sed -i 's/python3 run.py/python3 main_integration.py/g' startup.sh
    sed -i 's/python3 app.py/python3 main_integration.py/g' startup.sh

    echo "Fixed startup.sh to run:"
    grep "python" startup.sh | head -2
    echo ""
    echo "To apply this fix:"
    echo "  git add startup.sh"
    echo "  git commit -m 'fix: Run main_integration.py instead of app.py'"
    echo "  git push origin main"
fi

# Option 4: Create new startup script
echo ""
echo "Option 4: Create New Startup Script"
cat > start_production.py << 'EOF'
#!/usr/bin/env python3
"""
Production startup script for BrainOps AI Agents
Ensures the new integrated system runs
"""

import os
import sys
import uvicorn

# Set environment variables
os.environ['PYTHONUNBUFFERED'] = '1'

# Import and run the main integration app
from main_integration import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"

    print(f"🚀 Starting BrainOps AI OS - Integrated System")
    print(f"Running on http://{host}:{port}")
    print(f"AUREA, Memory, Board, and all 59 agents activated")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
EOF

chmod +x start_production.py

echo "Created start_production.py"
echo "To use this, set Render start command to:"
echo "  python start_production.py"
echo ""

# Show the current vs needed
echo "========================================="
echo "CURRENT vs NEEDED"
echo "========================================="
echo ""
echo "CURRENT (WRONG):"
echo "  uvicorn app:app --host 0.0.0.0 --port 10000"
echo ""
echo "NEEDED (CORRECT):"
echo "  uvicorn main_integration:app --host 0.0.0.0 --port 10000"
echo "  OR"
echo "  python main_integration.py"
echo ""

# API command if we had the key
echo "========================================="
echo "IF YOU HAVE RENDER API KEY:"
echo "========================================="
echo ""
echo "You could use this curl command:"
echo ""
cat << 'APICOMMAND'
curl -X PATCH https://api.render.com/v1/services/YOUR_SERVICE_ID \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "startCommand": "python main_integration.py"
  }'
APICOMMAND
echo ""
echo "Get your API key from: https://dashboard.render.com/account/settings"
echo "Get service ID from: Dashboard → Service → Settings → ID"
echo ""

echo "========================================="
echo "VERIFICATION"
echo "========================================="
echo ""
echo "After changing the start command and redeploying, these should work:"
echo "  curl https://brainops-ai-agents.onrender.com/aurea/status"
echo "  curl https://brainops-ai-agents.onrender.com/memory/stats"
echo "  curl https://brainops-ai-agents.onrender.com/board/members"
echo ""
echo "Run this to verify:"
echo "  python3 /home/matt-woodworth/dev/test-after-fix.py"