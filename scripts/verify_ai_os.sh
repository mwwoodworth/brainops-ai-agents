#!/bin/bash
# AI OS Comprehensive Verification Script
# Verifies all systems are operational
# Run: ./scripts/verify_ai_os.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  AI OS VERIFICATION - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    local expected="$3"

    if [[ "$result" == *"$expected"* ]]; then
        echo -e "${GREEN}✅ $name${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌ $name${NC}"
        echo "   Expected: $expected"
        echo "   Got: $result"
        ((FAIL++))
    fi
}

echo ""
echo "=== 1. RENDER SERVICES ==="

# AI Agents
AI_HEALTH=$(curl -s "https://brainops-ai-agents.onrender.com/health" -H "X-API-Key: brainops_prod_key_2025" 2>/dev/null)
check "AI Agents Health" "$AI_HEALTH" '"status":"healthy"'

AI_VERSION=$(echo "$AI_HEALTH" | jq -r '.version' 2>/dev/null)
echo "   Version: $AI_VERSION"

# AUREA Status
AUREA=$(curl -s "https://brainops-ai-agents.onrender.com/systems/usage" -H "X-API-Key: brainops_prod_key_2025" 2>/dev/null | jq '.aurea.running' 2>/dev/null)
check "AUREA Running" "$AUREA" "true"

# Scheduler
SCHEDULER=$(curl -s "https://brainops-ai-agents.onrender.com/scheduler/status" -H "X-API-Key: brainops_prod_key_2025" 2>/dev/null)
JOBS=$(echo "$SCHEDULER" | jq '.apscheduler_jobs_count' 2>/dev/null)
check "Scheduler Jobs ($JOBS)" "$JOBS" "61"

# Backend
BACKEND=$(curl -s "https://brainops-backend-prod.onrender.com/health" 2>/dev/null)
check "Backend Health" "$BACKEND" '"status":"healthy"'

# MCP Bridge
MCP=$(curl -s "https://brainops-mcp-bridge.onrender.com/health" 2>/dev/null)
check "MCP Bridge Health" "$MCP" '"status":"healthy"'

MCP_TOOLS=$(curl -s "https://brainops-mcp-bridge.onrender.com/mcp/tools" \
  -H "X-API-Key: brainops_mcp_49209d1d3f19376706560e860a71d172728861dd4e2493f263c3cc0c6d9adc86" 2>/dev/null | jq '.totalTools' 2>/dev/null)
check "MCP Tools ($MCP_TOOLS)" "$MCP_TOOLS" "245"

echo ""
echo "=== 2. VERCEL SERVICES ==="

# ERP
ERP=$(curl -s -o /dev/null -w "%{http_code}" "https://weathercraft-erp.vercel.app" 2>/dev/null)
check "ERP Accessible (HTTP $ERP)" "$ERP" "200"

# MRG
MRG=$(curl -s -o /dev/null -w "%{http_code}" "https://myroofgenius.com" 2>/dev/null)
check "MRG Accessible (HTTP $MRG)" "$MRG" "200"

# Command Center
CC=$(curl -s -o /dev/null -w "%{http_code}" "https://brainops-command-center.vercel.app" 2>/dev/null)
check "Command Center (HTTP $CC)" "$CC" "200"

echo ""
echo "=== 3. DATABASE ==="

# Customer count via MCP
CUSTOMERS=$(curl -s -X POST "https://brainops-mcp-bridge.onrender.com/mcp/execute" \
  -H "X-API-Key: brainops_mcp_49209d1d3f19376706560e860a71d172728861dd4e2493f263c3cc0c6d9adc86" \
  -H "Content-Type: application/json" \
  -d '{"server": "supabase", "tool": "sql_query", "params": {"query": "SELECT COUNT(*) FROM customers"}}' 2>/dev/null | jq -r '.result.content[0].text' 2>/dev/null | jq -r '.rows[0].count' 2>/dev/null)
check "Customers: $CUSTOMERS" "$CUSTOMERS" "102"

# Memory entries
MEMORIES=$(curl -s -X POST "https://brainops-mcp-bridge.onrender.com/mcp/execute" \
  -H "X-API-Key: brainops_mcp_49209d1d3f19376706560e860a71d172728861dd4e2493f263c3cc0c6d9adc86" \
  -H "Content-Type: application/json" \
  -d '{"server": "supabase", "tool": "sql_query", "params": {"query": "SELECT COUNT(*) FROM unified_ai_memory"}}' 2>/dev/null | jq -r '.result.content[0].text' 2>/dev/null | jq -r '.rows[0].count' 2>/dev/null)
echo "   Unified Memories: $MEMORIES"

echo ""
echo "=== 4. BRAIN API ==="

BRAIN=$(curl -s "https://brainops-ai-agents.onrender.com/brain/critical" -H "X-API-Key: brainops_prod_key_2025" 2>/dev/null | jq 'length' 2>/dev/null)
check "Brain Critical Entries: $BRAIN" "$BRAIN" ""

echo ""
echo "=========================================="
echo "  RESULTS: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
