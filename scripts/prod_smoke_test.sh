#!/usr/bin/env bash
set -u
set -o pipefail

# Production smoke tests for BrainOps services + DB.
#
# Usage:
#   AGENTS_API_KEY=... DB_PASSWORD=... ./scripts/prod_smoke_test.sh
#
# Optional env:
#   AGENTS_URL=https://brainops-ai-agents.onrender.com
#   BACKEND_URL=https://brainops-backend-prod.onrender.com
#   MCP_URL=https://brainops-mcp-bridge.onrender.com
#   BACKEND_API_KEY=...   (if backend endpoints require auth)
#   MCP_API_KEY=...       (if MCP Bridge requires auth)
#   DATABASE_URL=postgres://user:pass@host:5432/dbname?sslmode=require
#   DB_HOST=... DB_PORT=... DB_USER=... DB_NAME=... DB_PASSWORD=...
#   RUN_REAL_AGENT_EXEC=true   (executes /agents/{id}/execute; may call paid AI APIs)

AGENTS_URL="${AGENTS_URL:-https://brainops-ai-agents.onrender.com}"
BACKEND_URL="${BACKEND_URL:-https://brainops-backend-prod.onrender.com}"
MCP_URL="${MCP_URL:-https://brainops-mcp-bridge.onrender.com}"

AGENTS_API_KEY="${AGENTS_API_KEY:-${X_API_KEY:-}}"
BACKEND_API_KEY="${BACKEND_API_KEY:-}"
MCP_API_KEY="${MCP_API_KEY:-}"

RUN_REAL_AGENT_EXEC="${RUN_REAL_AGENT_EXEC:-false}"

PASS=0
FAIL=0
WARN=0

have() { command -v "$1" >/dev/null 2>&1; }

indent() { sed 's/^/  /'; }

http_call() {
  local name="$1"
  local method="$2"
  local url="$3"
  local data="${4:-}"
  local api_key="${5:-}"

  local header_file body_file
  header_file="$(mktemp)"
  body_file="$(mktemp)"

  local curl_args=(
    -sS
    --max-time 25
    --connect-timeout 10
    -X "$method"
    -H "Accept: application/json"
    -D "$header_file"
    -o "$body_file"
  )

  if [[ -n "$api_key" ]]; then
    curl_args+=(-H "X-API-Key: $api_key")
  fi

  if [[ -n "$data" ]]; then
    curl_args+=(-H "Content-Type: application/json" --data "$data")
  fi

  local http_code curl_rc
  http_code=""
  set +e
  http_code="$(curl "${curl_args[@]}" "$url" -w "%{http_code}")"
  curl_rc=$?

  echo "==> $name"
  echo "URL: $method $url"

  if [[ $curl_rc -ne 0 ]]; then
    echo "RESULT: curl failed (exit=$curl_rc)"
    echo "BODY:"
    cat "$body_file" 2>/dev/null | indent || true
    echo
    rm -f "$header_file" "$body_file"
    FAIL=$((FAIL + 1))
    return 0
  fi

  echo "HTTP: $http_code"

  local content_type
  content_type="$(tr -d '\r' <"$header_file" | awk 'BEGIN{IGNORECASE=1} /^content-type:/{print $0; exit}')"
  if [[ -n "$content_type" ]]; then
    echo "$content_type"
  fi

  echo "BODY:"
  if have jq && jq -e . "$body_file" >/dev/null 2>&1; then
    jq . "$body_file" | indent
  else
    cat "$body_file" | indent
  fi
  echo

  if [[ "$http_code" =~ ^2|^3 ]]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
  fi

  rm -f "$header_file" "$body_file"
  return 0
}

psql_call() {
  local name="$1"
  local sql="$2"

  if ! have psql; then
    echo "==> $name"
    echo "RESULT: skipped (psql not installed)"
    echo
    WARN=$((WARN + 1))
    return 0
  fi

  local out rc
  set +e
  if [[ -n "${DATABASE_URL:-}" ]]; then
    out="$(psql "$DATABASE_URL" -X -v ON_ERROR_STOP=1 -Atc "$sql" 2>&1)"
    rc=$?
  else
    if [[ -z "${DB_HOST:-}" || -z "${DB_USER:-}" || -z "${DB_NAME:-}" || -z "${DB_PASSWORD:-}" ]]; then
      echo "==> $name"
      echo "RESULT: skipped (set DATABASE_URL or DB_HOST/DB_USER/DB_NAME/DB_PASSWORD)"
      echo
      WARN=$((WARN + 1))
      return 0
    fi
    out="$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -p "${DB_PORT:-5432}" -X -v ON_ERROR_STOP=1 -Atc "$sql" 2>&1)"
    rc=$?
  fi

  echo "==> $name"
  echo "SQL: $sql"
  if [[ $rc -ne 0 ]]; then
    echo "RESULT: failed"
    echo "$out" | indent
    echo
    FAIL=$((FAIL + 1))
    return 0
  fi

  echo "RESULT:"
  echo "$out" | indent
  echo
  PASS=$((PASS + 1))
  return 0
}

echo "BrainOps Production Smoke Test"
echo "================================"
echo "AGENTS_URL : $AGENTS_URL"
echo "BACKEND_URL: $BACKEND_URL"
echo "MCP_URL    : $MCP_URL"
echo
if [[ -n "$AGENTS_API_KEY" ]]; then
  echo "AGENTS_API_KEY: set"
else
  echo "AGENTS_API_KEY: (not set)  -> required for POST $AGENTS_URL/execute and /agents/*"
fi
if [[ -n "$BACKEND_API_KEY" ]]; then
  echo "BACKEND_API_KEY: set"
else
  echo "BACKEND_API_KEY: (not set) -> some backend endpoints may 401/403"
fi
if [[ -n "$MCP_API_KEY" ]]; then
  echo "MCP_API_KEY: set"
else
  echo "MCP_API_KEY: (not set) -> /mcp/* may 401/403"
fi
echo

echo "1) AI Agents Service"
echo "--------------------"
http_call "AI Agents: /health" "GET" "$AGENTS_URL/health"
http_call "AI Agents: /ai/tasks/stats" "GET" "$AGENTS_URL/ai/tasks/stats"
http_call "AI Agents: /scheduler/status" "GET" "$AGENTS_URL/scheduler/status"
http_call "AI Agents: POST /execute (cron trigger)" "POST" "$AGENTS_URL/execute" '{}' "$AGENTS_API_KEY"

if [[ "$RUN_REAL_AGENT_EXEC" == "true" ]]; then
  if ! have jq; then
    echo "==> AI Agents: REAL agent execution"
    echo "RESULT: skipped (jq required to pick an agent id; install jq or run manually)"
    echo
    WARN=$((WARN + 1))
  elif [[ -z "$AGENTS_API_KEY" ]]; then
    echo "==> AI Agents: REAL agent execution"
    echo "RESULT: skipped (AGENTS_API_KEY required)"
    echo
    WARN=$((WARN + 1))
  else
    echo "==> AI Agents: REAL agent execution (may call paid AI APIs)"
    agent_id="$(curl -sS --max-time 20 -H "X-API-Key: $AGENTS_API_KEY" "$AGENTS_URL/agents" | jq -r '.agents[0].id // empty' 2>/dev/null)"
    if [[ -z "$agent_id" ]]; then
      echo "RESULT: failed to select an agent id from /agents"
      echo
      FAIL=$((FAIL + 1))
    else
      http_call "AI Agents: POST /agents/$agent_id/execute" "POST" "$AGENTS_URL/agents/$agent_id/execute" '{"task":"prod smoke test - reply with OK"}' "$AGENTS_API_KEY"
    fi
  fi
fi

echo "2) Backend API"
echo "--------------"
http_call "Backend: /health" "GET" "$BACKEND_URL/health"
http_call "Backend: /api/v1/health" "GET" "$BACKEND_URL/api/v1/health" "" "$BACKEND_API_KEY"
http_call "Backend: /api/v1/ai/status" "GET" "$BACKEND_URL/api/v1/ai/status" "" "$BACKEND_API_KEY"
http_call "Backend: /api/v1/customers" "GET" "$BACKEND_URL/api/v1/customers" "" "$BACKEND_API_KEY"
http_call "Backend: /api/v1/jobs" "GET" "$BACKEND_URL/api/v1/jobs" "" "$BACKEND_API_KEY"
http_call "Backend: /api/v1/invoices" "GET" "$BACKEND_URL/api/v1/invoices" "" "$BACKEND_API_KEY"

echo "3) MCP Bridge"
echo "-------------"
http_call "MCP Bridge: /health" "GET" "$MCP_URL/health"
http_call "MCP Bridge: /mcp/tools" "GET" "$MCP_URL/mcp/tools" "" "$MCP_API_KEY"
http_call "MCP Bridge: /mcp/tools/definitions" "GET" "$MCP_URL/mcp/tools/definitions" "" "$MCP_API_KEY"

echo "4) Database Verification"
echo "------------------------"
psql_call "DB: ai_agents count" "SELECT COUNT(*) FROM ai_agents;"
psql_call "DB: ai_autonomous_tasks count" "SELECT COUNT(*) FROM ai_autonomous_tasks;"
psql_call "DB: unified_ai_memory count" "SELECT COUNT(*) FROM unified_ai_memory;"
psql_call "DB: unified_ai_memory by memory_type" "SELECT memory_type, COUNT(*) FROM unified_ai_memory GROUP BY memory_type ORDER BY COUNT(*) DESC;"
psql_call "DB: ai_knowledge_graph count" "SELECT COUNT(*) FROM ai_knowledge_graph;"
psql_call "DB: ai_knowledge_graph by node_type" "SELECT node_type, COUNT(*) FROM ai_knowledge_graph GROUP BY node_type ORDER BY COUNT(*) DESC;"
psql_call "DB: ai_knowledge_nodes count (if present)" "SELECT COUNT(*) FROM ai_knowledge_nodes;"
psql_call "DB: ai_knowledge_edges count (if present)" "SELECT COUNT(*) FROM ai_knowledge_edges;"

echo "Summary"
echo "-------"
echo "PASS=$PASS FAIL=$FAIL WARN=$WARN"
if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
