#!/usr/bin/env bash
# =============================================================================
# Truth Snapshot: Capture live system state as JSON
#
# Queries Supabase for all driftable DB metrics, then augments with service
# health endpoints for Docker versions and status.
#
# Usage:
#   ./scripts/truth/snapshot.sh                    # stdout
#   ./scripts/truth/snapshot.sh -o snapshot.json   # file
#
# Requires: DATABASE_URL, BRAINOPS_API_KEY env vars
#           (source ~/dev/_secure/BrainOps.env first)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SQL_FILE="$SCRIPT_DIR/snapshot.sql"
OUTPUT_FILE=""

while getopts "o:" opt; do
  case $opt in
    o) OUTPUT_FILE="$OPTARG" ;;
    *) echo "Usage: $0 [-o output.json]" >&2; exit 1 ;;
  esac
done

# Validate prerequisites
if [ -z "${DATABASE_URL:-}" ]; then
  echo "ERROR: DATABASE_URL not set. Run: source ~/dev/_secure/BrainOps.env" >&2
  exit 1
fi

# 1. Capture DB metrics
DB_JSON=$(psql "$DATABASE_URL" -t -A -f "$SQL_FILE" 2>/dev/null)
if [ -z "$DB_JSON" ]; then
  echo "ERROR: DB query returned empty result" >&2
  exit 1
fi

# 2. Capture service health (best-effort, don't fail on network errors)
API_KEY="${BRAINOPS_API_KEY:-${MASTER_API_KEY:-}}"

agents_health() {
  if [ -n "$API_KEY" ]; then
    curl -sf --max-time 10 "https://brainops-ai-agents.onrender.com/health" \
      -H "X-API-Key: $API_KEY" 2>/dev/null | \
      python3 -c "
import sys,json
d=json.load(sys.stdin)
print(json.dumps({'version':d.get('version','unknown'),'status':d.get('status','unknown')}))
" 2>/dev/null || echo '{"version":"unavailable","status":"unavailable"}'
  else
    echo '{"version":"no_api_key","status":"no_api_key"}'
  fi
}

backend_health() {
  if [ -n "$API_KEY" ]; then
    curl -sf --max-time 10 "https://brainops-backend-prod.onrender.com/health" \
      -H "X-API-Key: $API_KEY" 2>/dev/null | \
      python3 -c "
import sys,json
d=json.load(sys.stdin)
print(json.dumps({'version':d.get('version','unknown'),'status':d.get('status','unknown')}))
" 2>/dev/null || echo '{"version":"unavailable","status":"unavailable"}'
  else
    echo '{"version":"no_api_key","status":"no_api_key"}'
  fi
}

mcp_health() {
  curl -sf --max-time 10 "https://brainops-mcp-bridge.onrender.com/health" 2>/dev/null | \
    python3 -c "
import sys,json
d=json.load(sys.stdin)
print(json.dumps({
  'servers':d.get('mcpServers',d.get('connectedServers',d.get('servers','unknown'))),
  'tools':d.get('totalTools',d.get('tools','unknown')),
  'status':d.get('status','unknown')
}))
" 2>/dev/null || echo '{"servers":"unavailable","tools":"unavailable","status":"unavailable"}'
}

AGENTS_JSON=$(agents_health)
BACKEND_JSON=$(backend_health)
MCP_JSON=$(mcp_health)

# 3. Merge all into final snapshot
FINAL=$(python3 -c "
import json, sys

db = json.loads('''$DB_JSON''')
agents = json.loads('''$AGENTS_JSON''')
backend = json.loads('''$BACKEND_JSON''')
mcp = json.loads('''$MCP_JSON''')

snapshot = {
    'database': db,
    'services': {
        'agents': agents,
        'backend': backend,
        'mcp_bridge': mcp
    },
    'meta': {
        'generator': 'scripts/truth/snapshot.sh',
        'generator_version': '1.0.0'
    }
}

print(json.dumps(snapshot, indent=2, default=str))
")

if [ -n "$OUTPUT_FILE" ]; then
  echo "$FINAL" > "$OUTPUT_FILE"
  echo "Snapshot written to $OUTPUT_FILE" >&2
else
  echo "$FINAL"
fi
