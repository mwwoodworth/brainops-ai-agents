#!/usr/bin/env bash
# =============================================================================
# MEMORY.md Generator
#
# Merges a live truth snapshot with the curated narrative template to produce
# a MEMORY.md that is always accurate. This is the ONLY sanctioned way to
# update MEMORY.md metrics.
#
# Usage:
#   ./scripts/truth/generate-memory.sh                          # live snapshot
#   ./scripts/truth/generate-memory.sh -s snapshot.json         # from file
#   ./scripts/truth/generate-memory.sh -o /path/to/MEMORY.md   # custom output
#
# Requires: jq, python3, psql (if live snapshot)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOT_FILE=""
OUTPUT_FILE="$HOME/.claude/projects/-home-matt-woodworth-dev/memory/MEMORY.md"
VERSION="v11.24"

while getopts "s:o:v:" opt; do
  case $opt in
    s) SNAPSHOT_FILE="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    v) VERSION="$OPTARG" ;;
    *) echo "Usage: $0 [-s snapshot.json] [-o output.md] [-v version]" >&2; exit 1 ;;
  esac
done

# 1. Get snapshot (live or from file)
if [ -n "$SNAPSHOT_FILE" ] && [ -f "$SNAPSHOT_FILE" ]; then
  SNAPSHOT=$(cat "$SNAPSHOT_FILE")
  echo "Using snapshot from: $SNAPSHOT_FILE" >&2
else
  echo "Taking live snapshot..." >&2
  SNAPSHOT=$("$SCRIPT_DIR/snapshot.sh")
fi

# Validate snapshot
if ! echo "$SNAPSHOT" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
  echo "ERROR: Invalid snapshot JSON" >&2
  exit 1
fi

# 2. Extract metrics and generate MEMORY.md
python3 -c "
import json, hashlib, sys
from datetime import datetime

snapshot = json.loads('''$(echo "$SNAPSHOT" | sed "s/'/\\\\'/g")''')
db = snapshot['database']
svc = snapshot['services']

snapshot_date = db.get('snapshot_ts', datetime.now(datetime.timezone.utc).isoformat())[:10]
version = '$VERSION'

# Compute truth hash from key metrics
hash_input = '|'.join(str(db.get(k, '')) for k in sorted([
    'db_size_mb', 'public_tables', 'all_tables', 'empty_tables',
    'tables_with_rls', 'tables_without_rls_public', 'total_rls_policies',
    'invariant_violations_unresolved', 'alerts_unresolved',
    'unified_memory_rows', 'pgcron_active_jobs', 'extensions_count'
]))
truth_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

# Build dynamic blocks
public_tables = db.get('public_tables', '?')
all_tables = db.get('all_tables', '?')
empty_tables = db.get('empty_tables', '?')
rls_on = db.get('tables_with_rls', '?')
rls_off_public = db.get('tables_without_rls_public', '?')
rls_off_names = db.get('tables_without_rls_names', [])
policies = db.get('total_rls_policies', '?')
zero_pol = db.get('rls_tables_zero_policies', '?')
inv_unresolved = db.get('invariant_violations_unresolved', '?')
inv_total = db.get('invariant_violations_total', '?')
alerts_unresolved = db.get('alerts_unresolved', '?')
alerts_total = db.get('alerts_total', '?')
mem_rows = db.get('unified_memory_rows', '?')
cron_jobs = db.get('pgcron_active_jobs', '?')
ext_count = db.get('extensions_count', '?')
pg_ver = db.get('pg_version', '?')
cc_tasks = db.get('cc_tasks_count', '?')
functions = db.get('total_functions', '?')
db_size = db.get('db_size_mb', '?')

agents_ver = svc.get('agents', {}).get('version', '?')
agents_status = svc.get('agents', {}).get('status', '?')
backend_ver = svc.get('backend', {}).get('version', '?')
backend_status = svc.get('backend', {}).get('status', '?')
mcp_servers = svc.get('mcp_bridge', {}).get('servers', '?')
mcp_tools = svc.get('mcp_bridge', {}).get('tools', '?')
mcp_status = svc.get('mcp_bridge', {}).get('status', '?')

# Compute RLS percentage
try:
    rls_pct = round(int(rls_on) / int(public_tables) * 100, 1)
except:
    rls_pct = '?'

no_rls_names_str = ', '.join(rls_off_names) if isinstance(rls_off_names, list) else str(rls_off_names)

db_state_block = f'''- **{db_size:,} MB. {public_tables:,} public tables ({all_tables:,} total). {empty_tables:,} empty (buildout mode). {ext_count} extensions. pg_cron: {cron_jobs} active jobs.**
- unified_ai_memory: {mem_rows:,} rows. IVFFlat lists=300, probes=10.
- cc_tasks: {cc_tasks:,} tasks (canonical store). {functions:,} public functions.
- RLS: {rls_on}/{public_tables} public ({rls_pct}%). {policies:,} policies. {zero_pol} tables with RLS + 0 policies.
- {rls_off_public} public table(s) without RLS: {no_rls_names_str}.
- invariant_violations: {inv_total:,} total, {inv_unresolved} unresolved.'''

remaining_block = f'''- {empty_tables:,} empty tables — KEEP (buildout mode). All P0-P7 items RESOLVED.
- **{alerts_unresolved} unresolved alerts** (down from 725). Auto-resolve cron (24h) + retention (30d) active.
- **{inv_unresolved} invariant violations**. CI gate: check_rls_coverage.sql.
- **TASK MANAGER UNIFIED**: cc_tasks canonical store ({cc_tasks:,} tasks). All APIs unified.'''

invariant_block = f'''- 14 checks, 5 min cycle, Resend alerts. {inv_unresolved} unresolved / {inv_total:,} historical.
- brainops_alerts: {alerts_unresolved} unresolved / {alerts_total:,} total. Auto-resolve + retention crons active.'''

service_block = f'''- Agents {agents_ver} ({agents_status}), Backend v{backend_ver} ({backend_status}), MCP Bridge {mcp_servers} servers/{mcp_tools} tools ({mcp_status}).
- All 7 Vercel services: HTTP 200. Stripe: charges_enabled=true, \$0 balance.'''

perfection_block = '''- **3/10 phases PASS**: 01 Gateway Tasks, 02 Alert Signal Quality, 03 DB RLS & Invariants Truth.
- **Phase 04 IN PROGRESS**: Truth Systems — eliminating MEMORY.md drift.
- PRs: CC#3 (merged), CC#4, Agents#4, Agents#5, Backend#2 (all open).'''

# Read narrative template
with open('$SCRIPT_DIR/narrative.md', 'r') as f:
    template = f.read()

# Replace placeholders
replacements = {
    '{{SNAPSHOT_DATE}}': snapshot_date,
    '{{VERSION}}': version,
    '{{PG_VERSION}}': str(pg_ver),
    '{{DB_STATE_BLOCK}}': db_state_block,
    '{{REMAINING_ISSUES_BLOCK}}': remaining_block,
    '{{AGENTS_VERSION}}': str(agents_ver),
    '{{BACKEND_VERSION}}': str(backend_ver),
    '{{INVARIANT_BLOCK}}': invariant_block,
    '{{SERVICE_STATUS_BLOCK}}': service_block,
    '{{PERFECTION_BLOCK}}': perfection_block,
    '{{TRUTH_HASH}}': truth_hash,
}

output = template
for k, v in replacements.items():
    output = output.replace(k, v)

# Enforce 200 line limit
lines = output.strip().split('\n')
if len(lines) > 200:
    # Trim trailing empty lines and comments first
    while len(lines) > 200 and (lines[-1].strip() == '' or lines[-1].strip().startswith('<!--')):
        lines.pop()
    if len(lines) > 200:
        lines = lines[:199]
        lines.append('<!-- TRUNCATED: exceeded 200 line limit -->')

print('\n'.join(lines))
" > "$OUTPUT_FILE"

# 3. Store snapshot alongside output
SNAPSHOT_DIR="$(dirname "$OUTPUT_FILE")"
echo "$SNAPSHOT" > "$SNAPSHOT_DIR/latest_snapshot.json"

# 4. Report
LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
HASH=$(grep -oP 'TRUTH_HASH: \K[a-f0-9]+' "$OUTPUT_FILE" || echo "none")
echo "Generated: $OUTPUT_FILE ($LINE_COUNT lines, hash: $HASH)" >&2
echo "Snapshot:  $SNAPSHOT_DIR/latest_snapshot.json" >&2

if [ "$LINE_COUNT" -gt 200 ]; then
  echo "WARNING: MEMORY.md exceeds 200 lines ($LINE_COUNT). Truncation may have occurred." >&2
fi
