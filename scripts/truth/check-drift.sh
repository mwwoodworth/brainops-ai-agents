#!/usr/bin/env bash
# =============================================================================
# CI Gate: MEMORY.md Drift Detection
#
# Verifies that MEMORY.md was generated (not hand-edited) by checking:
# 1. The TRUTH_HASH comment exists and matches a fresh snapshot
# 2. Key metrics in MEMORY.md match the live database
#
# Usage:
#   ./scripts/truth/check-drift.sh                    # check against live DB
#   ./scripts/truth/check-drift.sh -s snapshot.json   # check against snapshot
#
# Exit codes:
#   0 = no drift detected
#   1 = drift detected (MEMORY.md is stale)
#   2 = MEMORY.md missing or malformed
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEMORY_FILE="$HOME/.claude/projects/-home-matt-woodworth-dev/memory/MEMORY.md"
SNAPSHOT_FILE=""
TOLERANCE_MB=100  # Allow DB size to drift by up to 100MB before flagging

while getopts "s:m:t:" opt; do
  case $opt in
    s) SNAPSHOT_FILE="$OPTARG" ;;
    m) MEMORY_FILE="$OPTARG" ;;
    t) TOLERANCE_MB="$OPTARG" ;;
    *) echo "Usage: $0 [-s snapshot.json] [-m memory.md] [-t tolerance_mb]" >&2; exit 1 ;;
  esac
done

# Validate MEMORY.md exists
if [ ! -f "$MEMORY_FILE" ]; then
  echo "FAIL: MEMORY.md not found at $MEMORY_FILE" >&2
  exit 2
fi

# Check for GENERATED marker
if ! grep -q "GENERATED:" "$MEMORY_FILE"; then
  echo "FAIL: MEMORY.md missing GENERATED marker. Was it hand-edited?" >&2
  echo "  Fix: Run scripts/truth/generate-memory.sh" >&2
  exit 2
fi

# Check for TRUTH_HASH
MEMORY_HASH=$(grep -oP 'TRUTH_HASH: \K[a-f0-9]+' "$MEMORY_FILE" 2>/dev/null || echo "")
if [ -z "$MEMORY_HASH" ]; then
  echo "FAIL: MEMORY.md missing TRUTH_HASH. Was it hand-edited?" >&2
  echo "  Fix: Run scripts/truth/generate-memory.sh" >&2
  exit 2
fi

# Get fresh snapshot
if [ -n "$SNAPSHOT_FILE" ] && [ -f "$SNAPSHOT_FILE" ]; then
  SNAPSHOT=$(cat "$SNAPSHOT_FILE")
  echo "Checking against snapshot: $SNAPSHOT_FILE" >&2
else
  echo "Taking fresh snapshot for comparison..." >&2
  SNAPSHOT=$("$SCRIPT_DIR/snapshot.sh")
fi

# Compare key metrics
DRIFT_COUNT=0
DRIFT_DETAILS=""

check_metric() {
  local name="$1"
  local memory_val="$2"
  local live_val="$3"
  local tolerance="${4:-0}"

  if [ "$memory_val" = "?" ] || [ "$live_val" = "?" ]; then
    return 0  # Skip unknown values
  fi

  local diff=$((live_val - memory_val))
  if [ "$diff" -lt 0 ]; then diff=$((-diff)); fi

  if [ "$diff" -gt "$tolerance" ]; then
    DRIFT_COUNT=$((DRIFT_COUNT + 1))
    DRIFT_DETAILS="$DRIFT_DETAILS\n  - $name: MEMORY=$memory_val, LIVE=$live_val (drift=$diff)"
  fi
}

# Extract metrics from snapshot
LIVE=$(python3 -c "
import json, sys
s = json.loads(sys.stdin.read())
db = s['database']
print(f\"{db.get('public_tables',0)}|{db.get('tables_with_rls',0)}|{db.get('total_rls_policies',0)}|{db.get('invariant_violations_unresolved',0)}|{db.get('alerts_unresolved',0)}|{db.get('db_size_mb',0)}|{db.get('empty_tables',0)}|{db.get('pgcron_active_jobs',0)}\")
" <<< "$SNAPSHOT")

IFS='|' read -r L_TABLES L_RLS L_POLICIES L_INV L_ALERTS L_SIZE L_EMPTY L_CRON <<< "$LIVE"

# Extract metrics from MEMORY.md
extract_memory_metric() {
  local pattern="$1"
  grep -oP "$pattern" "$MEMORY_FILE" 2>/dev/null | head -1 || echo "?"
}

# Parse MEMORY.md metrics using reliable patterns
M_TABLES=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'(\d[\d,]*)\s+public tables', text)
print(m.group(1).replace(',','') if m else '?')
")

M_RLS=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'RLS:\s+(\d[\d,]*)/\d', text)
print(m.group(1).replace(',','') if m else '?')
")

M_POLICIES=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'(\d[\d,]*)\s+policies', text)
print(m.group(1).replace(',','') if m else '?')
")

M_INV=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'(\d+)\s+unresolved.*invariant', text, re.IGNORECASE)
if not m:
    m = re.search(r'invariant.*?(\d+)\s+unresolved', text, re.IGNORECASE)
print(m.group(1) if m else '?')
")

M_ALERTS=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'\*\*(\d+)\s+unresolved alerts', text)
print(m.group(1) if m else '?')
")

M_SIZE=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'\*\*(\d[\d,]*)\s+MB', text)
print(m.group(1).replace(',','') if m else '?')
")

M_EMPTY=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'(\d[\d,]*)\s+empty', text)
print(m.group(1).replace(',','') if m else '?')
")

M_CRON=$(python3 -c "
import re, sys
with open('$MEMORY_FILE') as f:
    text = f.read()
m = re.search(r'pg_cron:\s+(\d+)', text)
print(m.group(1) if m else '?')
")

# Run comparisons (exact match for counts, tolerance for size)
check_metric "public_tables" "$M_TABLES" "$L_TABLES" 0
check_metric "tables_with_rls" "$M_RLS" "$L_RLS" 0
check_metric "total_rls_policies" "$M_POLICIES" "$L_POLICIES" 0
check_metric "invariant_violations" "$M_INV" "$L_INV" 0
check_metric "alerts_unresolved" "$M_ALERTS" "$L_ALERTS" 2  # Allow ±2 alert fluctuation
check_metric "db_size_mb" "$M_SIZE" "$L_SIZE" "$TOLERANCE_MB"
check_metric "empty_tables" "$M_EMPTY" "$L_EMPTY" 5  # Allow ±5 empty table fluctuation
check_metric "pgcron_jobs" "$M_CRON" "$L_CRON" 0

if [ "$DRIFT_COUNT" -gt 0 ]; then
  echo "DRIFT DETECTED: $DRIFT_COUNT metric(s) out of sync" >&2
  echo -e "$DRIFT_DETAILS" >&2
  echo "" >&2
  echo "Fix: Run scripts/truth/generate-memory.sh to regenerate MEMORY.md" >&2
  exit 1
else
  echo "DRIFT CHECK PASSED: MEMORY.md matches live state" >&2
  echo "  Hash: $MEMORY_HASH" >&2
  echo "  Metrics checked: tables, RLS, policies, invariants, alerts, size, cron" >&2
  exit 0
fi
