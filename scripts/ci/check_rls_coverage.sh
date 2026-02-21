#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# CI GATE: RLS Coverage Check
# Exits non-zero if any public non-system table lacks RLS.
#
# Usage:
#   ./scripts/ci/check_rls_coverage.sh
#
# Requires: DATABASE_URL or DB_HOST/DB_USER/DB_PASSWORD env vars
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SQL_FILE="$SCRIPT_DIR/check_rls_coverage.sql"

if [ -n "${DATABASE_URL:-}" ]; then
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$SQL_FILE"
elif [ -n "${DB_HOST:-}" ] && [ -n "${DB_USER:-}" ] && [ -n "${DB_PASSWORD:-}" ]; then
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "${DB_NAME:-postgres}" \
    -p "${DB_PORT:-6543}" -v ON_ERROR_STOP=1 -f "$SQL_FILE"
else
  echo "ERROR: Set DATABASE_URL or DB_HOST/DB_USER/DB_PASSWORD"
  exit 1
fi

echo "RLS coverage gate: PASSED"
