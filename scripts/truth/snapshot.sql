-- =============================================================================
-- Truth Snapshot: Live Database Metrics
-- Outputs a single JSON object with all driftable metrics.
-- Usage: psql "$DATABASE_URL" -t -A -f snapshot.sql
-- =============================================================================

SELECT json_build_object(
  -- Database size
  'db_size_mb', (SELECT pg_database_size(current_database())::bigint / (1024*1024)),

  -- Table counts
  'public_tables', (
    SELECT COUNT(*) FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname = 'public'
  ),
  'all_tables', (
    SELECT COUNT(*) FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog','information_schema')
  ),
  'empty_tables', (
    SELECT COUNT(*) FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname = 'public' AND c.reltuples = 0
  ),

  -- RLS coverage
  'tables_with_rls', (
    SELECT COUNT(*) FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname = 'public' AND c.relrowsecurity = true
  ),
  'tables_without_rls_public', (
    SELECT COUNT(*) FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname = 'public' AND c.relrowsecurity = false
  ),
  'tables_without_rls_names', (
    SELECT COALESCE(json_agg(c.relname ORDER BY c.relname), '[]'::json)
    FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE c.relkind = 'r' AND n.nspname = 'public' AND c.relrowsecurity = false
  ),
  'total_rls_policies', (
    SELECT COUNT(*) FROM pg_policies WHERE schemaname = 'public'
  ),
  'rls_tables_zero_policies', (
    SELECT COUNT(*) FROM (
      SELECT c.relname FROM pg_class c
      JOIN pg_namespace n ON c.relnamespace = n.oid
      LEFT JOIN pg_policies p ON p.tablename = c.relname AND p.schemaname = n.nspname
      WHERE c.relkind = 'r' AND n.nspname = 'public'
        AND c.relrowsecurity = true AND p.policyname IS NULL
    ) t
  ),

  -- Invariant engine
  'invariant_violations_unresolved', (
    SELECT COUNT(*) FROM invariant_violations WHERE resolved = false
  ),
  'invariant_violations_total', (
    SELECT COUNT(*) FROM invariant_violations
  ),

  -- Alerts
  'alerts_unresolved', (
    SELECT COUNT(*) FROM brainops_alerts WHERE resolved = false
  ),
  'alerts_total', (
    SELECT COUNT(*) FROM brainops_alerts
  ),

  -- Key table stats
  'unified_memory_rows', (SELECT COUNT(*) FROM unified_ai_memory),
  'cc_tasks_count', (SELECT COUNT(*) FROM cc_tasks),

  -- Infrastructure
  'pgcron_active_jobs', (SELECT COUNT(*) FROM cron.job),
  'extensions_count', (SELECT COUNT(*) FROM pg_extension),
  'pg_version', current_setting('server_version'),
  'total_functions', (
    SELECT COUNT(*) FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'public'
  ),

  -- Timestamp
  'snapshot_ts', NOW()
);
