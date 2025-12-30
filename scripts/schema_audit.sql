-- Schema audit script for PostgreSQL
-- Usage:
--   psql "$DATABASE_URL" -X -f scripts/schema_audit.sql > schema_audit.out

\pset pager off

-- Context
SELECT current_database() AS database_name,
       current_user AS database_user,
       now() AS run_at;

-- 1) List all tables with estimated row counts and sizes
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       c.reltuples::bigint AS est_rows_reltuples,
       s.n_live_tup AS est_rows_live,
       s.n_dead_tup AS est_rows_dead,
       pg_total_relation_size(c.oid) AS total_bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY n.nspname, c.relname;

-- 2) Foreign key relationships
SELECT n.nspname AS schema_name,
       r.relname AS table_name,
       c.conname AS fk_name,
       c.confrelid::regclass AS referenced_table,
       pg_get_constraintdef(c.oid) AS fk_definition,
       c.convalidated AS validated,
       c.condeferrable AS deferrable,
       c.condeferred AS initially_deferred
FROM pg_constraint c
JOIN pg_class r ON r.oid = c.conrelid
JOIN pg_namespace n ON n.oid = r.relnamespace
WHERE c.contype = 'f'
ORDER BY n.nspname, r.relname, c.conname;

-- 2b) Foreign keys missing a supporting index on referencing columns
WITH fk AS (
  SELECT c.oid AS con_oid,
         c.conname,
         c.conrelid,
         c.conkey
  FROM pg_constraint c
  WHERE c.contype = 'f'
)
SELECT n.nspname AS schema_name,
       r.relname AS table_name,
       fk.conname AS fk_name,
       pg_get_constraintdef(fk.con_oid) AS fk_definition
FROM fk
JOIN pg_class r ON r.oid = fk.conrelid
JOIN pg_namespace n ON n.oid = r.relnamespace
WHERE NOT EXISTS (
  SELECT 1
  FROM pg_index i
  WHERE i.indrelid = fk.conrelid
    AND i.indisvalid
    AND i.indisready
    AND (i.indkey::int[])[1:array_length(fk.conkey, 1)] = fk.conkey
)
ORDER BY n.nspname, r.relname, fk.conname;

-- 3) Index inventory and usage
SELECT schemaname AS schema_name,
       relname AS table_name,
       indexrelname AS index_name,
       idx_scan,
       idx_tup_read,
       idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC, idx_tup_read DESC;

-- 3b) Unused indexes (since last stats reset)
SELECT schemaname AS schema_name,
       relname AS table_name,
       indexrelname AS index_name,
       idx_scan,
       pg_relation_size(indexrelid) AS index_bytes
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- 3c) Invalid or not-ready indexes
SELECT n.nspname AS schema_name,
       c.relname AS index_name,
       i.indisvalid AS is_valid,
       i.indisready AS is_ready,
       i.indisunique AS is_unique,
       pg_get_indexdef(i.indexrelid) AS index_def
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE NOT i.indisvalid OR NOT i.indisready
ORDER BY n.nspname, c.relname;

-- 4) Column inventory (for code usage comparison)
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       a.attname AS column_name,
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
       a.attnotnull AS not_null,
       a.atthasdef AS has_default
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY n.nspname, c.relname, a.attnum;

-- 5) Orphaned data (FK violations) - writes results to a temp table
CREATE TEMP TABLE audit_orphans (
  fk_name text,
  child_table text,
  parent_table text,
  orphan_count bigint
);

DO $$
DECLARE
  r record;
  join_clause text;
  not_null_clause text;
  parent_first text;
  parent_null_check text;
  sql text;
BEGIN
  FOR r IN
    SELECT c.oid AS con_oid,
           c.conname,
           c.conrelid,
           c.confrelid,
           c.conkey,
           c.confkey
    FROM pg_constraint c
    WHERE c.contype = 'f'
  LOOP
    SELECT string_agg(format('c.%I = p.%I', ca.attname, pa.attname), ' AND ' ORDER BY ck.ord),
           string_agg(format('c.%I IS NOT NULL', ca.attname), ' AND ' ORDER BY ck.ord),
           (array_agg(pa.attname ORDER BY ck.ord))[1]
      INTO join_clause, not_null_clause, parent_first
    FROM unnest(r.conkey) WITH ORDINALITY AS ck(attnum, ord)
    JOIN unnest(r.confkey) WITH ORDINALITY AS pk(attnum, ord)
      ON ck.ord = pk.ord
    JOIN pg_attribute ca
      ON ca.attrelid = r.conrelid AND ca.attnum = ck.attnum
    JOIN pg_attribute pa
      ON pa.attrelid = r.confrelid AND pa.attnum = pk.attnum;

    IF join_clause IS NULL OR parent_first IS NULL THEN
      CONTINUE;
    END IF;

    parent_null_check := format('p.%I IS NULL', parent_first);

    sql := format(
      'INSERT INTO audit_orphans (fk_name, child_table, parent_table, orphan_count)
       SELECT %L, %L, %L, COUNT(*)
       FROM %s c
       LEFT JOIN %s p ON %s
       WHERE %s AND %s;',
      r.conname,
      r.conrelid::regclass::text,
      r.confrelid::regclass::text,
      r.conrelid::regclass,
      r.confrelid::regclass,
      join_clause,
      not_null_clause,
      parent_null_check
    );

    EXECUTE sql;
  END LOOP;
END $$;

SELECT fk_name,
       child_table,
       parent_table,
       orphan_count
FROM audit_orphans
WHERE orphan_count > 0
ORDER BY orphan_count DESC, fk_name;

-- 6) RLS policies
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       c.relrowsecurity AS rls_enabled,
       c.relforcerowsecurity AS rls_forced
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY n.nspname, c.relname;

SELECT schemaname AS schema_name,
       tablename AS table_name,
       policyname AS policy_name,
       roles,
       cmd,
       permissive,
       qual,
       with_check
FROM pg_policies
ORDER BY schemaname, tablename, policyname;

SELECT n.nspname AS schema_name,
       c.relname AS table_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND c.relrowsecurity
  AND NOT EXISTS (
    SELECT 1
    FROM pg_policies p
    WHERE p.schemaname = n.nspname
      AND p.tablename = c.relname
  )
ORDER BY n.nspname, c.relname;

-- 7) Triggers (and disabled triggers)
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       t.tgname AS trigger_name,
       t.tgenabled AS enabled_flag,
       p.proname AS function_name
FROM pg_trigger t
JOIN pg_class c ON c.oid = t.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_proc p ON p.oid = t.tgfoid
WHERE NOT t.tgisinternal
ORDER BY n.nspname, c.relname, t.tgname;

SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       t.tgname AS trigger_name,
       t.tgenabled AS enabled_flag
FROM pg_trigger t
JOIN pg_class c ON c.oid = t.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE NOT t.tgisinternal
  AND t.tgenabled <> 'O'
ORDER BY n.nspname, c.relname, t.tgname;

-- 8) Functions and procedures
SELECT n.nspname AS schema_name,
       p.proname AS routine_name,
       p.prokind AS routine_kind,
       p.prosecdef AS security_definer,
       p.provolatile AS volatility,
       p.proparallel AS parallel_safety,
       pg_get_function_result(p.oid) AS result_type,
       pg_get_function_arguments(p.oid) AS arguments
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY n.nspname, p.proname;

SELECT n.nspname AS schema_name,
       p.proname AS routine_name,
       pg_get_function_arguments(p.oid) AS arguments
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND p.prosecdef
ORDER BY n.nspname, p.proname;

-- 9) Query performance (pg_stat_statements if available)
CREATE TEMP TABLE audit_top_queries (
  userid oid,
  dbid oid,
  query text,
  calls bigint,
  total_time double precision,
  mean_time double precision,
  rows bigint,
  shared_blks_hit bigint,
  shared_blks_read bigint,
  shared_blks_written bigint
);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements') THEN
    EXECUTE $q$
      INSERT INTO audit_top_queries
      SELECT userid,
             dbid,
             query,
             calls,
             total_time,
             mean_time,
             rows,
             shared_blks_hit,
             shared_blks_read,
             shared_blks_written
      FROM pg_stat_statements
      ORDER BY total_time DESC
      LIMIT 50
    $q$;
  ELSE
    RAISE NOTICE 'pg_stat_statements is not installed; top query stats are unavailable.';
  END IF;
END $$;

SELECT *
FROM audit_top_queries
ORDER BY total_time DESC;

SELECT pid,
       usename,
       application_name,
       state,
       query_start,
       now() - query_start AS duration,
       query
FROM pg_stat_activity
WHERE state IS DISTINCT FROM 'idle'
ORDER BY duration DESC;

-- 10) Data integrity checks
SELECT n.nspname AS schema_name,
       c.relname AS table_name
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND NOT EXISTS (
    SELECT 1
    FROM pg_constraint con
    WHERE con.conrelid = c.oid
      AND con.contype = 'p'
  )
ORDER BY n.nspname, c.relname;

SELECT n.nspname AS schema_name,
       r.relname AS table_name,
       c.conname AS constraint_name,
       c.contype AS constraint_type,
       pg_get_constraintdef(c.oid) AS constraint_def
FROM pg_constraint c
JOIN pg_class r ON r.oid = c.conrelid
JOIN pg_namespace n ON n.oid = r.relnamespace
WHERE c.convalidated = false
ORDER BY n.nspname, r.relname, c.conname;

SELECT n.nspname AS schema_name,
       c.relname AS index_name,
       i.indisvalid AS is_valid,
       i.indisready AS is_ready
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE NOT i.indisvalid OR NOT i.indisready
ORDER BY n.nspname, c.relname;
