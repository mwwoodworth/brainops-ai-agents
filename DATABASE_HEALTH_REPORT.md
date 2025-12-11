# Database Health Report (Supabase `postgres.yomagoqdmxszqtdwuhab`)

Status: could not collect live metrics from this environment because DNS lookup to `aws-0-us-east-2.pooler.supabase.com` failed (network access is restricted here). Run the SQL below from a host that can reach Supabase to produce the actual report.

How to connect:
```
PGPASSWORD=$DB_PASSWORD psql -h aws-0-us-east-2.pooler.supabase.com -U postgres.yomagoqdmxszqtdwuhab -d postgres
```

## 1) Storage usage and table growth
- Current table sizes (tables+indexes), largest first:
```sql
SELECT schemaname,
       relname AS table_name,
       pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
       pg_size_pretty(pg_relation_size(relid)) AS table_size,
       pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size,
       n_live_tup AS approx_rows
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 30;
```
- Churn since stats reset (higher = faster growth or frequent rewrites):
```sql
SELECT schemaname,
       relname AS table_name,
       n_live_tup AS approx_rows,
       n_tup_ins + n_tup_upd + n_tup_del AS row_ops_since_reset,
       last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
FROM pg_stat_all_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY row_ops_since_reset DESC
LIMIT 30;
```
- (Optional) Create a snapshot table to measure growth over time and query deltas:
```sql
CREATE TABLE IF NOT EXISTS public.table_size_snapshots (
  collected_at timestamptz DEFAULT now(),
  schemaname text,
  table_name text,
  total_bytes bigint,
  table_bytes bigint,
  index_bytes bigint,
  approx_rows bigint
);

INSERT INTO public.table_size_snapshots (schemaname, table_name, total_bytes, table_bytes, index_bytes, approx_rows)
SELECT schemaname, relname,
       pg_total_relation_size(relid),
       pg_relation_size(relid),
       pg_total_relation_size(relid) - pg_relation_size(relid),
       n_live_tup
FROM pg_catalog.pg_statio_user_tables;

-- Fastest growth since prior snapshot
WITH latest AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY schemaname, table_name ORDER BY collected_at DESC) AS rn
  FROM public.table_size_snapshots
),
pair AS (
  SELECT cur.*, prev.total_bytes AS prev_total_bytes
  FROM latest cur
  JOIN latest prev
    ON cur.schemaname = prev.schemaname AND cur.table_name = prev.table_name
   AND cur.rn = 1 AND prev.rn = 2
)
SELECT schemaname, table_name,
       total_bytes - prev_total_bytes AS bytes_growth,
       pg_size_pretty(total_bytes - prev_total_bytes) AS pretty_growth,
       pg_size_pretty(total_bytes) AS current_size
FROM pair
WHERE prev_total_bytes IS NOT NULL
ORDER BY bytes_growth DESC
LIMIT 20;
```

## 2) Index usage and missing indexes
- Index efficiency and unused indexes:
```sql
SELECT schemaname, relname AS table_name, indexrelname AS index_name,
       idx_scan, idx_tup_read, idx_tup_fetch,
       pg_size_pretty(pg_relation_size(i.indexrelid)) AS index_size
FROM pg_stat_user_indexes si
JOIN pg_index i ON si.indexrelid = i.indexrelid
ORDER BY idx_scan ASC, pg_relation_size(i.indexrelid) DESC
LIMIT 50;
```
- Tables relying on sequential scans (candidate for indexing):
```sql
SELECT schemaname, relname AS table_name, seq_scan, idx_scan, n_live_tup
FROM pg_stat_all_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND seq_scan > 0
ORDER BY (seq_scan::numeric / GREATEST(idx_scan, 1)) DESC, n_live_tup DESC
LIMIT 30;
```
- Foreign keys without supporting indexes (should usually be indexed):
```sql
SELECT conrelid::regclass AS table_name,
       conname AS fk_name,
       confrelid::regclass AS references_table
FROM pg_constraint c
WHERE contype = 'f'
  AND NOT EXISTS (
    SELECT 1
    FROM pg_index i
    WHERE i.indrelid = c.conrelid
      AND i.indisvalid
      AND i.indkey::text = c.conkey::text
  );
```

## 3) Query performance (requires pg_stat_statements)
Ensure the extension is available:
```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```
- Most expensive queries by total time:
```sql
SELECT queryid, calls,
       total_exec_time/1000 AS total_seconds,
       mean_exec_time AS avg_ms,
       rows,
       query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```
- Slowest average (exclude rare queries):
```sql
SELECT queryid, calls, mean_exec_time AS avg_ms, stddev_exec_time AS stddev_ms,
       rows/calls AS rows_per_call,
       query
FROM pg_stat_statements
WHERE calls >= 20
ORDER BY mean_exec_time DESC
LIMIT 20;
```
- IO-heavy queries (buffer usage):
```sql
SELECT queryid, calls,
       shared_blks_hit, shared_blks_read, temp_blks_written,
       query
FROM pg_stat_statements
ORDER BY (shared_blks_read + shared_blks_hit) DESC
LIMIT 20;
```

## 4) Data integrity checks
- Invalid constraints (should be validated):
```sql
SELECT conname, conrelid::regclass AS table_name, contype
FROM pg_constraint
WHERE NOT convalidated;
```
- Orphaned rows per foreign key (generates checks you can run):
```sql
WITH fk AS (
  SELECT conname,
         conrelid::regclass AS child_table,
         confrelid::regclass AS parent_table,
         conkey AS child_cols,
         confkey AS parent_cols
  FROM pg_constraint
  WHERE contype = 'f'
)
SELECT format(
  'SELECT %L AS constraint_name, count(*) AS orphans FROM %I.%I c WHERE NOT EXISTS (SELECT 1 FROM %I.%I p WHERE %s);',
  conname,
  split_part(child_table::text, '.', 1), split_part(child_table::text, '.', 2),
  split_part(parent_table::text, '.', 1), split_part(parent_table::text, '.', 2),
  (SELECT string_agg(format('c.%I = p.%I', child_att.attname, parent_att.attname), ' AND ')
   FROM unnest(child_cols) WITH ORDINALITY AS c(col, ord)
   JOIN unnest(parent_cols) WITH ORDINALITY AS p(col, ord2) ON ord = ord2
   JOIN pg_attribute child_att ON child_att.attrelid = child_table AND child_att.attnum = c.col
   JOIN pg_attribute parent_att ON parent_att.attrelid = parent_table AND parent_att.attnum = p.col)
) AS orphan_check_sql
FROM fk;
-- Run the generated statements to get orphan counts per FK.
```

## 5) Storage by table vs indexes
```sql
SELECT schemaname,
       relname AS table_name,
       pg_size_pretty(pg_relation_size(relid)) AS table_size,
       pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 30;
```

## What to look for
- Fastest growth: tables with large `row_ops_since_reset` or biggest deltas between size snapshots.
- Missing indexes: tables with high `seq_scan` and low `idx_scan`; foreign keys listed by the missing-index query.
- Unused indexes: `idx_scan = 0` and sizable `index_size` are candidates to drop after verification.
- Slow queries: high `total_seconds` or `avg_ms`; correlate with sequential-scan tables and missing indexes.
- Integrity: any non-zero orphan counts or unvalidated constraints need cleanup and validation.
