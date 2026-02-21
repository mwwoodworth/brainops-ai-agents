-- ═══════════════════════════════════════════════════════════════
-- CI GATE: RLS Coverage Check
-- Fails if any public-schema table (not on the known-safe list)
-- exists without RLS enabled.
--
-- Usage:
--   psql -v ON_ERROR_STOP=1 -f check_rls_coverage.sql
--
-- Exit code:
--   0 = all public tables protected
--   non-zero = RLS regression detected
-- ═══════════════════════════════════════════════════════════════

DO $$
DECLARE
  _known_safe TEXT[] := ARRAY['spatial_ref_sys'];  -- PostGIS static lookup
  _violators TEXT[];
  _count INT;
BEGIN
  SELECT ARRAY_AGG(c.relname ORDER BY c.relname)
  INTO _violators
  FROM pg_class c
  JOIN pg_namespace n ON c.relnamespace = n.oid
  WHERE c.relkind = 'r'
    AND n.nspname = 'public'
    AND c.relrowsecurity = false
    AND c.relname != ALL(_known_safe);

  _count := COALESCE(array_length(_violators, 1), 0);

  IF _count > 0 THEN
    RAISE EXCEPTION 'RLS REGRESSION: % public table(s) without RLS: %',
      _count, array_to_string(_violators, ', ');
  ELSE
    RAISE NOTICE 'RLS CHECK PASSED: all public tables protected (% known exceptions: %)',
      array_length(_known_safe, 1), array_to_string(_known_safe, ', ');
  END IF;
END $$;

-- Also verify every RLS-enabled public table has at least 1 policy
DO $$
DECLARE
  _orphans TEXT[];
  _count INT;
BEGIN
  SELECT ARRAY_AGG(c.relname ORDER BY c.relname)
  INTO _orphans
  FROM pg_class c
  JOIN pg_namespace n ON c.relnamespace = n.oid
  LEFT JOIN pg_policies p ON p.tablename = c.relname AND p.schemaname = n.nspname
  WHERE c.relkind = 'r'
    AND n.nspname = 'public'
    AND c.relrowsecurity = true
    AND p.policyname IS NULL;

  _count := COALESCE(array_length(_orphans, 1), 0);

  IF _count > 0 THEN
    RAISE EXCEPTION 'RLS ORPHAN: % table(s) have RLS enabled but ZERO policies: %',
      _count, array_to_string(_orphans, ', ');
  ELSE
    RAISE NOTICE 'RLS POLICY CHECK PASSED: all RLS-enabled tables have policies';
  END IF;
END $$;
