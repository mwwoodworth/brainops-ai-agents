BEGIN;

-- 1) Restore missing revenue_audit_log table required by intelligent_task_orchestrator.
CREATE TABLE IF NOT EXISTS public.revenue_audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    action text NOT NULL,
    amount numeric(14,2) NOT NULL DEFAULT 0,
    task_id text,
    tenant_id uuid NOT NULL DEFAULT current_tenant_id(),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.revenue_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.revenue_audit_log FORCE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE ON public.revenue_audit_log TO app_agent_role;
GRANT SELECT ON public.revenue_audit_log TO app_backend_role;
GRANT SELECT ON public.revenue_audit_log TO app_mcp_role;
GRANT ALL ON public.revenue_audit_log TO service_role;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'revenue_audit_log'
          AND policyname = 'agent_all_revenue_audit_log'
    ) THEN
        EXECUTE $p$
            CREATE POLICY agent_all_revenue_audit_log
            ON public.revenue_audit_log
            FOR ALL TO app_agent_role
            USING (tenant_id = current_tenant_id())
            WITH CHECK (tenant_id = current_tenant_id())
        $p$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'revenue_audit_log'
          AND policyname = 'backend_select_revenue_audit_log'
    ) THEN
        EXECUTE $p$
            CREATE POLICY backend_select_revenue_audit_log
            ON public.revenue_audit_log
            FOR SELECT TO app_backend_role
            USING (tenant_id = current_tenant_id())
        $p$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'revenue_audit_log'
          AND policyname = 'mcp_select_revenue_audit_log'
    ) THEN
        EXECUTE $p$
            CREATE POLICY mcp_select_revenue_audit_log
            ON public.revenue_audit_log
            FOR SELECT TO app_mcp_role
            USING (tenant_id = current_tenant_id())
        $p$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'revenue_audit_log'
          AND policyname = 'service_role_all_revenue_audit_log'
    ) THEN
        EXECUTE $p$
            CREATE POLICY service_role_all_revenue_audit_log
            ON public.revenue_audit_log
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true)
        $p$;
    END IF;
END$$;

-- 2) Fix neural_pathways write permission gap for agent runtime.
GRANT SELECT, INSERT, UPDATE, DELETE ON public.neural_pathways TO app_agent_role;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'neural_pathways'
          AND policyname = 'agent_role_all_neural_pathways'
    ) THEN
        EXECUTE $p$
            CREATE POLICY agent_role_all_neural_pathways
            ON public.neural_pathways
            FOR ALL TO app_agent_role
            USING (true)
            WITH CHECK (true)
        $p$;
    END IF;
END$$;

-- 3) Hardening: replace unsafe current_setting(... )::uuid policy casts.
DO $$
DECLARE
    r RECORD;
    new_qual text;
    new_with_check text;
BEGIN
    FOR r IN
        SELECT schemaname, tablename, policyname, qual, with_check
        FROM pg_policies
        WHERE (
            (qual IS NOT NULL AND qual ILIKE '%app.current_tenant_id%' AND qual ILIKE '%::uuid%')
            OR
            (with_check IS NOT NULL AND with_check ILIKE '%app.current_tenant_id%' AND with_check ILIKE '%::uuid%')
        )
    LOOP
        new_qual := r.qual;
        new_with_check := r.with_check;

        IF new_qual IS NOT NULL THEN
            new_qual := replace(new_qual,
                '(tenant_id = (current_setting(''app.current_tenant_id''::text, true))::uuid)',
                '(tenant_id = current_tenant_id())');
            new_qual := replace(new_qual,
                '(tenant_id = (current_setting(''app.current_tenant_id''::text))::uuid)',
                '(tenant_id = current_tenant_id())');
            new_qual := replace(new_qual,
                '(tenant_id = (NULLIF(current_setting(''app.current_tenant_id''::text, true), ''''::text))::uuid)',
                '(tenant_id = current_tenant_id())');
        END IF;

        IF new_with_check IS NOT NULL THEN
            new_with_check := replace(new_with_check,
                '(tenant_id = (current_setting(''app.current_tenant_id''::text, true))::uuid)',
                '(tenant_id = current_tenant_id())');
            new_with_check := replace(new_with_check,
                '(tenant_id = (current_setting(''app.current_tenant_id''::text))::uuid)',
                '(tenant_id = current_tenant_id())');
            new_with_check := replace(new_with_check,
                '(tenant_id = (NULLIF(current_setting(''app.current_tenant_id''::text, true), ''''::text))::uuid)',
                '(tenant_id = current_tenant_id())');
        END IF;

        IF new_qual IS DISTINCT FROM r.qual THEN
            EXECUTE format('ALTER POLICY %I ON %I.%I USING (%s)',
                r.policyname, r.schemaname, r.tablename, new_qual);
        END IF;

        IF new_with_check IS DISTINCT FROM r.with_check THEN
            EXECUTE format('ALTER POLICY %I ON %I.%I WITH CHECK (%s)',
                r.policyname, r.schemaname, r.tablename, new_with_check);
        END IF;
    END LOOP;
END$$;

COMMIT;
