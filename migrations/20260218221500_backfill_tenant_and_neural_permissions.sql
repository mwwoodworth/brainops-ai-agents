BEGIN;

-- Stabilize backend neural graph writes (agent_orchestrator_v2).
GRANT SELECT, INSERT, UPDATE, DELETE ON public.neural_pathways TO app_backend_role;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'neural_pathways'
          AND policyname = 'backend_role_all_neural_pathways'
    ) THEN
        EXECUTE $p$
            CREATE POLICY backend_role_all_neural_pathways
            ON public.neural_pathways
            FOR ALL TO app_backend_role
            USING (true)
            WITH CHECK (true)
        $p$;
    END IF;
END$$;

-- Backfill historical rows that predate strict tenant RLS checks.
UPDATE public.ai_agent_executions
SET tenant_id = '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
WHERE tenant_id IS NULL;

UPDATE public.agent_health_status
SET tenant_id = '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
WHERE tenant_id IS NULL;

UPDATE public.ai_autonomous_tasks
SET tenant_id = '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
WHERE tenant_id IS NULL;

UPDATE public.task_notifications
SET tenant_id = '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
WHERE tenant_id IS NULL;

COMMIT;
