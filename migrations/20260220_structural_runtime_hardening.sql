BEGIN;

-- Restore learning/performance tables expected by ai_decision_tree.
CREATE TABLE IF NOT EXISTS public.ai_agent_performance (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project text,
    agent_name text,
    success_rate double precision,
    avg_response_time_ms double precision,
    total_interactions integer,
    performance_data jsonb,
    updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.ai_learning_records (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name text,
    learning_type text NOT NULL,
    description text NOT NULL,
    context jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence numeric DEFAULT 0.5,
    applied boolean DEFAULT false,
    applied_at timestamp,
    impact_score numeric,
    created_at timestamp DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_agent_performance_agent
    ON public.ai_agent_performance (agent_name);
CREATE INDEX IF NOT EXISTS idx_ai_agent_performance_name
    ON public.ai_agent_performance (agent_name);
CREATE INDEX IF NOT EXISTS idx_ai_agent_performance_project
    ON public.ai_agent_performance (project);
CREATE INDEX IF NOT EXISTS idx_ai_agent_performance_updated
    ON public.ai_agent_performance (updated_at);

CREATE INDEX IF NOT EXISTS idx_ai_learning_agent
    ON public.ai_learning_records (agent_name);
CREATE INDEX IF NOT EXISTS idx_ai_learning_applied
    ON public.ai_learning_records (applied);
CREATE INDEX IF NOT EXISTS idx_ai_learning_type
    ON public.ai_learning_records (learning_type);

GRANT SELECT, INSERT, UPDATE ON public.ai_agent_performance TO app_agent_role;
GRANT SELECT ON public.ai_agent_performance TO app_backend_role;
GRANT SELECT ON public.ai_agent_performance TO app_mcp_role;
GRANT ALL ON public.ai_agent_performance TO service_role;

GRANT SELECT, INSERT, UPDATE ON public.ai_learning_records TO app_agent_role;
GRANT SELECT ON public.ai_learning_records TO app_backend_role;
GRANT SELECT ON public.ai_learning_records TO app_mcp_role;
GRANT ALL ON public.ai_learning_records TO service_role;

-- Ensure tenant defaults/backfill on high-frequency runtime tables so inserts
-- without explicit tenant_id do not violate strict RLS checks.
DO $$
DECLARE
    target_tenant uuid := '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='ai_agent_executions' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.ai_agent_executions ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.ai_agent_executions SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='agent_execution_logs' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.agent_execution_logs ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.agent_execution_logs SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='agent_executions' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.agent_executions ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.agent_executions SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='task_execution_history' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.task_execution_history ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.task_execution_history SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='memory_verification_artifacts' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.memory_verification_artifacts ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.memory_verification_artifacts SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='live_brain_memories' AND column_name='tenant_id') THEN
        EXECUTE 'ALTER TABLE public.live_brain_memories ALTER COLUMN tenant_id SET DEFAULT current_tenant_id()';
        EXECUTE 'UPDATE public.live_brain_memories SET tenant_id = $1 WHERE tenant_id IS NULL' USING target_tenant;
    END IF;
END $$;

COMMIT;
