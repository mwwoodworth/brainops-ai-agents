-- ============================================================
-- P1-TASKMATE-001: TaskMate Cross-Model Task Manager
-- Created: 2026-02-13
-- Run as: postgres (not agent_worker — no DDL permissions)
-- ============================================================

BEGIN;

-- Core tasks table
CREATE TABLE IF NOT EXISTS public.taskmate_tasks (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id       TEXT UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    priority      TEXT NOT NULL DEFAULT 'P2',
    status        TEXT NOT NULL DEFAULT 'open',
    owner         TEXT,
    blocked_by    TEXT,
    evidence      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at     TIMESTAMPTZ,
    tenant_id     UUID NOT NULL
);

-- Comments / audit trail
CREATE TABLE IF NOT EXISTS public.taskmate_comments (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id       TEXT NOT NULL,
    author        TEXT NOT NULL,
    body          TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id     UUID NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_taskmate_tasks_status ON public.taskmate_tasks(status);
CREATE INDEX IF NOT EXISTS idx_taskmate_tasks_priority ON public.taskmate_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_taskmate_tasks_owner ON public.taskmate_tasks(owner);
CREATE INDEX IF NOT EXISTS idx_taskmate_tasks_tenant ON public.taskmate_tasks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_taskmate_comments_task ON public.taskmate_comments(task_id);
CREATE INDEX IF NOT EXISTS idx_taskmate_comments_tenant ON public.taskmate_comments(tenant_id);

-- RLS
ALTER TABLE public.taskmate_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.taskmate_tasks FORCE ROW LEVEL SECURITY;
ALTER TABLE public.taskmate_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.taskmate_comments FORCE ROW LEVEL SECURITY;

-- RLS policies: tenant isolation
CREATE POLICY taskmate_tasks_tenant ON public.taskmate_tasks
    FOR ALL
    USING (tenant_id::text = current_setting('app.current_tenant_id', true));

CREATE POLICY taskmate_comments_tenant ON public.taskmate_comments
    FOR ALL
    USING (tenant_id::text = current_setting('app.current_tenant_id', true));

-- Agent worker access (service role already bypasses RLS)
CREATE POLICY taskmate_tasks_agent ON public.taskmate_tasks
    FOR ALL TO agent_worker
    USING (true)
    WITH CHECK (true);

CREATE POLICY taskmate_comments_agent ON public.taskmate_comments
    FOR ALL TO agent_worker
    USING (true)
    WITH CHECK (true);

-- Grant minimal permissions to agent_worker
GRANT SELECT, INSERT, UPDATE ON public.taskmate_tasks TO agent_worker;
GRANT SELECT, INSERT ON public.taskmate_comments TO agent_worker;

COMMIT;
